#!/usr/bin/env python3
"""Post-process a fused hand npz into a stabilised one: gate -> smooth -> rigidify.

Nothing here re-runs a model. These are CPU passes over an existing run's output, so a stabilised
delivery can be rebuilt without touching the GPU.

Three passes, in this order, each targeting a defect that was MEASURED on the type 1 run of
episode_047 rather than assumed:

  1. depth gate      EgoForce placed 11.46% of detections at or behind the camera (wrist Z <= 0 on
                     10.66% of rows, min -0.419 m). The per-joint rate was near-uniform across all
                     21 joints, which says the ROOT TRANSLATION failed, not the articulation. Such a
                     row is not a noisy measurement, it is an impossible one, and it projects to
                     absurd pixels (reproj p95 was 7.5e7 px). Dropping these first also stops them
                     forming spurious tracks: 106 tracks -> 85.
  2. temporal_smooth Zero-phase local weighted polynomial fit. Upstream measured per-frame estimator
                     noise at 83% of visible jitter; the estimator is re-run from scratch each frame
                     with no memory of the last one.
  3. rigidify        Fits a rigid-bone kinematic model and smooths THE POSE PARAMETERS, so every
                     output is an exactly-rigid hand. Baseline bone-length CV was 5.1% median.

Passes 2 and 3 are vendored verbatim in `fusion/stabilise/` - see PROVENANCE.md, including the
unresolved licence question. This file holds all the adaptation to our npz schema.

The stabilised keypoints are NOT raw per-frame measurements and must not be presented as such. The
originals survive in `kp2d_raw` / `kp3d_cam_raw_pre` (smoothing) and `kp2d_prerigid` /
`kp3d_cam_prerigid` (rigidify), so every pass is auditable and reversible.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(HERE, 'stabilise')

# Keys that are run-level scalars/matrices, not per-detection rows. Listed explicitly because a
# shape[0] test alone would misfire on a run with very few detections (K is (3,3)).
NOT_ROW_KEYED = {
    'K', 'width', 'height', 'fps', 'step', 'fusion_mode', 'pp_rebase',
    'n_dropped_dupes', 'n_pnp_fallback', 'keypoint_order',
}

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True, help='fused <stem>_hand21_keypoints.npz')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--stem', default=None, help='output stem (default: derived from --npz)')
    p.add_argument('--min-z', type=float, default=0.05,
                   help='a detection needs repair if ANY joint is nearer than this (metres)')
    p.add_argument('--max-bridge', type=int, default=90,
                   help='widest two-sided depth interpolation, frames')
    p.add_argument('--one-sided', type=int, default=10,
                   help='a single-sided depth anchor is accepted only this close, frames')
    p.add_argument('--h', type=float, default=4.0, help='smoothing half-window, frames')
    p.add_argument('--order', type=int, default=2, help='smoothing polynomial order')
    p.add_argument('--rigid-h', type=float, default=16.0, help='pose-smoothing half-window, frames')
    p.add_argument('--drop-instead-of-repair', action='store_true',
                   help='old behaviour: delete unrepairable rows rather than bridging their depth')
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-rigidify', action='store_true')
    return p.parse_args()


def find_sibling_3d(fused_path):
    """The stage-1b npz next to the fused one. It carries EgoForce's own 2D head."""
    hits = [p for p in glob.glob(os.path.join(os.path.dirname(fused_path), '*_3d_keypoints.npz'))]
    return hits[0] if len(hits) == 1 else None


def repair_rows(src, dst, sibling_3d, min_z, max_bridge=90, one_sided=10):
    """Repair detections whose 3D lift failed, rather than deleting them.

    An earlier version simply dropped any row with a joint nearer than `min_z`. That was wrong, and
    the review of episode_048 showed why: over frames 1249-1306 EgoForce detected the right hand in
    91/91 frames, but its wrist solved to -0.101 m - behind the lens - so the gate deleted 58
    consecutive frames and left a 1.93 s hole in an otherwise continuous track.

    Only the DEPTH was broken. The detection was fine: on the 369 bad rows of that clip, EgoForce's
    own 2D head (`kp2d_head`, which is not derived from the 3D lift) puts 100% of joints on the image
    and 100% inside the detector box, while the projection of the broken 3D manages 42% and 23%.
    Throwing the row away discards good pixels to be rid of bad metres.

    So: keep the 2D head, interpolate the per-joint depth from the nearest healthy rows of the same
    hand, and back-project. The row is then labelled `lifted_2d` with `depth_measured=False`, which is
    the mono-pipeline's existing vocabulary for exactly this - 2D we trust, depth we borrowed.

    A row is repaired when either test fails, both physical rather than tuned:
      * any joint nearer than `min_z`;
      * the lift's wrist projects off the image while the head's wrist does not - a direct
        self-contradiction. This is what produced the visible "sudden drift": rows that passed the
        depth test, projected hundreds of pixels outside the frame, and snapped back.

    Bridging follows the shape the reference study found (fusion/stabilise/PROVENANCE.md): two-sided
    interpolation is better than one-sided at any gap length, so a single-sided anchor is accepted
    only within `one_sided` frames. Beyond `max_bridge`, or with no anchor at all, the row is dropped
    - depth cannot be invented from nothing.
    """
    z = np.load(src, allow_pickle=True)
    n = len(z['frame_idx'])
    fr = z['frame_idx'].astype(int)
    ir = z['is_right_wilor'].astype(int)
    kp3 = z['kp3d_cam'].astype(float).copy()
    K = np.asarray(z['K'], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = int(z['width']), int(z['height'])

    head = None
    if sibling_3d and os.path.exists(sibling_3d):
        s3 = np.load(sibling_3d, allow_pickle=True)
        if 'kp2d_head' in s3.files:
            lut = {(int(f), int(i)): h for f, i, h
                   in zip(s3['frame_idx'], s3['is_right'], s3['kp2d_head'].astype(float))}
            head = np.stack([lut.get((int(fr[r]), int(ir[r])), np.full((21, 2), np.nan))
                             for r in range(n)])

    lift2d = z['kp2d'].astype(float)
    off = lambda p: (p[..., 0] < 0) | (p[..., 0] >= W) | (p[..., 1] < 0) | (p[..., 1] >= H)
    bad = (kp3[..., 2] < min_z).any(1)
    if head is not None:
        contradiction = off(lift2d[:, 0]) & ~off(head[:, 0]) & ~np.isnan(head[:, 0, 0])
        bad = bad | contradiction

    stats = {'rows_in': n, 'bad': int(bad.sum()), 'repaired': 0, 'dropped': 0,
             'no_head': 0, 'no_anchor': 0}
    keep = np.ones(n, bool)

    if head is not None:
        for side in (0, 1):
            rows = np.where(ir == side)[0]
            if rows.size == 0:
                continue
            rows = rows[np.argsort(fr[rows])]
            good = rows[~bad[rows]]
            if good.size == 0:
                keep[rows[bad[rows]]] = False
                stats['no_anchor'] += int(bad[rows].sum())
                continue
            gf = fr[good]
            for r in rows[bad[rows]]:
                if np.isnan(head[r]).any():
                    keep[r] = False
                    stats['no_head'] += 1
                    continue
                f = fr[r]
                pre = good[gf < f][-1] if (gf < f).any() else None
                post = good[gf > f][0] if (gf > f).any() else None
                # Pick the anchors once; depth and the 2D offset must use the SAME ones, or a row
                # can take its depth from one end and its offset from both.
                if pre is not None and post is not None and (fr[post] - fr[pre]) <= max_bridge:
                    w = (f - fr[pre]) / max(fr[post] - fr[pre], 1)
                    Zj = (1 - w) * kp3[pre, :, 2] + w * kp3[post, :, 2]
                elif pre is not None and (f - fr[pre]) <= one_sided:
                    Zj, post = kp3[pre, :, 2], None
                elif post is not None and (fr[post] - f) <= one_sided:
                    Zj, pre = kp3[post, :, 2], None
                else:
                    keep[r] = False
                    stats['no_anchor'] += 1
                    continue
                # Use the head's pixels as they are. Correcting them by the head-vs-lift offset at
                # the gap boundaries looked obviously right and measured worse, so it is recorded
                # here rather than re-attempted. On episode_048, against this version's 123
                # frame-to-frame jumps over 200 px and a bone-length CV of 0.0101:
                #   per-joint offset  101 jumps, CV 0.0219, 18 tracks -> 29   (warps the skeleton)
                #   one global offset 133 jumps, CV 0.0405                    (barely beats raw)
                # The offset removes the boundary step but at the cost of the hand's own shape, and
                # the smoother then fragments the track it was supposed to hold together.
                u, v = head[r, :, 0], head[r, :, 1]
                kp3[r] = np.stack([(u - cx) * Zj / fx, (v - cy) * Zj / fy, Zj], axis=-1)
                stats['repaired'] += 1
    else:
        keep = ~bad
        stats['no_head'] = int(bad.sum())

    stats['dropped'] = int((~keep).sum())
    out = {}
    for k in z.files:
        v = z[k]
        row_keyed = (k not in NOT_ROW_KEYED and isinstance(v, np.ndarray)
                     and v.ndim >= 1 and v.shape[0] == n)
        out[k] = v[keep] if row_keyed else v
    out['kp3d_cam'] = kp3[keep].astype(np.float32)
    # kp2d is the projection of kp3d, so a repaired row's 2D becomes its own head by construction.
    P = kp3[keep]
    out['kp2d'] = np.stack([P[..., 0] * fx / P[..., 2] + cx,
                            P[..., 1] * fy / P[..., 2] + cy], axis=-1).astype(np.float32)
    rep = bad[keep]
    if 'source' in z.files:
        src_arr = np.asarray(z['source']).astype(object)[keep]
        src_arr[rep] = 'lifted_2d'
        out['source'] = src_arr.astype(str)
    if 'depth_measured' in z.files:
        dm = np.asarray(z['depth_measured']).astype(bool)[keep]
        dm[rep] = False
        out['depth_measured'] = dm
    out['depth_repaired'] = rep
    out['repair_min_z'] = np.float32(min_z)
    np.savez_compressed(dst, **out)
    return stats


def metrics(path, min_z=0.05):
    """Jitter, bone-length CV and motion retention. Independent of the passes' own reporting."""
    d = np.load(path, allow_pickle=True)
    kp3 = d['kp3d_cam'].astype(float)
    kp2 = d['kp2d'].astype(float)
    fr = d['frame_idx'].astype(int)
    ir = d['is_right_wilor'].astype(int)
    ok = (kp3[..., 2] >= min_z).all(1)

    out = {'rows': int(len(kp3)), 'rows_clean': int(ok.sum())}
    for side, nm in ((0, 'left'), (1, 'right')):
        m = (ir == side) & ok
        f = fr[m]
        o = np.argsort(f)
        f = f[o]
        k2, k3 = kp2[m][o], kp3[m][o]
        idx = {int(v): i for i, v in enumerate(f)}
        a2, a3, disp = [], [], {h: [] for h in (1, 5, 20, 30)}
        for i, fv in enumerate(f):
            a, c = idx.get(int(fv) - 1), idx.get(int(fv) + 1)
            if a is not None and c is not None:
                a2.append(np.linalg.norm(k2[c] - 2 * k2[i] + k2[a], axis=-1))
                a3.append(np.linalg.norm(k3[c] - 2 * k3[i] + k3[a], axis=-1))
            for h in disp:
                j = idx.get(int(fv) + h)
                if j is not None:
                    disp[h].append(float(np.linalg.norm(k2[j, 0] - k2[i, 0])))
        e = {}
        if a2:
            e['accel_2d_median_px'] = float(np.median(np.concatenate(a2)))
            e['accel_3d_median_m'] = float(np.median(np.concatenate(a3)))
        for h, v in disp.items():
            e[f'wrist_disp_p90_h{h}_px'] = float(np.percentile(v, 90)) if v else None
        e['n'] = int(m.sum())
        out[nm] = e

    if ok.any():
        P = kp3[ok]
        L = np.linalg.norm(P[:, [a for a, _ in EDGES]] - P[:, [b for _, b in EDGES]], axis=-1)
        cv = L.std(0) / np.maximum(L.mean(0), 1e-9)
        out['bone_cv_median'] = float(np.median(cv))
        out['bone_cv_max'] = float(cv.max())
    return out


def run(cmd):
    print('  $', ' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'stage failed ({r.returncode}):\n{r.stdout}\n{r.stderr}')
    return r.stdout.strip()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = a.stem or os.path.basename(a.npz).replace('_hand21_keypoints.npz', '')
    final = os.path.join(a.out, f'{stem}_hand21_keypoints.npz')

    stats = {'input': os.path.abspath(a.npz), 'params': vars(a),
             'before': metrics(a.npz, a.min_z), 'stages': []}
    print(f'[stabilise {stem}] {stats["before"]["rows"]} rows in')

    tmpd = tempfile.mkdtemp(prefix='stabilise_')
    cur = a.npz

    if not a.no_gate:
        dst = os.path.join(tmpd, 'repaired.npz')
        sib = find_sibling_3d(a.npz)
        if sib is None:
            print('  repair: no sibling *_3d_keypoints.npz - falling back to DROPPING bad rows. '
                  'Copy the 3D npz next to the fused one to enable repair.')
        st = repair_rows(cur, dst, None if a.drop_instead_of_repair else sib,
                         a.min_z, a.max_bridge, a.one_sided)
        kept = st['rows_in'] - st['dropped']
        print(f"  repair: {st['bad']} bad of {st['rows_in']} -> repaired {st['repaired']}, "
              f"dropped {st['dropped']} (no anchor {st['no_anchor']}, no 2D head {st['no_head']}); "
              f"kept {kept}/{st['rows_in']} ({100.0 * kept / st['rows_in']:.2f}%)")
        stats['stages'].append({'stage': 'depth_repair', 'sibling_3d': sib, **st})
        cur = dst

    if not a.no_smooth:
        dst = os.path.join(tmpd, 'smoothed.npz')
        o = run([sys.executable, os.path.join(STAB, 'temporal_smooth.py'), '--npz', cur,
                 '--out', dst, '--h', str(a.h), '--order', str(a.order)])
        stats['stages'].append({'stage': 'temporal_smooth', **json.loads(o)})
        print(f'  smooth: {o.replace(chr(10), " ")}')
        cur = dst

    if not a.no_rigidify:
        dst = os.path.join(tmpd, 'rigid.npz')
        o = run([sys.executable, os.path.join(STAB, 'rigidify.py'), '--npz', cur,
                 '--out', dst, '--h', str(a.rigid_h)])
        stats['stages'].append({'stage': 'rigidify', **json.loads(o)})
        print(f'  rigidify: {o.replace(chr(10), " ")}')
        cur = dst

    np.savez_compressed(final, **{k: v for k, v in np.load(cur, allow_pickle=True).items()})
    stats['after'] = metrics(final, a.min_z)
    stats['output'] = final
    if stats['after']['rows'] > stats['before']['rows_clean']:
        stats['motion_retention_caveat'] = (
            "before/after are NOT the same row set: `before` is measured on rows that already "
            "passed the depth test, while repair puts previously-failing rows back. Retention "
            "above 100% is that changed denominator, not extra motion. The jitter and bone-CV "
            "figures are unaffected - they are per-row properties.")

    sp = os.path.join(a.out, f'{stem}_stabilise_stats.json')
    with open(sp, 'w') as fh:
        json.dump(stats, fh, indent=2)

    b, af = stats['before'], stats['after']
    print(f'\n[stabilise {stem}] wrote {final}')
    print(f'  {"metric":28s} {"before":>12s} {"after":>12s}   change')
    for side in ('left', 'right'):
        for key, fmt in (('accel_2d_median_px', '.3f'), ('accel_3d_median_m', '.5f')):
            x, y = b[side].get(key), af[side].get(key)
            if x and y:
                print(f'  {side + " " + key:28s} {x:12{fmt}} {y:12{fmt}}   {100 * (y - x) / x:+6.1f}%')
    for key in ('bone_cv_median', 'bone_cv_max'):
        print(f'  {key:28s} {b[key]:12.5f} {af[key]:12.5f}   {100 * (af[key] - b[key]) / b[key]:+6.1f}%')
    print('\n  motion retention (wrist p90 displacement, should approach 100% at long horizon):')
    for side in ('left', 'right'):
        row = '    ' + f'{side:6s}'
        for h in (1, 5, 20, 30):
            x, y = b[side].get(f'wrist_disp_p90_h{h}_px'), af[side].get(f'wrist_disp_p90_h{h}_px')
            row += f'  h{h}={100 * y / x:5.1f}%' if x and y else f'  h{h}=  n/a'
        print(row)
    print(f'\n  stats -> {sp}')


if __name__ == '__main__':
    main()
