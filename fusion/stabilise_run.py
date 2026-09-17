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
                   help='drop a detection if ANY joint is nearer than this (metres)')
    p.add_argument('--h', type=float, default=4.0, help='smoothing half-window, frames')
    p.add_argument('--order', type=int, default=2, help='smoothing polynomial order')
    p.add_argument('--rigid-h', type=float, default=16.0, help='pose-smoothing half-window, frames')
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--no-smooth', action='store_true')
    p.add_argument('--no-rigidify', action='store_true')
    return p.parse_args()


def gate_depth(src, dst, min_z):
    """Drop detections with any joint at or behind `min_z`. Returns (n_in, n_kept)."""
    z = np.load(src, allow_pickle=True)
    n = len(z['frame_idx'])
    keep = (z['kp3d_cam'][..., 2] >= min_z).all(1)
    out = {}
    for k in z.files:
        v = z[k]
        row_keyed = (k not in NOT_ROW_KEYED and isinstance(v, np.ndarray)
                     and v.ndim >= 1 and v.shape[0] == n)
        out[k] = v[keep] if row_keyed else v
    out['depth_gate_min_z'] = np.float32(min_z)
    out['depth_gate_dropped'] = np.int32(n - int(keep.sum()))
    np.savez_compressed(dst, **out)
    return n, int(keep.sum())


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
        dst = os.path.join(tmpd, 'gated.npz')
        n, kept = gate_depth(cur, dst, a.min_z)
        print(f'  gate: kept {kept}/{n} ({100.0 * kept / n:.2f}%), dropped {n - kept}')
        stats['stages'].append({'stage': 'depth_gate', 'rows_in': n, 'rows_out': kept,
                                'dropped': n - kept, 'min_z_m': a.min_z})
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
