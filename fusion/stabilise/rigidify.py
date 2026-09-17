# VENDORED - DO NOT EDIT IN PLACE.
# Source: https://github.com/Maiemdiab/egocentric-hand-stabilisation
# Commit: 2ca531b   Vendored: 2026-09-11
# Unmodified except for this header. See fusion/stabilise/PROVENANCE.md.
#!/usr/bin/env python3
"""Final stage for C: exact bone rigidity, and articulation smoothed in POSE space.

Measured motivation. Splitting the residual vibration into the wrist (global) and the fingers
relative to the wrist (articulation), over 70 clips:

                       global    articulation
    C (rescue+smooth)  0.0563       0.0560
    MINT               0.0361       0.0142      <- 3.9x steadier in articulation

MINT wins the articulation half because it emits MANO parameters: a low-dimensional articulated
model physically cannot make one fingertip shimmer independently of its neighbours. Our `fuse()`
lets all 21 points slide, so it can, and does.

`kinematic.py` gives us the same structural property -- a rigid palm plus fixed-length links, 38
parameters -- so this pass fits that model to each frame and then smooths THE PARAMETERS rather
than the keypoints. Smoothing pose instead of points is what removes the shimmer: any smoothed
parameter vector still maps to an exactly-rigid hand, whereas smoothing points does not.

Two representations are deliberately not smoothed as raw numbers:
  * the palm rotation is smoothed as a rotation MATRIX and re-orthonormalised (SVD), because a
    rotation vector flips sign near pi and would tear;
  * each link is smoothed as a unit DIRECTION and renormalised, because the (theta, phi) angles
    wrap at +-pi and averaging across the wrap points the finger backwards.

Rigidified rows are flagged in `rigid`; the pre-rigid values are kept in `kp2d_prerigid` /
`kp3d_cam_prerigid` so the stage is auditable and reversible.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def out_keys(z): return set(z.files)


HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from kinematic import (CHAINS, NB, PALM_IDX, canonical_hand, forward, fit_frame, _init,
                       _dirs_from_angles, rodrigues, _rotvec, _reproj)      # noqa: E402
from temporal_smooth import build_tracks, smooth_series                     # noqa: E402


def forward_rt(R, t, dirs, canon):
    """Same map as kinematic.forward, but from an explicit rotation matrix and unit directions."""
    P = np.empty((21, 3))
    P[PALM_IDX] = (R @ canon['palm_local'].T).T + t
    u = dirs * canon['lens'][:, None]
    for k, (a, b) in enumerate(CHAINS):
        P[b] = P[a] + u[k]
    return P


def project(P, K):
    Z = np.maximum(P[:, 2], 1e-4)
    return np.stack([K[0, 0] * P[:, 0] / Z + K[0, 2], K[1, 1] * P[:, 1] / Z + K[1, 2]], -1)


def fit_and_smooth(k2, k3, raw, K, canon_fallback, h=5.0, order=2, smooth=True, **kw):
    """Fit the 38-param model to every row of one track, then smooth pose and re-forward."""
    canon = canonical_hand(raw) or canon_fallback
    if canon is None: return None, None, None
    n = len(k2)
    Rs = np.empty((n, 3, 3)); ts = np.empty((n, 3)); ds = np.empty((n, NB, 3)); rep = np.empty(n)
    prev = None
    for i in range(n):
        Zw = raw[i, :, 2].copy() if np.isfinite(raw[i]).all() else k3[i, :, 2].copy()
        P0 = forward(prev, canon) if prev is not None else k3[i]
        if not np.isfinite(P0).all(): P0 = k3[i]
        P, x = fit_frame(k2[i], Zw, K, canon, P0, **kw)
        if prev is not None and _reproj(P, k2[i], K) > 12.0:
            P2, x2 = fit_frame(k2[i], Zw, K, canon, k3[i], **kw)      # re-seed cold if it went badly
            if _reproj(P2, k2[i], K) < _reproj(P, k2[i], K): P, x = P2, x2
        prev = x
        Rs[i] = rodrigues(x[:3]); ts[i] = x[3:6]
        ds[i] = _dirs_from_angles(x[6:].reshape(NB, 2))
        rep[i] = _reproj(P, k2[i], K)
    return (Rs, ts, ds), canon, rep


def smooth_pose(t, Rs, ts, ds, w0, h=5.0, order=2):
    Rf = smooth_series(t, Rs.reshape(len(t), 9), w0, h, order).reshape(-1, 3, 3)
    out_R = np.empty_like(Rf)
    for i in range(len(Rf)):                       # nearest true rotation to the smoothed matrix
        U, _, Vt = np.linalg.svd(Rf[i])
        d = np.sign(np.linalg.det(U @ Vt))
        out_R[i] = U @ np.diag([1, 1, d]) @ Vt
    out_t = smooth_series(t, ts, w0, h, order)
    df = smooth_series(t, ds.reshape(len(t), NB * 3), w0, h, order).reshape(-1, NB, 3)
    nrm = np.linalg.norm(df, axis=-1, keepdims=True)
    out_d = np.where(nrm > 1e-9, df / np.maximum(nrm, 1e-9), ds)
    return out_R, out_t, out_d


def bone_cv(P3, side=None):
    """Spread of each bone's length. Grouped by hand, because a left and a right hand are allowed
    to differ in size -- pooling them reports their difference as if it were instability."""
    if side is None: side = np.zeros(len(P3), int)
    out = []
    for s in np.unique(side):
        P = P3[side == s]
        if len(P) < 2: continue
        L = np.stack([np.linalg.norm(P[:, b] - P[:, a], axis=-1) for a, b in CHAINS], 1)
        m = L.mean(0); ok = m > 1e-6
        if ok.any(): out.append(float(np.mean(L[:, ok].std(0) / m[ok]) * 100))
    return float(np.mean(out)) if out else float('nan')


def jitter_split(fi, k2):
    tr = build_tracks(fi, k2); G, A = [], []
    for idx in tr:
        f = fi[idx]; K = k2[idx]
        m = {int(x): j for j, x in enumerate(f)}
        run = [j for j, x in enumerate(f) if (int(x) - 1) in m and (int(x) + 1) in m]
        for j in run:
            a, b = m[int(f[j]) - 1], m[int(f[j]) + 1]
            p = float(np.linalg.norm(K[j][5] - K[j][17]))
            if p < 1e-6: continue
            G.append(float(np.linalg.norm(K[b][0] - 2 * K[j][0] + K[a][0])) / p)
            rel = lambda q: K[q] - K[q][0]
            A.append(float(np.median(np.linalg.norm(rel(b) - 2 * rel(j) + rel(a), axis=-1))) / p)
    return (float(np.median(G)) if G else float('nan')), (float(np.median(A)) if A else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--h', type=float, default=16.0,
                    help='pose-smoothing half-window in frames. Measured on episode_045: h=5 is a '
                         'wash (the per-frame fit adds as much noise as smoothing removes), h=16 '
                         'gives jitter 0.030/0.020 keeping 85%%/94%% of p90/p99 motion, h=24 gives '
                         '0.023/0.013 keeping 68%%/84%%. MINT for reference is 0.036/0.014 but keeps '
                         'only 60%%/36%% -- it buys its stability by suppressing real motion.')
    ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--no-pose-smooth', action='store_true',
                    help='fit only. NOT recommended: the per-frame least-squares fit is noisier '
                         'than its input (0.048 -> 0.081 articulation), so the smoothing is what '
                         'makes this stage a win rather than a regression.')
    ap.add_argument('--canon', choices=('per-clip', 'per-track'), default='per-clip',
                    help="per-clip estimates ONE hand shape per side for the whole clip, so bone "
                         "lengths stay consistent across tracks as well as within them. per-track "
                         "lets each track pick its own, which leaves ~1.3%% clip-level spread.")
    ap.add_argument('--sigma-z', type=float, default=0.006)
    ap.add_argument('--keep-unrigidified', action='store_true',
                    help='keep drawn rows the fit could not reach (tracks shorter than 4 frames). '
                         'They are a fraction of a percent but they still breathe by ~14%%, so by '
                         'default they are dropped and the delivered set is exactly rigid.')
    ap.add_argument('--max-nfev', type=int, default=120)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True); f = z.files
    fi = z['frame_idx'].astype(int)
    k2 = z['kp2d'].astype(float).copy(); k3 = z['kp3d_cam'].astype(float).copy()
    raw = np.full_like(k3, np.nan)
    if 'kp3d_cam_wilor_raw' in f:
        r = z['kp3d_cam_wilor_raw'].astype(float)
        # an older bridge pass appended rows without growing this array; tolerate the short case
        raw[:len(r)] = r[:len(raw)]
    K = np.asarray(z['K'], float)
    if 'kept' in f and 'hand' in f:
        drawn = z['kept'].astype(bool) & np.isin(z['hand'].astype(int), (0, 1))
    elif 'drawn' in f: drawn = z['drawn'].astype(bool)
    else: drawn = np.ones(len(fi), bool)
    src = z['source'] if 'source' in f else np.array(['fused'] * len(fi))

    idx = np.where(drawn)[0]
    g0, a0 = jitter_split(fi[idx], k2[idx])
    hnd0 = z['hand'].astype(int) if 'hand' in f else np.zeros(len(fi), int)
    cv0 = bone_cv(k3[idx], hnd0[idx])

    # a clip-wide fallback shape, for tracks whose own WiLoR rows are all missing (e.g. all bridged)
    fin = raw[idx][np.isfinite(raw[idx]).all(axis=(1, 2))]
    canon_fb = canonical_hand(fin) if len(fin) >= 3 else None
    # one canonical hand per side: the wearer's bones do not change length between tracks
    hnd = z['hand'].astype(int) if 'hand' in f else np.zeros(len(fi), int)
    canon_side = {}
    for s_ in (0, 1):
        sel = idx[hnd[idx] == s_]
        r = raw[sel][np.isfinite(raw[sel]).all(axis=(1, 2))] if len(sel) else np.zeros((0, 21, 3))
        canon_side[s_] = canonical_hand(r) if len(r) >= 3 else canon_fb

    k2o, k3o = k2.copy(), k3.copy()
    rigid = np.zeros(len(fi), bool)
    nt = 0
    from temporal_smooth import SRC_W
    for tr in build_tracks(fi[idx], k2[idx]):
        gi = idx[tr]
        if len(gi) < 4: continue
        side = int(np.median(hnd[gi])) if a.canon == 'per-clip' else None
        fb = canon_side.get(side, canon_fb) if a.canon == 'per-clip' else canon_fb
        rawin = np.full_like(raw[gi], np.nan) if a.canon == 'per-clip' else raw[gi]
        res, canon, rep = fit_and_smooth(k2[gi], k3[gi], rawin, K, fb,
                                         sigma_z=a.sigma_z, max_nfev=a.max_nfev)
        if res is None: continue
        Rs, ts, ds = res
        t = fi[gi].astype(float)
        if not a.no_pose_smooth and len(gi) >= 5:
            w0 = np.array([SRC_W.get(str(s), 0.6) for s in src[gi]])
            Rs, ts, ds = smooth_pose(t, Rs, ts, ds, w0, a.h, a.order)
        for j, g in enumerate(gi):
            P = forward_rt(Rs[j], ts[j], ds[j], canon)
            k3[g] = P; k2[g] = project(P, K)
        rigid[gi] = True; nt += 1

    n_drop = 0
    if not a.keep_unrigidified and 'kept' in out_keys(z):
        bad = drawn & ~rigid
        n_drop = int(bad.sum())
        if n_drop:
            kept = z['kept'].astype(bool).copy(); kept[bad] = False
            drawn = drawn & rigid; idx = np.where(drawn)[0]
    g1, a1 = jitter_split(fi[idx], k2[idx])
    cv1 = bone_cv(k3[idx], hnd0[idx])
    out = {k: z[k] for k in f}
    if n_drop: out['kept'] = kept
    out['kp2d_prerigid'] = k2o.astype(np.float32); out['kp3d_cam_prerigid'] = k3o.astype(np.float32)
    out['kp2d'] = k2.astype(np.float32); out['kp3d_cam'] = k3.astype(np.float32)
    out['rigid'] = rigid
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(tracks=nt, rows=int(rigid.sum()), dropped_unrigidified=n_drop,
                          bone_cv_pct_before=round(cv0, 4), bone_cv_pct_after=round(cv1, 4),
                          jitter_global_before=round(g0, 4), jitter_global_after=round(g1, 4),
                          jitter_artic_before=round(a0, 4), jitter_artic_after=round(a1, 4)), indent=2))


if __name__ == '__main__':
    main()
