# VENDORED - DO NOT EDIT IN PLACE.
# Source: https://github.com/Maiemdiab/egocentric-hand-stabilisation
# Commit: 2ca531b   Vendored: 2026-09-11
# Unmodified except for this header. See fusion/stabilise/PROVENANCE.md.
#!/usr/bin/env python3
"""Zero-phase temporal smoothing of hand keypoints.

Measured cause of residual "vibration": ~90% of the frame-to-frame motion of the drawn skeleton
happens with the SAME estimator on both frames. It is not source switching -- it is that the
estimator is re-run from scratch on every frame with no memory of the previous one. Neither per-frame
detection nor the handedness rescue filters that.

This does, with four properties that matter:

  zero phase   Local weighted quadratic fit evaluated AT the sample, using frames on both sides.
               Offline, so acausal is free, and a symmetric window introduces no lag -- unlike an
               EMA or any causal filter, which would drag the skeleton behind the hand.
  gap aware    Fitting happens over real frame indices inside one track. A detection gap widens the
               local neighbourhood rather than being silently treated as adjacent, so the filter
               never pulls a hand across a hole it did not observe.
  weighted     Per-source reliability: `fused` (both models agree, measured depth) is trusted more
               than `lifted_2d` (borrowed depth) or `wilor` (no MediaPipe corroboration).
  robust       Two Tukey biweight passes, so one bad frame bends the curve toward itself instead of
               dragging its neighbours with it.

2D and 3D are smoothed independently and BOTH are flagged. They are deliberately not tied together:
for `fused` rows the upstream kp2d is MediaPipe's pixels, which sit ~14 px better on the hand than the
projection of its own 3D, so deriving one from the other would trade a real alignment gain for
internal tidiness.

Smoothed rows are marked in `smoothed` and the original values kept in `kp2d_raw` / `kp3d_cam_raw_pre`
so this is auditable and reversible. A smoothed keypoint is NOT a raw per-frame measurement and must
not be presented as one.
"""
from __future__ import annotations
import argparse, json
import numpy as np

WRIST = 0
SRC_W = {'fused': 1.0, 'wilor': 0.6, 'wilor_pnpfail': 0.6, 'lifted_2d': 0.45}


def build_tracks(fi, k2, gate=150.0, maxgap=10, min_len=4):
    order = np.argsort(fi, kind='stable'); tk = []
    for gi in order:
        f = int(fi[gi]); xy = k2[gi, 0]; best, bd = None, gate
        for t in tk:
            if 0 < f - t['lf'] <= maxgap:
                d = float(np.hypot(*(t['lxy'] - xy)))
                if d < bd: best, bd = t, d
        if best is None: tk.append(dict(lxy=xy, lf=f, idx=[gi]))
        else: best['lxy'] = xy; best['lf'] = f; best['idx'].append(gi)
    return [np.array(sorted(t['idx'], key=lambda i: int(fi[i]))) for t in tk if len(t['idx']) >= min_len]


def smooth_series(t, Y, w0, h=4.0, order=2, robust_passes=2, tukey_c=3.0, max_gap=10):
    """Local weighted polynomial fit evaluated at each sample. t = frame indices, Y = (n, d)."""
    n = len(t)
    out = Y.copy()
    if n < 3: return out
    for i in range(n):
        # neighbourhood in FRAME distance, not sample count, and never across a big hole
        lo = i
        while lo > 0 and t[i] - t[lo - 1] <= h and t[lo] - t[lo - 1] <= max_gap: lo -= 1
        hi = i
        while hi < n - 1 and t[hi + 1] - t[i] <= h and t[hi + 1] - t[hi] <= max_gap: hi += 1
        sl = slice(lo, hi + 1)
        tt = t[sl].astype(float) - float(t[i])
        yy = Y[sl]
        m = len(tt)
        if m < order + 1:
            continue
        u = np.abs(tt) / (h + 1e-9)
        w = (1.0 - np.clip(u, 0, 1) ** 3) ** 3                    # tricube
        w = w * w0[sl]
        V = np.vander(tt, order + 1)                              # fit y(t) around t_i
        for _ in range(robust_passes + 1):
            W = np.sqrt(np.maximum(w, 1e-12))[:, None]
            try:
                coef, *_ = np.linalg.lstsq(V * W, yy * W, rcond=None)
            except np.linalg.LinAlgError:
                break
            pred = V @ coef
            resid = np.linalg.norm(yy - pred, axis=-1)
            s = np.median(resid) * 1.4826 + 1e-9
            w = w * (1.0 - np.clip(resid / (tukey_c * s), 0, 1) ** 2) ** 2
        out[i] = coef[-1]                                          # value at tt = 0
    return out


def smooth_npz(z, drawn, h=4.0, order=2, tukey_c=3.0):
    fi = z['frame_idx'].astype(int)
    k2 = z['kp2d'].astype(float).copy()
    k3 = z['kp3d_cam'].astype(float).copy()
    src = z['source'] if 'source' in z.files else np.array(['fused'] * len(fi))
    k2o, k3o = k2.copy(), k3.copy()
    smoothed = np.zeros(len(fi), bool)

    idx_drawn = np.where(drawn)[0]
    if len(idx_drawn) < 4:
        return k2, k3, smoothed, dict(tracks=0, rows=0)
    tracks = build_tracks(fi[idx_drawn], k2[idx_drawn])
    ntr = 0
    for tr in tracks:
        gi = idx_drawn[tr]
        t = fi[gi]
        w0 = np.array([SRC_W.get(str(s), 0.6) for s in src[gi]])
        for j in range(21):
            k2[gi, j] = smooth_series(t, k2o[gi, j], w0, h, order, 2, tukey_c)
            k3[gi, j] = smooth_series(t, k3o[gi, j], w0, h, order, 2, tukey_c)
        smoothed[gi] = True
        ntr += 1
    return k2, k3, smoothed, dict(tracks=ntr, rows=int(smoothed.sum()))


def jitter(fi, k2):
    tr = build_tracks(fi, k2); out = []
    for idx in tr:
        f = fi[idx]; K = k2[idx]; m = {x: j for j, x in enumerate(f)}
        for j, x in enumerate(f):
            if (x - 1) in m and (x + 1) in m:
                p = float(np.linalg.norm(K[j][5] - K[j][17]))
                if p > 1e-6:
                    out.append(float(np.median(np.linalg.norm(K[m[x+1]] - 2*K[j] + K[m[x-1]], axis=-1)) / p))
    return float(np.median(out)) if out else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--h', type=float, default=4.0, help='half-window in FRAMES (window ~2h+1)')
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--tukey', type=float, default=3.0)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    f = z.files
    fi = z['frame_idx'].astype(int)
    if 'kept' in f and 'hand' in f:
        drawn = z['kept'].astype(bool) & np.isin(z['hand'].astype(int), (0, 1))
    elif 'drawn' in f:
        drawn = z['drawn'].astype(bool)
    else:
        drawn = np.ones(len(fi), bool)

    k2_before = z['kp2d'].astype(float)
    j0 = jitter(fi[drawn], k2_before[drawn])
    k2, k3, smoothed, st = smooth_npz(z, drawn, a.h, a.order, a.tukey)
    j1 = jitter(fi[drawn], k2[drawn])

    # lag check: cross-correlate raw vs smoothed wrist path; a symmetric filter must peak at 0
    lag = 0.0
    tr = build_tracks(fi[drawn], k2_before[drawn])
    if tr:
        big = max(tr, key=len); gi = np.where(drawn)[0][big]
        a0 = k2_before[gi][:, WRIST, 0]; a1 = k2[gi][:, WRIST, 0]
        a0 = a0 - a0.mean(); a1 = a1 - a1.mean()
        if len(a0) > 20:
            cc = np.correlate(a0, a1, 'full'); lag = float(np.argmax(cc) - (len(a0) - 1))

    out = {k: z[k] for k in f}
    out['kp2d_raw'] = k2_before.astype(np.float32)
    out['kp3d_cam_raw_pre'] = z['kp3d_cam'].astype(np.float32)
    out['kp2d'] = k2.astype(np.float32)
    out['kp3d_cam'] = k3.astype(np.float32)
    out['smoothed'] = smoothed
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(tracks=st['tracks'], rows_smoothed=st['rows'],
                          jitter_before=round(j0, 4), jitter_after=round(j1, 4),
                          reduction_pct=round(100 * (j0 - j1) / j0, 1) if j0 else None,
                          lag_frames=lag), indent=2))


if __name__ == '__main__':
    main()
