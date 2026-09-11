# VENDORED - DO NOT EDIT IN PLACE.
# Source: https://github.com/Maiemdiab/egocentric-hand-stabilisation
# Commit: 2ca531b   Vendored: 2026-09-11
# Unmodified except for this header. See fusion/stabilise/PROVENANCE.md.
#!/usr/bin/env python3
"""Exact-rigidity hand fit.

The delivered `fuse()` lets all 21 keypoints slide independently, so bone lengths breathe by 12-17%.
Alternating projection (push lengths, pull back to the rays) only ever reaches a compromise, because
the observed rays and the true bone lengths are frequently not simultaneously satisfiable.

So we do not constrain the skeleton -- we PARAMETERISE it, and rigidity becomes structural:

    palm  {0,5,9,13,17}  a rigid body, 6 DoF, in the canonical shape of this track's own WiLoR hand
    thumb 1-2-3-4        a chain hanging off the wrist, fixed link lengths, free directions
    index 6-7-8          a chain off MCP 5      (likewise middle / ring / pinky)

38 parameters. Bone lengths and palm geometry cannot deviate no matter what the optimiser does, so
bone-length CV is exactly zero and the only thing left to trade is reprojection error, which we
minimise against the observed 2D with a weak prior holding depth where WiLoR put it.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

PALM_IDX = [0, 5, 9, 13, 17]
# (parent_point, child_point) for every non-palm link, in evaluation order
CHAINS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb hangs off the wrist
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
NB = len(CHAINS)                              # 16 free links
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-9: return np.eye(3)
    k = r / th; Kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(th)*Kx + (1-np.cos(th))*(Kx@Kx)


def canonical_hand(raw):
    """Canonical palm shape (local frame) + the 16 link lengths, from a track's raw WiLoR hands."""
    ok = np.isfinite(raw).all(axis=(1, 2))
    if ok.sum() < 3: return None
    R = raw[ok]
    lens = np.array([np.median(np.linalg.norm(R[:, b] - R[:, a], axis=-1)) for a, b in CHAINS])
    # palm in a canonical frame: centre, then align each sample to the first by Procrustes
    P = R[:, PALM_IDX, :]
    P = P - P.mean(1, keepdims=True)
    ref = P[0]
    acc = np.zeros_like(ref)
    for s in P:
        U, _, Vt = np.linalg.svd(s.T @ ref)
        d = np.sign(np.linalg.det(U @ Vt))
        Rr = U @ np.diag([1, 1, d]) @ Vt
        acc += s @ Rr
    palm_local = acc / len(P)
    return dict(palm_local=palm_local, lens=lens)


def _dirs_from_angles(ang):
    th = ang[:, 0]; ph = ang[:, 1]
    st = np.sin(th)
    return np.stack([st*np.cos(ph), st*np.sin(ph), np.cos(th)], 1)


def forward(params, canon):
    """38 params -> (21,3) points. Bone lengths are exact by construction."""
    r = params[:3]; t = params[3:6]; ang = params[6:].reshape(NB, 2)
    P = np.empty((21, 3))
    P[PALM_IDX] = (rodrigues(r) @ canon['palm_local'].T).T + t
    u = _dirs_from_angles(ang) * canon['lens'][:, None]
    for k, (a, b) in enumerate(CHAINS):
        P[b] = P[a] + u[k]
    return P


def _init(P0, canon):
    """Seed from a (non-rigid) observed skeleton: Procrustes the palm, read off link directions."""
    q = P0[PALM_IDX]; c = q.mean(0); qc = q - c
    U, _, Vt = np.linalg.svd(canon['palm_local'].T @ qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    rv, _ = _rotvec(R)
    ang = np.empty((NB, 2))
    for k, (a, b) in enumerate(CHAINS):
        v = P0[b] - P0[a]; n = np.linalg.norm(v)
        v = v / n if n > 1e-9 else np.array([0., 0., 1.])
        ang[k] = [np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0])]
    return np.concatenate([rv, c, ang.ravel()])


def _rotvec(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(tr)
    if th < 1e-8: return np.zeros(3), 0.0
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(th))
    return v * th, th


def fit_frame(m2d, Zw, K, canon, P0, sigma_2d=5.0, sigma_z=0.012, max_nfev=150):
    """Fit the rigid-palm / fixed-length skeleton to observed 2D, holding depth where it shipped.

    The depth prior targets the DELIVERED z, not a new estimate, so this pass changes the shape of
    the skeleton without moving the dataset's metric scale -- the known ~12% WiLoR depth bias is
    left exactly as delivered, to be handled separately and disclosed."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    zok = np.isfinite(Zw) & (Zw > 1e-4)

    def resid(p):
        P = forward(p, canon)
        Z = np.maximum(P[:, 2], 1e-4)
        u = fx * P[:, 0] / Z + cx; v = fy * P[:, 1] / Z + cy
        r2 = np.concatenate([(u - m2d[:, 0]), (v - m2d[:, 1])]) / sigma_2d
        rz = ((P[zok, 2] - Zw[zok]) / sigma_z) if zok.any() else np.zeros(0)
        return np.concatenate([r2, rz])

    p0 = _init(P0, canon)
    sol = least_squares(resid, p0, method='lm', max_nfev=max_nfev, xtol=1e-6, ftol=1e-6)
    return forward(sol.x, canon), sol.x


def fit_track(k2, k3, raw, K, canon_fallback, warm_start=True, **kw):
    """Fit every row of one track. Returns points, per-row reprojection error, and exactness check."""
    canon = canonical_hand(raw) or canon_fallback
    n = len(k2)
    out = np.empty((n, 21, 3)); rep = np.empty(n)
    prev = None
    for i in range(n):
        # depth target: WiLoR's own where it exists, otherwise the delivered z (borrowed profile).
        # Either way it is the value that already shipped -- this pass does not re-scale the data.
        Zw = raw[i, :, 2].copy() if np.isfinite(raw[i]).all() else k3[i, :, 2].copy()
        # seed from the previous frame's solution when we have one: same objective, faster and
        # better-conditioned convergence. It is an initialiser only, never a temporal filter.
        P0 = forward(prev, canon) if (warm_start and prev is not None) else k3[i]
        P, x = fit_frame(k2[i], Zw, K, canon, P0, **kw)
        if warm_start and prev is not None and _reproj(P, k2[i], K) > 12.0:
            P2, x2 = fit_frame(k2[i], Zw, K, canon, k3[i], **kw)   # re-seed cold if it went badly
            if _reproj(P2, k2[i], K) < _reproj(P, k2[i], K): P, x = P2, x2
        prev = x
        out[i] = P; rep[i] = _reproj(P, k2[i], K)
    return out, rep, canon


def _reproj(P, m, K):
    Z = np.maximum(P[:, 2], 1e-4)
    u = K[0,0]*P[:,0]/Z + K[0,2]; v = K[1,1]*P[:,1]/Z + K[1,2]
    return float(np.median(np.hypot(u - m[:,0], v - m[:,1])))
