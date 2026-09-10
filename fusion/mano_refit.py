"""Articulated MANO refit: fit EgoForce's hand to RTMPose's 2D landmarks.

This is the part of Experiment 2 that the existing rigid-PnP fusion cannot do.

``fuse_2d_3d.py::pnp_refit`` keeps WiLoR's root-relative skeleton **rigid** and solves for a single
rotation + translation that lands it on MediaPipe's pixels. That is the right thing to do if you only
trust the 2D globally, and it guarantees the bone lengths stay MANO-consistent - but it means a
wrongly bent finger stays wrongly bent no matter how confident the 2D observation is. Two models run,
and their finger estimates are never jointly optimised.

Here we optimise the MANO **parameters** instead: global orientation, the 15 joint rotations, shape
and translation. So the articulation itself can move to satisfy the image evidence, while the hand
stays a hand by construction because every candidate is a MANO sample.

Objective, per detection::

    L =  w_reproj  * Σ_j  c_j · huber( ‖ π(J_j) − x_j ‖ )      confidence-weighted image evidence
       + w_depth   * ‖ t − t₀ ‖²                               keep EgoForce's metric placement
       + w_pose    * mean( (θ − θ₀)² )                         stay near EgoForce's articulation
       + w_orient  * mean( (r − r₀)² )
       + w_beta    * mean( (β − β₀)² )                         keep EgoForce's bone lengths
       + w_limit   * mean( relu(‖θ_j‖ − θ_max)² )              reject implausible joint rotations

with ``x_j``/``c_j`` the RTMPose landmark and its confidence, ``π`` the pinhole projection through
the same ``K`` the 3D stage stored, and subscript ``₀`` EgoForce's prediction.

Two notes on honesty:

* The **depth term is what keeps the fit metric.** A 2D reprojection loss is scale-depth degenerate:
  a hand twice as large at twice the depth projects identically. RTMPose contributes nothing to
  depth, so the translation must stay anchored to EgoForce, which is the model that actually resolved
  it. ``w_depth`` is therefore a load-bearing prior, not a regulariser to tune away.
* The **joint limits here are a coarse magnitude cap**, not a per-DOF anatomical range. MANO's
  per-joint axis-angle frames are not clean flexion axes, so a real anatomical limit set would have
  to be derived and validated separately. The load-bearing plausibility constraint is the MANO
  parameterisation plus ``w_pose`` pulling toward EgoForce's (already plausible) prediction.

Rotations are optimised in the 6D representation (Zhou et al.) and converted with
``rotation_6d_to_axis_angle_direct`` - the same conversion ``models/limb_model.py`` uses, so the
numerics match the forward pass EgoForce itself ran.

This module needs only torch + smplx + the MANO files. No TensorRT, no mmdet, no CUDA - so the refit
can be iterated on CPU against npz files produced earlier on a GPU box.
"""

import os
import sys
from dataclasses import dataclass, field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from fusion.topology import BONES, NUM_HAND_JOINTS, WRIST
from utils.rotations import rotation_6d_to_axis_angle_direct

N_MANO_JOINTS = 15          # MANO articulated joints, 3 per finger
N_SHAPE = 10


@dataclass
class RefitWeights:
    """Loss weights. Defaults are a starting point, NOT tuned on real footage - see plan_of_action.md."""

    reproj: float = 1.0
    depth: float = 250.0        # on metres², so 250 * (1 cm)² = 0.025 in loss units
    pose: float = 4.0
    orient: float = 8.0
    beta: float = 2.0
    limit: float = 20.0
    limit_max_angle: float = 1.75      # radians, ~100 deg per-joint rotation magnitude cap
    huber_delta_px: float = 6.0
    min_conf: float = 0.30             # RTMPose landmarks below this contribute nothing

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RefitResult:
    """Per-detection outcome. Arrays are all (N, ...) and aligned with the input order."""

    joints3d: np.ndarray            # (N,21,3) refit camera-space joints, metres
    uv: np.ndarray                  # (N,21,2) their projection through K
    betas: np.ndarray               # (N,10)
    global_orient6: np.ndarray      # (N,6)
    hand_pose6: np.ndarray          # (N,90)
    transl: np.ndarray              # (N,3)
    residual_px: np.ndarray         # (N,) confidence-weighted median reprojection error, after
    residual_px_init: np.ndarray    # (N,) the same metric for EgoForce's unrefit prediction
    bone_change: np.ndarray         # (N,) max |Δbone| / bone vs the EgoForce prediction
    depth_change_m: np.ndarray      # (N,) |Δ wrist depth| in metres
    iters: int = 0
    weights: dict = field(default_factory=dict)


def project_pinhole_torch(points, K):
    """Differentiable pinhole projection of (..., 3) camera-frame points."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = points[..., 2].clamp(min=1e-4)
    u = fx * points[..., 0] / z + cx
    v = fy * points[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def _huber(dist, delta):
    """Huber on the pixel DISTANCE (not per-coordinate): linear past `delta`, quadratic inside."""
    quad = 0.5 * dist.pow(2)
    lin = delta * (dist - 0.5 * delta)
    return torch.where(dist <= delta, quad, lin)


def _weighted_median(values, weights):
    """Per-row weighted median of (N, J) values. Rows with no weight yield NaN."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for i in range(values.shape[0]):
        w = weights[i]
        keep = w > 0
        if not keep.any():
            continue
        v = values[i][keep]
        ww = w[keep]
        order = np.argsort(v)
        v, ww = v[order], ww[order]
        cum = np.cumsum(ww)
        out[i] = v[np.searchsorted(cum, 0.5 * cum[-1])]
    return out


def _bone_lengths_torch(joints):
    a = torch.as_tensor([e[0] for e in BONES], dtype=torch.long, device=joints.device)
    b = torch.as_tensor([e[1] for e in BONES], dtype=torch.long, device=joints.device)
    return torch.linalg.norm(joints[:, a] - joints[:, b], dim=-1)


class ManoRefiner:
    """Batched articulated refit of EgoForce MANO parameters against 2D landmarks."""

    def __init__(self, mano_path, device='cpu', weights=None):
        from models.mano_layer import MANOHandModel

        self.device = torch.device(device)
        self.weights = weights or RefitWeights()
        # use_pose_pca=False matches the demo pipeline: 45 axis-angle pose parameters, not PCA.
        self.mano = MANOHandModel(mano_path, device=str(self.device), use_pose_pca=False)

    def _forward(self, betas, go6, hp6, transl, is_right):
        """MANO forward from 6D-rotation parameters. Returns (B,21,3) camera-space joints."""
        b = betas.shape[0]
        go_aa = rotation_6d_to_axis_angle_direct(go6.reshape(b, 6))
        hp_aa = rotation_6d_to_axis_angle_direct(hp6.reshape(b * N_MANO_JOINTS, 6))
        hp_aa = hp_aa.reshape(b, N_MANO_JOINTS * 3)
        global_xfrom = torch.cat([go_aa, transl.reshape(b, 3)], dim=1)
        _, joints = self.mano.forward_kinematics(
            shape_params=betas.reshape(b, N_SHAPE),
            joint_angles=hp_aa,
            global_xfrom=global_xfrom,
            is_right_hand=is_right,
        )
        return joints, hp_aa.reshape(b, N_MANO_JOINTS, 3)

    def refit(self, init, target_uv, target_conf, is_right, K,
              iters=80, lr=0.02, chunk=256, verbose=False):
        """Refit every detection against its 2D landmarks.

        Parameters
        ----------
        init : dict with 'betas' (N,10), 'global_orient6' (N,6), 'hand_pose6' (N,90),
            'transl' (N,3) - EgoForce's prediction, used as both initialisation and prior.
        target_uv : (N,21,2) RTMPose landmarks in source-image pixels.
        target_conf : (N,21) RTMPose per-landmark confidence.
        is_right : (N,) 0 = left, 1 = right.
        K : (3,3) the intrinsics the 3D stage stored.
        """
        n = init['betas'].shape[0]
        if n == 0:
            empty = lambda *s: np.zeros((0, *s), dtype=np.float32)  # noqa: E731
            return RefitResult(empty(NUM_HAND_JOINTS, 3), empty(NUM_HAND_JOINTS, 2), empty(N_SHAPE),
                               empty(6), empty(N_MANO_JOINTS * 6), empty(3),
                               np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                               iters=iters, weights=self.weights.as_dict())

        out = {k: [] for k in ('joints3d', 'uv', 'betas', 'go6', 'hp6', 'transl',
                               'res', 'res0', 'bone', 'depth')}

        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            res = self._refit_chunk(
                {k: v[start:stop] for k, v in init.items()},
                target_uv[start:stop], target_conf[start:stop], is_right[start:stop],
                K, iters=iters, lr=lr, verbose=verbose,
            )
            for key in out:
                out[key].append(res[key])
            if verbose:
                print(f'  [refit] {stop}/{n} detections')

        cat = {k: np.concatenate(v, axis=0) for k, v in out.items()}
        return RefitResult(
            joints3d=cat['joints3d'], uv=cat['uv'], betas=cat['betas'],
            global_orient6=cat['go6'], hand_pose6=cat['hp6'], transl=cat['transl'],
            residual_px=cat['res'], residual_px_init=cat['res0'],
            bone_change=cat['bone'], depth_change_m=cat['depth'],
            iters=iters, weights=self.weights.as_dict(),
        )

    def _refit_chunk(self, init, target_uv, target_conf, is_right, K, iters, lr, verbose):
        w = self.weights
        dev = self.device
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731

        betas0 = t(init['betas'])
        go6_0 = t(init['global_orient6'])
        hp6_0 = t(init['hand_pose6'])
        transl0 = t(init['transl'])
        is_right_t = t(np.asarray(is_right).reshape(-1), torch.bool)
        K_t = t(np.asarray(K, dtype=np.float64))
        uv_t = t(target_uv)
        conf_t = t(target_conf).clamp(min=0.0)
        # A landmark below min_conf carries no evidence. Zeroing rather than downweighting keeps a
        # confidently-wrong low-confidence point from dragging a finger.
        conf_t = torch.where(conf_t >= w.min_conf, conf_t, torch.zeros_like(conf_t))
        # Guard against non-finite targets (a producer wrote NaN for an undetected joint).
        conf_t = torch.where(torch.isfinite(uv_t).all(dim=-1), conf_t, torch.zeros_like(conf_t))
        uv_t = torch.nan_to_num(uv_t, nan=0.0, posinf=0.0, neginf=0.0)

        betas = betas0.clone().requires_grad_(True)
        go6 = go6_0.clone().requires_grad_(True)
        hp6 = hp6_0.clone().requires_grad_(True)
        transl = transl0.clone().requires_grad_(True)

        with torch.no_grad():
            joints_init, _ = self._forward(betas0, go6_0, hp6_0, transl0, is_right_t)
            uv_init = project_pinhole_torch(joints_init, K_t)
            bones_init = _bone_lengths_torch(joints_init)
            depth_init = joints_init[:, WRIST, 2]

        optimizer = torch.optim.Adam([betas, go6, hp6, transl], lr=lr)
        for step in range(iters):
            optimizer.zero_grad(set_to_none=True)
            joints, hp_aa = self._forward(betas, go6, hp6, transl, is_right_t)
            uv = project_pinhole_torch(joints, K_t)

            dist = torch.linalg.norm(uv - uv_t, dim=-1)
            reproj = (conf_t * _huber(dist, w.huber_delta_px)).sum(dim=1)
            reproj = reproj / conf_t.sum(dim=1).clamp(min=1e-6)

            depth_pen = (transl - transl0).pow(2).sum(dim=1)
            pose_pen = (hp6 - hp6_0).pow(2).mean(dim=1)
            orient_pen = (go6 - go6_0).pow(2).mean(dim=1)
            beta_pen = (betas - betas0).pow(2).mean(dim=1)
            mag = torch.linalg.norm(hp_aa, dim=-1)
            limit_pen = torch.relu(mag - w.limit_max_angle).pow(2).mean(dim=1)

            loss = (w.reproj * reproj + w.depth * depth_pen + w.pose * pose_pen
                    + w.orient * orient_pen + w.beta * beta_pen + w.limit * limit_pen)
            loss.sum().backward()
            optimizer.step()

            if verbose and (step + 1) % max(1, iters // 4) == 0:
                print(f'    [refit] iter {step + 1}/{iters} loss={float(loss.mean()):.4f} '
                      f'reproj={float(reproj.mean()):.3f}')

        with torch.no_grad():
            joints, _ = self._forward(betas, go6, hp6, transl, is_right_t)
            uv = project_pinhole_torch(joints, K_t)
            bones = _bone_lengths_torch(joints)
            bone_change = ((bones - bones_init).abs()
                           / bones_init.clamp(min=1e-6)).max(dim=1).values
            depth_change = (joints[:, WRIST, 2] - depth_init).abs()

        conf_np = conf_t.detach().cpu().numpy()
        dist_np = torch.linalg.norm(uv - uv_t, dim=-1).detach().cpu().numpy()
        dist0_np = torch.linalg.norm(uv_init - uv_t, dim=-1).detach().cpu().numpy()

        return dict(
            joints3d=joints.detach().cpu().numpy().astype(np.float32),
            uv=uv.detach().cpu().numpy().astype(np.float32),
            betas=betas.detach().cpu().numpy().astype(np.float32),
            go6=go6.detach().cpu().numpy().astype(np.float32),
            hp6=hp6.detach().cpu().numpy().astype(np.float32),
            transl=transl.detach().cpu().numpy().astype(np.float32),
            res=_weighted_median(dist_np, conf_np),
            res0=_weighted_median(dist0_np, conf_np),
            bone=bone_change.detach().cpu().numpy().astype(np.float64),
            depth=depth_change.detach().cpu().numpy().astype(np.float64),
        )


def accept_refit(result, max_reproj_px=20.0, max_bone_change=0.15, max_depth_change_m=0.05,
                 require_improvement=True):
    """Decide which refits to keep.

    Mirrors the spirit of ``pnp_refit``'s ``--pnp-max-px`` gate, plus two guards a rigid fit did not
    need: articulation is free to move here, so a refit that "wins" on reprojection by stretching the
    bones or sliding the hand in depth has stopped being a measurement of the same hand.

    Returns a boolean mask and a dict of per-reason counts.
    """
    res = np.asarray(result.residual_px, dtype=np.float64)
    res0 = np.asarray(result.residual_px_init, dtype=np.float64)
    bone = np.asarray(result.bone_change, dtype=np.float64)
    depth = np.asarray(result.depth_change_m, dtype=np.float64)

    finite = np.isfinite(result.joints3d).all(axis=(1, 2)) & np.isfinite(res)
    positive_z = (result.joints3d[..., 2] > 1e-6).all(axis=1)
    within_px = res <= max_reproj_px
    bone_ok = bone <= max_bone_change
    depth_ok = depth <= max_depth_change_m
    improved = np.ones_like(within_px) if not require_improvement else (res <= res0 + 1e-9)

    ok = finite & positive_z & within_px & bone_ok & depth_ok & improved
    reasons = dict(
        total=int(len(ok)), accepted=int(ok.sum()),
        rejected_nonfinite=int((~finite).sum()),
        rejected_behind_camera=int((finite & ~positive_z).sum()),
        rejected_reproj=int((finite & positive_z & ~within_px).sum()),
        rejected_bone_change=int((finite & positive_z & within_px & ~bone_ok).sum()),
        rejected_depth_change=int((finite & positive_z & within_px & bone_ok & ~depth_ok).sum()),
        rejected_no_improvement=int((finite & positive_z & within_px & bone_ok & depth_ok
                                     & ~improved).sum()),
    )
    return ok, reasons
