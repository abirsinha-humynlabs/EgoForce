#!/usr/bin/env python3
"""Self-test for everything in ``fusion/`` that does not need a GPU, a checkpoint or MANO.

Run it before spending GPU time:

    python fusion/selftest.py

It needs only numpy, torch (CPU) and opencv - no TensorRT, no mmdet, no mmpose, no model weights.
That covers the parts most likely to be quietly wrong: the keypoint topology, the stride-aware
coverage and gap arithmetic, the duplicate-suppression rules, the 2D/3D matching, the projection
maths and the accept/reject accounting.

What it deliberately does NOT cover, because it cannot without the real environment:

* EgoForce inference itself (needs CUDA + TensorRT + weights)
* RTMPose inference (needs mmpose + the Hand5 checkpoint)
* the MANO forward pass and therefore the refit's actual convergence (needs the MANO pkl files)

So a green run means "the plumbing and the arithmetic are right", not "the pipeline produces good
hands". See plan_of_action.md for what still has to be checked on a GPU box.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import tempfile

import numpy as np

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ------------------------------------------------------------------ topology


@check('topology: 21 joints, 20 finger edges, all joints connected')
def _():
    from fusion.topology import (HAND_EDGES, JOINT_NAMES, NUM_HAND_JOINTS, PALM_EDGES,
                                 skeleton_edges)
    assert len(JOINT_NAMES) == NUM_HAND_JOINTS == 21, JOINT_NAMES
    assert len(HAND_EDGES) == 20, HAND_EDGES
    assert len(PALM_EDGES) == 3, PALM_EDGES
    covered = {j for e in HAND_EDGES for j in e}
    assert covered == set(range(21)), sorted(set(range(21)) - covered)
    assert skeleton_edges().shape == (23, 2)
    assert skeleton_edges(include_palm=False).shape == (20, 2)


@check('topology: edge list matches the mono-pipeline hand_topology.HAND_EDGES exactly')
def _():
    from fusion.topology import HAND_EDGES
    # Copied verbatim from hand_labelling_21kp/hand_topology.py. If this ever fails, the two repos
    # have diverged on keypoint order and every fused row is silently mis-wired.
    expected = [(0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    assert HAND_EDGES == expected, HAND_EDGES


@check('topology: joint order agrees with mano_joint_mapping in models/mano_layer.py')
def _():
    from fusion.topology import JOINT_NAMES
    # MANO's own order: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb,
    # then the five fingertips appended thumb, index, middle, ring, pinky (16..20).
    mano_names = (['wrist']
                  + [f'index_{s}' for s in ('mcp', 'pip', 'dip')]
                  + [f'middle_{s}' for s in ('mcp', 'pip', 'dip')]
                  + [f'pinky_{s}' for s in ('mcp', 'pip', 'dip')]
                  + [f'ring_{s}' for s in ('mcp', 'pip', 'dip')]
                  + [f'thumb_{s}' for s in ('mcp', 'pip', 'dip')]
                  + ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'])
    mapping = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    derived = [mano_names[i] for i in mapping]
    assert derived == JOINT_NAMES, list(zip(derived, JOINT_NAMES))


@check('topology: project_pinhole matches the closed form')
def _():
    from fusion.topology import pinhole_K, project_pinhole
    K = pinhole_K(800.0, 700.0, 640.0, 360.0)
    pts = np.array([[[0.0, 0.0, 1.0], [0.1, -0.2, 2.0]]])
    uv = project_pinhole(pts, K)
    assert approx(uv[0, 0, 0], 640.0) and approx(uv[0, 0, 1], 360.0), uv
    assert approx(uv[0, 1, 0], 800.0 * 0.1 / 2.0 + 640.0), uv
    assert approx(uv[0, 1, 1], 700.0 * -0.2 / 2.0 + 360.0), uv


@check('topology: bone_lengths on a unit-spaced synthetic hand')
def _():
    from fusion.topology import BONES, bone_lengths
    kp = np.zeros((1, 21, 3))
    kp[0, :, 0] = np.arange(21)                       # every joint 1 apart along x
    lens = bone_lengths(kp)
    assert lens.shape == (1, 20)
    for k, (a, b) in enumerate(BONES):
        assert approx(lens[0, k], abs(b - a)), (k, a, b, lens[0, k])


# ------------------------------------------------------------------ metrics


@check('metrics: coverage_metrics finds a planted gap with exact duration')
def _():
    from fusion.metrics import coverage_metrics
    fps, step = 30.0, 1
    processed = np.arange(0, 100)
    # Right hand seen on 0..9 and 40..49; frames 10..39 are a 30-frame = 1.0 s gap.
    frames = np.concatenate([np.arange(0, 10), np.arange(40, 50)])
    labels = np.ones(frames.size, dtype=int)
    out = coverage_metrics(frames, labels, processed, fps, step)
    right = out['right']
    assert right['frames_detected'] == 20, right
    assert right['n_gaps'] == 1, right
    assert approx(right['longest_gap_s'], 1.0, 1e-3), right
    assert approx(right['median_recovery_s'], 1.0, 1e-3), right
    assert out['left']['detections'] == 0 and out['left']['coverage'] == 0.0, out['left']


@check('metrics: coverage gap arithmetic is stride-aware')
def _():
    from fusion.metrics import coverage_metrics
    fps, step = 30.0, 3                                # sampled at 10 fps
    processed = np.arange(0, 90, step)
    frames = np.array([0, 3, 6, 30, 33])               # missing processed slots 9..27 = 7 slots
    labels = np.zeros(frames.size, dtype=int)
    out = coverage_metrics(frames, labels, processed, fps, step)['left']
    assert out['n_gaps'] == 1, out
    # 7 missing slots x 3 frames / 30 fps = 0.7 s. A stride-unaware version would say 0.233 s.
    assert approx(out['longest_gap_s'], 0.7, 1e-3), out


@check('metrics: track_metrics counts a planted label flip and drops short tracks')
def _():
    from fusion.metrics import track_metrics
    frames = np.arange(0, 20)
    kp2d = np.zeros((20, 21, 2))
    kp2d[:, :, 0] = 100.0                              # a stationary hand -> one track
    kp2d[:, :, 1] = 100.0
    labels = np.ones(20, dtype=int)
    labels[10] = 0                                     # one flip out and one back = 2
    out = track_metrics(frames, kp2d, labels, width=1920, fps=30.0, min_len=1)
    assert out['n_tracks_kept'] == 1, out
    assert out['label_flips'] == 2, out

    # A 3-detection track is discarded by a min_len of 10, and the detections are accounted for.
    out2 = track_metrics(frames[:3], kp2d[:3], labels[:3], width=1920, fps=30.0, min_len=10)
    assert out2['n_tracks_kept'] == 0, out2
    assert out2['n_tracks_dropped_short'] == 1, out2
    assert out2['detections_lost_to_short_tracks'] == 3, out2


@check('metrics: track_metrics separates two hands that never overlap')
def _():
    from fusion.metrics import track_metrics
    frames = np.repeat(np.arange(10), 2)
    kp2d = np.zeros((20, 21, 2))
    kp2d[0::2, :, 0] = 200.0                           # left hand, far from
    kp2d[1::2, :, 0] = 1600.0                          # right hand
    kp2d[:, :, 1] = 500.0
    labels = np.tile([0, 1], 10)
    out = track_metrics(frames, kp2d, labels, width=1920, fps=30.0, min_len=1)
    assert out['n_tracks_kept'] == 2, out
    assert out['label_flips'] == 0, out
    assert out['fragments_left'] == 1 and out['fragments_right'] == 1, out


@check('metrics: depth_qc flags a shallow fingertip inside a plausible wrist')
def _():
    from fusion.metrics import depth_qc
    kp = np.zeros((2, 21, 3))
    kp[:, :, 2] = 0.5
    kp[1, 8, 2] = 0.0002                               # index tip at 0.2 mm
    ok, stats = depth_qc(kp)
    assert ok.tolist() == [True, False], ok
    assert stats['implausible_rows'] == 1, stats
    assert stats['checked'] == 'all 21 joints'


@check('metrics: reprojection_qc is ~0 for consistent 3D/2D and large for a wrong K')
def _():
    from fusion.metrics import reprojection_qc
    from fusion.topology import pinhole_K, project_pinhole
    K = pinhole_K(800.0, 800.0, 640.0, 360.0)
    kp3 = np.random.default_rng(0).normal(size=(5, 21, 3)) * 0.05 + np.array([0, 0, 0.6])
    kp2 = project_pinhole(kp3, K)
    good = reprojection_qc(kp3, kp2, K)
    assert good['ALL']['median_px'] < 1e-6, good

    K_wrong = pinhole_K(800.0, 800.0, 600.0, 360.0)    # cx off by 40 px
    bad = reprojection_qc(kp3, kp2, K_wrong)
    assert bad['ALL']['median_px'] > 30.0, bad


@check('metrics: reprojection_qc splits by source label')
def _():
    from fusion.metrics import reprojection_qc
    from fusion.topology import pinhole_K, project_pinhole
    K = pinhole_K(800.0, 800.0, 640.0, 360.0)
    kp3 = np.zeros((4, 21, 3))
    kp3[:, :, 2] = 0.5
    kp2 = project_pinhole(kp3, K)
    src = np.array(['fused', 'fused', 'wilor', 'lifted_2d'])
    out = reprojection_qc(kp3, kp2, K, src)
    assert set(out) == {'fused', 'wilor', 'lifted_2d', 'ALL'}, out
    assert out['fused']['n'] == 2 and out['ALL']['n'] == 4, out


@check('metrics: fingertip_error reports a known offset in millimetres')
def _():
    from fusion.metrics import fingertip_error
    gt = np.zeros((3, 21, 3))
    gt[:, :, 2] = 0.5
    pred = gt.copy()
    pred[:, :, 0] += 0.01                              # a uniform 10 mm shift
    out = fingertip_error(pred, gt)
    assert out['is_proxy'] is False
    assert approx(out['camera_space']['all_joints']['mean_mm'], 10.0, 1e-3), out
    # A pure translation is invisible root-relative, which is the whole point of reporting both.
    assert approx(out['root_relative']['all_joints']['mean_mm'], 0.0, 1e-6), out


@check('metrics: foot_candidates counts a degenerate span and a low-in-frame wrist')
def _():
    from fusion.metrics import foot_candidates
    from fusion.topology import pinhole_K
    K = pinhole_K(800.0, 800.0, 640.0, 360.0)
    kp2 = np.zeros((3, 21, 2))
    kp3 = np.zeros((3, 21, 3))
    kp3[:, :, 2] = 1.0
    kp2[0, :, 1] = 100.0                               # high in frame, tight
    kp2[1, :, 1] = 1000.0                              # low in frame (H=1080), tight
    kp2[2, :, 0] = np.linspace(0, 1900, 21)            # 1900 px wide at Z=1 m -> impossible
    kp2[2, :, 1] = 100.0
    out = foot_candidates(kp2, kp3, K, height=1080)
    assert out['is_proxy'] is True
    assert out['low_in_frame'] == 1, out
    assert out['degenerate_span'] == 1, out


@check('metrics: bone_consistency reports zero spread for a rigid hand')
def _():
    from fusion.metrics import bone_consistency
    kp = np.zeros((5, 21, 3))
    kp[:, :, 0] = np.arange(21) * 0.02
    kp[:, :, 2] = np.linspace(0.4, 0.8, 5)[:, None]    # moves in depth, same shape
    out = bone_consistency(kp)
    assert out['n'] == 5, out
    assert out['bone_cv_max'] < 1e-9, out


# ------------------------------------------------------------------ fusion plumbing


@check('fuse: match_detections pairs by exact bbox')
def _():
    from fusion.fuse_egoforce_rtmpose import match_detections
    ego = dict(frame_idx=np.array([0, 0, 1]),
               bbox=np.array([[10, 10, 50, 50], [100, 10, 140, 50], [12, 12, 52, 52]], np.float32),
               kp2d=np.zeros((3, 21, 2), np.float32))
    ego['kp2d'][:, 0] = [[30, 30], [120, 30], [32, 32]]
    rtm = dict(frame_idx=np.array([0, 0, 1]),
               bbox=np.array([[100, 10, 140, 50], [10, 10, 50, 50], [12, 12, 52, 52]], np.float32),
               kp2d=np.zeros((3, 21, 2), np.float32))
    pairs, m_ego, m_rtm, stats = match_detections(ego, rtm, 120.0, True)
    assert stats['by_bbox'] == 3 and stats['by_wrist'] == 0, stats
    assert m_ego == {0, 1, 2} and m_rtm == {0, 1, 2}
    # Boxes were shuffled: row 0 of ego must pair with row 1 of rtm, not row 0.
    assert (0, 0, 1) in pairs and (0, 1, 0) in pairs, pairs


@check('fuse: match_detections falls back to wrist proximity and respects the gate')
def _():
    from fusion.fuse_egoforce_rtmpose import match_detections
    ego = dict(frame_idx=np.array([0]), kp2d=np.zeros((1, 21, 2), np.float32))
    ego['kp2d'][0, 0] = [100, 100]
    rtm = dict(frame_idx=np.array([0]), kp2d=np.zeros((1, 21, 2), np.float32))
    rtm['kp2d'][0, 0] = [140, 100]                     # 40 px away

    pairs, _, _, stats = match_detections(ego, rtm, 120.0, False)
    assert stats['by_wrist'] == 1 and len(pairs) == 1, (pairs, stats)

    pairs, _, _, stats = match_detections(ego, rtm, 20.0, False)   # gate below the distance
    assert pairs == [] and stats['by_wrist'] == 0, (pairs, stats)


@check('fuse: match_detections never pairs across frames')
def _():
    from fusion.fuse_egoforce_rtmpose import match_detections
    ego = dict(frame_idx=np.array([0]), kp2d=np.zeros((1, 21, 2), np.float32))
    rtm = dict(frame_idx=np.array([5]), kp2d=np.zeros((1, 21, 2), np.float32))
    pairs, m_ego, m_rtm, _ = match_detections(ego, rtm, 500.0, False)
    assert pairs == [] and not m_ego and not m_rtm, pairs


@check('fuse: dedup suppresses a duplicate but keeps two real hands that overlap')
def _():
    from fusion.fuse_egoforce_rtmpose import _bbox_of, _iomin, dedup
    width = 1920

    def blob(x0, x1, y0, y1):
        """A synthetic 21-joint hand filling the given box. Both axes must vary: a constant y
        makes a zero-height bbox, and then IoU/IoMin are 0 and nothing can ever be suppressed."""
        kp = np.zeros((21, 2))
        kp[:, 0] = np.linspace(x0, x1, 21)
        kp[:, 1] = np.linspace(y0, y1, 21)
        return kp

    # Two detections of ONE hand: a tight fit nested inside a sprawling one, wrists 5 px apart.
    kp_tight = blob(100, 200, 100, 190)
    kp_wide = blob(50, 400, 60, 400)
    kp_wide[0] = kp_tight[0] + 5
    keep = dedup(np.array([0, 0]), [kp_tight, kp_wide], ['fused', 'wilor'], width)
    assert keep.tolist() == [True, False], keep

    # Two REAL hands: heavily overlapping boxes but wrists far apart. Both must survive - this is
    # the two-handed-manipulation case the wrist gate exists for.
    kp_a = blob(100, 900, 400, 600)
    kp_b = blob(150, 950, 420, 620)
    kp_a[0] = [100, 400]
    kp_b[0] = [900, 620]                               # 800 px apart, way over 0.05*1920 = 96
    assert _iomin(_bbox_of(kp_a), _bbox_of(kp_b)) > 0.55, 'fixture must actually overlap'
    keep = dedup(np.array([0, 0]), [kp_a, kp_b], ['fused', 'wilor'], width)
    assert keep.tolist() == [True, True], keep


@check('fuse: two fused rows never suppress each other')
def _():
    from fusion.fuse_egoforce_rtmpose import dedup
    kp = np.zeros((21, 2)); kp[:, 0] = np.linspace(100, 200, 21); kp[:, 1] = 100
    keep = dedup(np.array([0, 0]), [kp, kp.copy()], ['fused', 'fused'], 1920)
    assert keep.tolist() == [True, True], keep


@check('fuse: borrow_depth inverts the pinhole projection')
def _():
    from fusion.fuse_egoforce_rtmpose import borrow_depth
    from fusion.topology import pinhole_K, project_pinhole
    K = pinhole_K(800.0, 750.0, 640.0, 360.0)
    kp3 = np.zeros((21, 3)); kp3[:, 0] = 0.05; kp3[:, 1] = -0.03; kp3[:, 2] = 0.6
    uv = project_pinhole(kp3, K)
    out = borrow_depth(uv, 0, {}, kp3[:, 2], 1920, K)
    assert np.allclose(out, kp3, atol=1e-5), np.abs(out - kp3).max()


@check('fuse: borrow_depth prefers a nearby measured wrist over the clip median')
def _():
    from fusion.fuse_egoforce_rtmpose import borrow_depth
    from fusion.topology import pinhole_K
    K = pinhole_K(800.0, 800.0, 640.0, 360.0)
    uv = np.zeros((21, 2)); uv[:, 0] = 700; uv[:, 1] = 400
    nearby_profile = np.full(21, 0.30)
    median_profile = np.full(21, 0.90)
    measured = {0: [(np.array([705.0, 402.0]), nearby_profile)]}
    out = borrow_depth(uv, 0, measured, median_profile, 1920, K)
    assert approx(out[0, 2], 0.30), out[0]

    far = {0: [(np.array([1900.0, 400.0]), nearby_profile)]}   # > 0.15 * 1920 = 288 px away
    out = borrow_depth(uv, 0, far, median_profile, 1920, K)
    assert approx(out[0, 2], 0.90), out[0]


# ------------------------------------------------------------------ refit maths


@check('refit: torch projection agrees with the numpy one')
def _():
    import torch

    from fusion.mano_refit import project_pinhole_torch
    from fusion.topology import pinhole_K, project_pinhole
    K = pinhole_K(812.5, 790.25, 641.5, 359.5)
    pts = np.random.default_rng(1).normal(size=(4, 21, 3)) * 0.05 + np.array([0, 0, 0.55])
    ref = project_pinhole(pts, K)
    got = project_pinhole_torch(torch.tensor(pts, dtype=torch.float64),
                                torch.tensor(K, dtype=torch.float64)).numpy()
    assert np.allclose(ref, got, atol=1e-9), np.abs(ref - got).max()


@check('refit: projection is differentiable w.r.t. the 3D points')
def _():
    import torch

    from fusion.mano_refit import project_pinhole_torch
    from fusion.topology import pinhole_K
    K = torch.tensor(pinhole_K(800.0, 800.0, 640.0, 360.0))
    pts = torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float64, requires_grad=True)
    project_pinhole_torch(pts, K).sum().backward()
    assert pts.grad is not None and torch.isfinite(pts.grad).all(), pts.grad
    # du/dx = fx/z = 1600 at z = 0.5
    assert approx(pts.grad[0, 0, 0].item(), 1600.0, 1e-6), pts.grad


@check('refit: huber is quadratic inside delta and linear outside')
def _():
    import torch

    from fusion.mano_refit import _huber
    d = torch.tensor([0.0, 1.0, 6.0, 100.0], dtype=torch.float64)
    out = _huber(d, 6.0)
    assert approx(out[0], 0.0) and approx(out[1], 0.5), out
    assert approx(out[2], 0.5 * 36.0), out                      # continuous at delta
    assert approx(out[3], 6.0 * (100.0 - 3.0)), out


@check('refit: weighted median ignores zero-weight joints')
def _():
    from fusion.mano_refit import _weighted_median
    values = np.array([[1.0, 2.0, 100.0]])
    weights = np.array([[1.0, 1.0, 0.0]])              # the outlier carries no weight
    assert _weighted_median(values, weights)[0] in (1.0, 2.0)
    assert np.isnan(_weighted_median(values, np.zeros_like(weights))[0])


@check('refit: 6D -> axis-angle round-trips through the repo helpers')
def _():
    import torch

    from utils.rotations import axis_angle_to_rotation_6d, rotation_6d_to_axis_angle_direct
    rng = np.random.default_rng(2)
    aa = rng.normal(size=(32, 3)) * 0.4                # modest angles, away from the pi wrap
    aa_t = torch.tensor(aa, dtype=torch.float32)
    back = rotation_6d_to_axis_angle_direct(axis_angle_to_rotation_6d(aa_t)).numpy()
    assert np.allclose(aa, back, atol=1e-4), np.abs(aa - back).max()


@check('refit: accept_refit accounts for every rejection reason exactly once')
def _():
    from fusion.mano_refit import RefitResult, accept_refit

    n = 6
    joints = np.zeros((n, 21, 3), np.float32)
    joints[:, :, 2] = 0.5
    joints[1] = np.nan                                 # non-finite
    joints[2, :, 2] = -0.5                             # behind the camera
    res = RefitResult(
        joints3d=joints,
        uv=np.zeros((n, 21, 2), np.float32),
        betas=np.zeros((n, 10), np.float32),
        global_orient6=np.zeros((n, 6), np.float32),
        hand_pose6=np.zeros((n, 90), np.float32),
        transl=np.zeros((n, 3), np.float32),
        residual_px=np.array([5.0, 5.0, 5.0, 999.0, 5.0, 5.0]),
        residual_px_init=np.array([9.0, 9.0, 9.0, 9.0, 9.0, 1.0]),   # row 5 got worse
        bone_change=np.array([0.01, 0.0, 0.0, 0.0, 0.99, 0.0]),      # row 4 stretched
        depth_change_m=np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    ok, reasons = accept_refit(res, max_reproj_px=20.0, max_bone_change=0.15,
                               max_depth_change_m=0.05)
    assert ok.tolist() == [True, False, False, False, False, False], ok
    assert reasons['accepted'] == 1, reasons
    assert reasons['rejected_nonfinite'] == 1, reasons
    assert reasons['rejected_behind_camera'] == 1, reasons
    assert reasons['rejected_reproj'] == 1, reasons
    assert reasons['rejected_bone_change'] == 1, reasons
    assert reasons['rejected_no_improvement'] == 1, reasons
    counted = sum(v for k, v in reasons.items() if k.startswith('rejected_'))
    assert counted + reasons['accepted'] == reasons['total'] == n, reasons


@check('refit: an empty input produces empty, correctly-shaped output')
def _():
    from fusion.mano_refit import ManoRefiner, RefitWeights
    # No MANO files needed: refit() short-circuits before touching the model. __init__ is bypassed
    # for the same reason, so the one attribute the empty path reads has to be supplied by hand.
    refiner = object.__new__(ManoRefiner)
    refiner.weights = RefitWeights()
    empty = dict(betas=np.zeros((0, 10), np.float32), global_orient6=np.zeros((0, 6), np.float32),
                 hand_pose6=np.zeros((0, 90), np.float32), transl=np.zeros((0, 3), np.float32))
    out = refiner.refit(empty, np.zeros((0, 21, 2)), np.zeros((0, 21)), np.zeros(0), np.eye(3))
    assert out.joints3d.shape == (0, 21, 3), out.joints3d.shape
    assert out.residual_px.shape == (0,), out.residual_px.shape


# ------------------------------------------------------------------ calibration / video


@check('calibration: read_K handles the flat, camera and per-eye schemas')
def _():
    from fusion.calibration import read_K
    with tempfile.TemporaryDirectory() as tmp:
        flat = os.path.join(tmp, 'flat.json')
        with open(flat, 'w') as fh:
            json.dump(dict(fx=1.0, fy=2.0, cx=3.0, cy=4.0), fh)
        assert read_K(flat) == ([1.0, 2.0, 3.0, 4.0], 'flat')

        cam = os.path.join(tmp, 'cam.json')
        with open(cam, 'w') as fh:
            json.dump({'rectified': {'camera': dict(fx=5.0, fy=6.0, cx=7.0, cy=8.0)}}, fh)
        assert read_K(cam) == ([5.0, 6.0, 7.0, 8.0], 'rectified')

        eyes = os.path.join(tmp, 'eyes.json')
        with open(eyes, 'w') as fh:
            json.dump({'rectified': {'left': dict(fx=9.0, fy=10.0, cx=11.0, cy=12.0),
                                     'right': dict(fx=1.0, fy=1.0, cx=1.0, cy=1.0)}}, fh)
        assert read_K(eyes, eye='left')[0] == [9.0, 10.0, 11.0, 12.0]
        assert read_K(eyes, eye='right')[0] == [1.0, 1.0, 1.0, 1.0]

        # No 'rectified' block -> falls back to raw, loudly.
        raw = os.path.join(tmp, 'raw.json')
        with open(raw, 'w') as fh:
            json.dump({'raw': {'camera': dict(fx=1.0, fy=1.0, cx=1.0, cy=1.0)}}, fh)
        assert read_K(raw)[1] == 'raw'


@check('video_io: frame-selection arithmetic matches the mono-pipeline convention')
def _():
    from fusion.video_io import VideoInfo
    fps, total = 30.0, 900

    # sample_fps 10 on a 30 fps source -> step 3
    step = max(1, int(round(fps / 10.0)))
    assert step == 3

    # 2 s in, 4 s long -> frames [60, 180)
    start = int(round(2.0 * fps))
    end = min(total, start + int(round(4.0 * fps)))
    info = VideoInfo('x', 1920, 1080, fps, total, step, start, end, 10.0)
    assert (start, end) == (60, 180)
    assert info.n_selected == 40, info.n_selected        # 120 frames / stride 3
    assert approx(info.output_fps, 10.0)

    # duration_sec >= 1e8 means "to the end"
    info2 = VideoInfo('x', 1920, 1080, fps, total, 1, 0, total, 0.0)
    assert info2.n_selected == total and approx(info2.output_fps, fps)


@check('video_io + topology: drawing does not crash and marks the pixels it should')
def _():
    from fusion.topology import draw_forearm, draw_hand, draw_points
    frame = np.zeros((360, 640, 3), np.uint8)
    kp = np.stack([np.linspace(100, 300, 21), np.linspace(100, 250, 21)], axis=1)
    draw_hand(frame, kp, 'left')
    draw_hand(frame, kp + 20, 'right')
    draw_points(frame, kp, (255, 255, 255))
    draw_forearm(frame, kp[:3])
    assert frame.any(), 'nothing was drawn'

    # Non-finite and wildly off-frame joints must be skipped, not crash or wrap around.
    frame2 = np.zeros((360, 640, 3), np.uint8)
    bad = kp.copy()
    bad[5] = np.nan
    bad[6] = [1e9, -1e9]
    draw_hand(frame2, bad, 'left')
    assert frame2.any()


@check('testcases.yaml parses and every case names known stages')
def _():
    import yaml
    known = {'egoforce_3d', 'rtmpose_2d', 'fuse', 'fuse_rigid'}
    path = os.path.join(ROOT_DIR, 'fusion', 'testcases.yaml')
    with open(path) as fh:
        config = yaml.safe_load(fh)
    assert config.get('clips'), 'no clips block'
    ids = set()
    for case in config['cases']:
        assert 'id' in case and 'stages' in case, case
        assert case['id'] not in ids, f"duplicate case id {case['id']}"
        ids.add(case['id'])
        unknown = set(case['stages']) - known
        assert not unknown, f"{case['id']} names unknown stage(s) {unknown}"
        for stage in case['stages']:
            assert isinstance(case.get(stage, {}) or {}, dict), case
    assert {'T1_egoforce_only', 'T2b_articulated'} <= ids, ids


@check('run_testcases: flag rendering handles switches, values and lists')
def _():
    from fusion.run_testcases import as_flags, flag
    assert flag('hand_conf') == '--hand-conf'
    assert as_flags(dict(no_kalman=True)) == ['--no-kalman']
    assert as_flags(dict(no_kalman=False, mode=None)) == []
    assert as_flags(dict(hand_conf=0.15)) == ['--hand-conf', '0.15']
    assert as_flags(dict(K=[1, 2, 3, 4])) == ['--K', '1', '2', '3', '4']


def main():
    failures = []
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            failures.append((name, f'{type(exc).__name__}: {exc}'))
            print(f'FAIL  {name}')
            print(f'      {type(exc).__name__}: {exc}')
        else:
            print(f'ok    {name}')

    print()
    print(f'{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed')
    if failures:
        print('\nFAILURES:')
        for name, why in failures:
            print(f'  {name}: {why}')
        return 1
    print('\nGreen means the plumbing and arithmetic are right. It does NOT mean the pipeline '
          'produces good hands - EgoForce, RTMPose and the MANO forward pass are all untested here. '
          'See plan_of_action.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
