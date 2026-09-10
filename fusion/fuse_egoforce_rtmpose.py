#!/usr/bin/env python3
"""Stage 2 (replacement) - fuse EgoForce 3D with RTMPose-m Hand5 2D by refitting articulation.

Writes ``<stem>_hand21_keypoints.npz`` in the **exact schema the mono-pipeline's ``fuse_2d_3d.py``
produces**, so ``run_clip.py --from-npz <that file>`` re-runs tracking, filtering, handedness and the
review overlay with no GPU and no changes to that repo.

Two modes:

``--mode articulated`` (default, Experiment 2)
    Refit EgoForce's MANO parameters against RTMPose's confidence-weighted landmarks. See
    ``fusion/mano_refit.py`` for the objective. This is the mode that can fix a wrongly bent finger,
    which the rigid PnP fusion structurally cannot.

``--mode egoforce-only`` (Experiment 1)
    Pass EgoForce's 3D straight through into the fused schema. No 2D model, nothing to match. This
    is the baseline the model swap should be judged against.

The **rigid** fusion is deliberately not reimplemented here. The mono-pipeline's existing
``fuse_2d_3d.py --fusion pnp`` already does it and accepts our two producer npz files unchanged, so
the rigid baseline comes for free and stays bit-identical to the historical one. ``run_testcases.py``
wires that up as test case ``T2a``.

Source labels use the mono-pipeline's **existing vocabulary**, not new names, so ``postprocess.py``
works untouched. The mapping is:

    'fused'          EgoForce + RTMPose, refit accepted
    'wilor_pnpfail'  matched, but the refit was rejected -> raw EgoForce 3D  (measured depth)
    'wilor'          EgoForce only, RTMPose had no detection                  (measured depth)
    'lifted_2d'      RTMPose only -> depth BORROWED, depth_measured=False

An additive ``producer`` array records the true model names, since 'wilor' rows contain no WiLoR.

Known interaction with that repo, worth being explicit about: ``postprocess.track_features`` computes
``mp_support`` as ``mean(source in ('fused', 'lifted_2d'))``, which **excludes 'wilor_pnpfail'** even
though those rows were 2D-matched. A track that is mostly 'wilor_pnpfail' therefore reads as
uncorroborated and can be dropped as 'wilor_only'. Nothing here can fix that; it is listed in
plan_of_action.md.

Example
-------
    python fusion/fuse_egoforce_rtmpose.py \
        --egoforce work/clip_3d_keypoints.npz \
        --rtmpose  work/clip_2d_keypoints.npz \
        --out work --stem clip --video /path/to/clip.mp4 --overlay
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import collections
import json
import time

import numpy as np

from fusion.mano_refit import ManoRefiner, RefitWeights, accept_refit
from fusion.metrics import (bone_consistency, coverage_metrics, depth_qc, fingertip_agreement,
                            foot_candidates, reprojection_qc, track_metrics)
from fusion.topology import NUM_HAND_JOINTS, WRIST

# NEVER 0.0: a consumer filtering `fuse_residual_px < 5` would otherwise KEEP every uncorroborated
# single-model row and DROP the cross-model-corroborated ones - exactly inverted. Same reasoning as
# the mono-pipeline's NOT_COMPUTED.
NOT_COMPUTED = float('nan')

MANO_KEYS = ('mano_betas', 'mano_global_orient', 'mano_hand_pose', 'mano_transl')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse EgoForce 3D with RTMPose 2D by refitting MANO articulation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--egoforce', required=True, help='EgoForce _3d_keypoints.npz')
    parser.add_argument('--rtmpose', default=None,
                        help='RTMPose _2d_keypoints.npz (omit for --mode egoforce-only)')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--stem', default=None, help='output basename')
    parser.add_argument('--mode', default='articulated', choices=['articulated', 'egoforce-only'])

    parser.add_argument('--mano-path', default=None,
                        help='MANO model dir (default: settings.config.MANO_PATH)')
    parser.add_argument('--device', default=None,
                        help='torch device for the refit (default: cuda if available, else cpu). '
                             'The refit needs no TensorRT, so CPU works.')

    match = parser.add_argument_group('matching')
    match.add_argument('--match-dist', type=float, default=120.0,
                       help='wrist-proximity fallback gate, px')
    match.add_argument('--no-bbox-match', action='store_true',
                       help='skip exact-bbox matching and always use wrist proximity')

    refit = parser.add_argument_group('refit')
    refit.add_argument('--iters', type=int, default=80)
    refit.add_argument('--lr', type=float, default=0.02)
    refit.add_argument('--chunk', type=int, default=256, help='detections optimised per batch')
    refit.add_argument('--w-reproj', type=float, default=RefitWeights.reproj)
    refit.add_argument('--w-depth', type=float, default=RefitWeights.depth)
    refit.add_argument('--w-pose', type=float, default=RefitWeights.pose)
    refit.add_argument('--w-orient', type=float, default=RefitWeights.orient)
    refit.add_argument('--w-beta', type=float, default=RefitWeights.beta)
    refit.add_argument('--w-limit', type=float, default=RefitWeights.limit)
    refit.add_argument('--limit-max-angle', type=float, default=RefitWeights.limit_max_angle)
    refit.add_argument('--huber-px', type=float, default=RefitWeights.huber_delta_px)
    refit.add_argument('--min-kpt-conf', type=float, default=RefitWeights.min_conf)

    gate = parser.add_argument_group('accept / reject')
    gate.add_argument('--max-reproj-px', type=float, default=20.0)
    gate.add_argument('--max-bone-change', type=float, default=0.15,
                      help='reject a refit that changes any bone length by more than this fraction')
    gate.add_argument('--max-depth-change-m', type=float, default=0.05)
    gate.add_argument('--allow-no-improvement', action='store_true',
                      help='keep a refit whose reprojection did not improve on EgoForce')

    dedup = parser.add_argument_group('duplicate suppression')
    dedup.add_argument('--nms-iou', type=float, default=0.4)
    dedup.add_argument('--nms-iomin', type=float, default=0.55)
    dedup.add_argument('--nms-wrist-frac', type=float, default=0.05)

    qc = parser.add_argument_group('qc')
    qc.add_argument('--depth-min', type=float, default=0.05)
    qc.add_argument('--depth-max', type=float, default=3.0)
    qc.add_argument('--track-min-len', type=int, default=10,
                    help="mirror the mono-pipeline's --min-len so the report shows what it will drop")

    parser.add_argument('--video', default=None, help='source video, only needed for --overlay')
    parser.add_argument('--overlay', action='store_true',
                        help='write a comparison mp4: EgoForce raw vs refit vs RTMPose observation')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


# ------------------------------------------------------------------ duplicate suppression


def _bbox_of(kp):
    return (float(kp[:, 0].min()), float(kp[:, 1].min()),
            float(kp[:, 0].max()), float(kp[:, 1].max()))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _iomin(a, b):
    """Intersection over the SMALLER box - catches a tight hand nested in a sprawling duplicate."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (min(aa, bb) + 1e-9)


def dedup(frames, kp2d, sources, width, iou_thr=0.4, iomin_thr=0.55, wrist_frac=0.05):
    """Suppress duplicate detections of ONE physical hand.

    Same algorithm and same rationale as ``fuse_2d_3d.dedup``: a candidate is suppressed only if it
    overlaps a kept detection AND their WRISTS are close. The wrist test is what stops the
    two-handed-manipulation failure, where one hand steadying a part and the other reaching across it
    give IoMin ~ 1.0 while being two real hands. Two 'fused' rows never suppress each other.
    """
    priority = {'fused': 2, 'wilor': 1, 'wilor_pnpfail': 1, 'lifted_2d': 0}
    boxes = [_bbox_of(k) for k in kp2d]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    wrists = [k[WRIST] for k in kp2d]
    wrist_gate = wrist_frac * float(width)

    keep = np.ones(len(frames), bool)
    by_frame = collections.defaultdict(list)
    for i, f in enumerate(frames):
        by_frame[int(f)].append(i)

    for idxs in by_frame.values():
        kept = []
        for i in sorted(idxs, key=lambda i: (priority.get(sources[i], 0), -areas[i]), reverse=True):
            duplicate = False
            for j in kept:
                if sources[i] == 'fused' and sources[j] == 'fused':
                    continue
                if float(np.hypot(*(wrists[i] - wrists[j]))) > wrist_gate:
                    continue
                if _iou(boxes[i], boxes[j]) > iou_thr or _iomin(boxes[i], boxes[j]) > iomin_thr:
                    duplicate = True
                    break
            if duplicate:
                keep[i] = False
            else:
                kept.append(i)
    return keep


# ------------------------------------------------------------------ matching


def match_detections(ego, rtm, match_dist, use_bbox):
    """Pair EgoForce rows with RTMPose rows. Returns ``(pairs, matched_ego, matched_rtm)``.

    Exact-bbox matching is tried first: when ``run_rtmpose_2d.py --boxes npz`` was used, every
    RTMPose row was produced FROM an EgoForce box, so the pairing is exact and cannot mis-associate.
    Wrist proximity is the fallback for independently-detected boxes, matching the mono-pipeline's
    behaviour.
    """
    ego_by_frame = collections.defaultdict(list)
    for i, f in enumerate(ego['frame_idx']):
        ego_by_frame[int(f)].append(i)
    rtm_by_frame = collections.defaultdict(list)
    for i, f in enumerate(rtm['frame_idx']):
        rtm_by_frame[int(f)].append(i)

    pairs = []
    n_bbox, n_wrist = 0, 0
    have_rtm_bbox = use_bbox and 'bbox' in rtm and len(rtm['bbox']) == len(rtm['frame_idx'])

    for frame in sorted(set(ego_by_frame) | set(rtm_by_frame)):
        ego_rows = list(ego_by_frame.get(frame, []))
        rtm_rows = list(rtm_by_frame.get(frame, []))
        used = set()

        if have_rtm_bbox:
            for ei in list(ego_rows):
                ebox = np.asarray(ego['bbox'][ei], dtype=np.float32)
                for ri in rtm_rows:
                    if ri in used:
                        continue
                    if np.allclose(ebox, np.asarray(rtm['bbox'][ri], dtype=np.float32), atol=0.51):
                        pairs.append((frame, ei, ri))
                        used.add(ri)
                        ego_rows.remove(ei)
                        n_bbox += 1
                        break

        for ei in ego_rows:
            wrist = ego['kp2d'][ei, WRIST]
            best, best_d = -1, match_dist
            for ri in rtm_rows:
                if ri in used:
                    continue
                d = float(np.hypot(*(rtm['kp2d'][ri, WRIST] - wrist)))
                if d < best_d:
                    best, best_d = ri, d
            if best >= 0:
                used.add(best)
                pairs.append((frame, ei, best))
                n_wrist += 1

    matched_ego = {ei for _, ei, _ in pairs}
    matched_rtm = {ri for _, _, ri in pairs}
    return pairs, matched_ego, matched_rtm, dict(by_bbox=n_bbox, by_wrist=n_wrist)


# ------------------------------------------------------------------ depth borrowing


def borrow_depth(uv, frame, measured_by_frame, median_profile, width, K):
    """Give a 2D-only detection a depth profile borrowed from the nearest measured detection.

    Mirrors ``fuse_2d_3d``'s approach, and like it the result is marked ``depth_measured=False``.
    Depth here is BORROWED, not measured; do not use these rows for anything metric.
    """
    Z = median_profile
    frames = np.array(sorted(measured_by_frame)) if measured_by_frame else np.zeros(0, int)
    if frames.size:
        near = frames[np.argsort(np.abs(frames - int(frame)))[:5]]
        candidates = [(float(np.hypot(*(wxy - uv[WRIST]))), z)
                      for nf in near for wxy, z in measured_by_frame[int(nf)]]
        if candidates:
            d, z = min(candidates, key=lambda c: c[0])
            if d < 0.15 * width:
                Z = z

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (uv[:, 0] - cx) * Z / fx
    Y = (uv[:, 1] - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


# ------------------------------------------------------------------ main


def load_egoforce(path):
    data = dict(np.load(path, allow_pickle=False))
    required = ('kp3d_cam', 'kp2d', 'frame_idx', 'is_right', 'K', 'width', 'height', 'fps', 'step')
    for key in required:
        if key not in data:
            raise SystemExit(f'{path} is missing "{key}" - is it an EgoForce _3d_keypoints.npz?')
    return data


def main():
    args = parse_args()
    t0 = time.time()

    ego = load_egoforce(args.egoforce)
    K = np.asarray(ego['K'], dtype=np.float64)
    width, height = int(ego['width']), int(ego['height'])
    step = int(ego['step'])
    fps = float(ego['fps'])
    stem = args.stem or os.path.basename(args.egoforce).replace('_3d_keypoints.npz', '')
    os.makedirs(args.out, exist_ok=True)

    rtm = None
    if args.mode == 'articulated':
        if not args.rtmpose:
            raise SystemExit('--mode articulated requires --rtmpose (or use --mode egoforce-only)')
        rtm = dict(np.load(args.rtmpose, allow_pickle=False))
        # The two stages must have run on the SAME pixel grid and the SAME stride, else the matcher
        # quietly matches nothing and the output degenerates with no error. Same guard as fuse_2d_3d.
        for key in ('width', 'height'):
            if int(rtm[key]) != int(ego[key]):
                raise SystemExit(f'2D/3D {key} mismatch: {int(rtm[key])} vs {int(ego[key])} - the '
                                 f'two stages ran on different frames/crops; re-run both')
        if int(rtm['step']) != step:
            raise SystemExit(f"2D/3D stride mismatch: {int(rtm['step'])} vs {step}")
        for key in MANO_KEYS:
            if key not in ego:
                raise SystemExit(
                    f'{args.egoforce} has no "{key}". The articulated refit needs EgoForce\'s MANO '
                    f'parameters - regenerate it with fusion/run_egoforce_3d.py.')

    n_ego = len(ego['frame_idx'])
    print(f'[fuse {stem}] mode={args.mode} egoforce_rows={n_ego} '
          f"rtmpose_rows={0 if rtm is None else len(rtm['frame_idx'])} "
          f'{width}x{height}@{fps:.2f} step={step}')

    rows = collections.defaultdict(list)

    def add(kp3d, kp2d, frame, is_right_ego, is_right_2d, score, source, producer, raw3d,
            residual, measured, bone_change=np.nan, depth_change=np.nan, kp2d_obs=None,
            kp2d_conf=None):
        rows['kp3d'].append(np.asarray(kp3d, dtype=np.float32))
        rows['kp2d'].append(np.asarray(kp2d, dtype=np.float32))
        rows['raw3d'].append(np.asarray(raw3d, dtype=np.float32))
        rows['frame'].append(int(frame))
        rows['is_right_ego'].append(int(is_right_ego))
        rows['is_right_2d'].append(int(is_right_2d))
        rows['score'].append(float(score))
        rows['source'].append(source)
        rows['producer'].append(producer)
        rows['residual'].append(float(residual))
        rows['measured'].append(bool(measured))
        rows['bone_change'].append(float(bone_change))
        rows['depth_change'].append(float(depth_change))
        nan2 = np.full((NUM_HAND_JOINTS, 2), np.nan, dtype=np.float32)
        nan1 = np.full((NUM_HAND_JOINTS,), np.nan, dtype=np.float32)
        rows['kp2d_obs'].append(nan2 if kp2d_obs is None else np.asarray(kp2d_obs, np.float32))
        rows['kp2d_conf'].append(nan1 if kp2d_conf is None else np.asarray(kp2d_conf, np.float32))

    refit_reasons = {}
    match_stats = {}

    if args.mode == 'egoforce-only':
        for i in range(n_ego):
            add(ego['kp3d_cam'][i], ego['kp2d'][i], ego['frame_idx'][i],
                ego['is_right'][i], -1, 0.0, 'wilor', 'egoforce',
                ego['kp3d_cam'][i], NOT_COMPUTED, True)
    else:
        pairs, matched_ego, matched_rtm, match_stats = match_detections(
            ego, rtm, args.match_dist, not args.no_bbox_match)
        print(f'[fuse {stem}] matched {len(pairs)} pairs '
              f"(bbox-exact {match_stats['by_bbox']}, wrist {match_stats['by_wrist']})")

        if pairs:
            ego_idx = np.array([ei for _, ei, _ in pairs], dtype=np.int64)
            rtm_idx = np.array([ri for _, _, ri in pairs], dtype=np.int64)

            conf = (rtm['kp2d_conf'][rtm_idx] if 'kp2d_conf' in rtm
                    else np.ones((len(rtm_idx), NUM_HAND_JOINTS), dtype=np.float32))

            weights = RefitWeights(
                reproj=args.w_reproj, depth=args.w_depth, pose=args.w_pose, orient=args.w_orient,
                beta=args.w_beta, limit=args.w_limit, limit_max_angle=args.limit_max_angle,
                huber_delta_px=args.huber_px, min_conf=args.min_kpt_conf)

            mano_path = args.mano_path
            if mano_path is None:
                from settings import config as cfg
                mano_path = cfg.MANO_PATH

            device = args.device
            if device is None:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'[fuse {stem}] refitting {len(pairs)} detections on {device} '
                  f'({args.iters} iters, chunk {args.chunk})')

            refiner = ManoRefiner(mano_path, device=device, weights=weights)
            init = dict(betas=ego['mano_betas'][ego_idx],
                        global_orient6=ego['mano_global_orient'][ego_idx],
                        hand_pose6=ego['mano_hand_pose'][ego_idx],
                        transl=ego['mano_transl'][ego_idx])
            result = refiner.refit(init, rtm['kp2d'][rtm_idx], conf, ego['is_right'][ego_idx], K,
                                   iters=args.iters, lr=args.lr, chunk=args.chunk,
                                   verbose=args.verbose)
            accepted, refit_reasons = accept_refit(
                result, max_reproj_px=args.max_reproj_px, max_bone_change=args.max_bone_change,
                max_depth_change_m=args.max_depth_change_m,
                require_improvement=not args.allow_no_improvement)
            print(f'[fuse {stem}] refit accepted {refit_reasons["accepted"]}/'
                  f'{refit_reasons["total"]} -> {json.dumps(refit_reasons)}')

            for k, (frame, ei, ri) in enumerate(pairs):
                raw3d = ego['kp3d_cam'][ei]
                obs = rtm['kp2d'][ri]
                if accepted[k]:
                    add(result.joints3d[k], result.uv[k], frame, ego['is_right'][ei],
                        rtm['is_right'][ri], rtm['score'][ri], 'fused', 'egoforce+rtmpose',
                        raw3d, result.residual_px[k], True,
                        result.bone_change[k], result.depth_change_m[k], obs, conf[k])
                else:
                    add(raw3d, ego['kp2d'][ei], frame, ego['is_right'][ei],
                        rtm['is_right'][ri], rtm['score'][ri], 'wilor_pnpfail', 'egoforce',
                        raw3d, result.residual_px[k], True,
                        result.bone_change[k], result.depth_change_m[k], obs, conf[k])
        else:
            matched_ego, matched_rtm = set(), set()

        for i in range(n_ego):                                  # EgoForce only
            if i in matched_ego:
                continue
            add(ego['kp3d_cam'][i], ego['kp2d'][i], ego['frame_idx'][i], ego['is_right'][i],
                -1, 0.0, 'wilor', 'egoforce', ego['kp3d_cam'][i], NOT_COMPUTED, True)

        # RTMPose only: 2D is real, depth must be BORROWED. Marked depth_measured=False so a metric
        # consumer can exclude fabricated depth.
        measured_by_frame = collections.defaultdict(list)
        for k in range(len(rows['kp3d'])):
            if rows['measured'][k]:
                measured_by_frame[rows['frame'][k]].append(
                    (rows['kp2d'][k][WRIST], rows['kp3d'][k][:, 2]))
        if measured_by_frame:
            profile = np.median(np.stack([z for lst in measured_by_frame.values() for _, z in lst]),
                                axis=0)
        else:
            profile = np.full(NUM_HAND_JOINTS, 0.45, dtype=np.float32)

        for i in range(len(rtm['frame_idx'])):
            if i in matched_rtm:
                continue
            uv = np.asarray(rtm['kp2d'][i], dtype=np.float64)
            kp3 = borrow_depth(uv, int(rtm['frame_idx'][i]), measured_by_frame, profile, width, K)
            add(kp3, uv, rtm['frame_idx'][i], -1, rtm['is_right'][i], rtm['score'][i],
                'lifted_2d', 'rtmpose', np.full((NUM_HAND_JOINTS, 3), np.nan, np.float32),
                NOT_COMPUTED, False,
                kp2d_obs=uv,
                kp2d_conf=(rtm['kp2d_conf'][i] if 'kp2d_conf' in rtm else None))

    if not rows['kp3d']:
        raise SystemExit(f'[fuse {stem}] no detections anywhere in this clip; nothing to write')

    frames = np.asarray(rows['frame'], dtype=np.int32)
    sources = list(rows['source'])
    keep = dedup(frames, rows['kp2d'], sources, width,
                 args.nms_iou, args.nms_iomin, args.nms_wrist_frac)
    order = [i for i in np.argsort(frames, kind='stable') if keep[i]]

    def take(key, dtype=np.float32):
        return np.asarray([rows[key][i] for i in order], dtype=dtype)

    out = dict(
        # --- the mono-pipeline fused schema, verbatim
        kp3d_cam=take('kp3d'),
        kp3d_cam_wilor_raw=take('raw3d'),
        kp2d=take('kp2d'),
        frame_idx=take('frame', np.int32),
        is_right_wilor=take('is_right_ego', np.int8),
        is_right_mp=take('is_right_2d', np.int8),
        mp_score=take('score'),
        source=np.asarray([sources[i] for i in order]),
        fuse_residual_px=take('residual'),
        depth_measured=take('measured', bool),
        K=K,
        n_dropped_dupes=np.int32(int((~keep).sum())),
        n_pnp_fallback=np.int32(int(sum(1 for i in order if sources[i] == 'wilor_pnpfail'))),
        width=np.int32(width), height=np.int32(height),
        fps=np.float32(fps), step=np.int32(step),
        fusion_mode=np.array(args.mode),
        pp_rebase=np.bool_(bool(ego['pp_rebase'])) if 'pp_rebase' in ego else np.bool_(True),
        # --- additive
        producer=np.asarray([rows['producer'][i] for i in order]),
        is_right_egoforce=take('is_right_ego', np.int8),
        is_right_rtmpose=take('is_right_2d', np.int8),
        rtmpose_score=take('score'),
        kp2d_rtmpose=take('kp2d_obs'),
        kp2d_rtmpose_conf=take('kp2d_conf'),
        refit_bone_change=take('bone_change'),
        refit_depth_change_m=take('depth_change'),
        keypoint_order=np.array('OpenPose-21'),
    )

    plausible, depth_stats = depth_qc(out['kp3d_cam'], args.depth_min, args.depth_max)
    out['depth_plausible'] = plausible

    npz_path = os.path.join(args.out, f'{stem}_hand21_keypoints.npz')
    np.savez_compressed(npz_path, **out)

    # ------------------------------------------------------------------ stats sidecar
    src_list = out['source'].tolist()
    fused_mask = out['source'] == 'fused'
    hand_label = np.where(out['is_right_wilor'] >= 0, out['is_right_wilor'], out['is_right_mp'])

    stats = dict(
        stem=stem, mode=args.mode,
        egoforce_npz=os.path.basename(args.egoforce),
        rtmpose_npz=None if args.rtmpose is None else os.path.basename(args.rtmpose),
        hands=len(order),
        source_mix={s: src_list.count(s) for s in sorted(set(src_list))},
        producer_mix={p: out['producer'].tolist().count(p)
                      for p in sorted(set(out['producer'].tolist()))},
        dropped_dupes=int((~keep).sum()),
        match=match_stats,
        refit=dict(reasons=refit_reasons,
                   residual_px_median=(round(float(np.nanmedian(out['fuse_residual_px'][fused_mask])), 3)
                                       if fused_mask.any() else None),
                   bone_change_median=(round(float(np.nanmedian(out['refit_bone_change'][fused_mask])), 5)
                                       if fused_mask.any() else None),
                   depth_change_m_median=(round(float(np.nanmedian(
                       out['refit_depth_change_m'][fused_mask])), 5) if fused_mask.any() else None)),
        reprojection_px=reprojection_qc(out['kp3d_cam'], out['kp2d'], K, out['source']),
        depth=depth_stats,
        bone_consistency=bone_consistency(out['kp3d_cam'][out['depth_measured']]),
        coverage=coverage_metrics(out['frame_idx'], hand_label,
                                  ego.get('processed_frames', out['frame_idx']), fps, step),
        tracks=track_metrics(out['frame_idx'], out['kp2d'], hand_label, width, fps, step,
                             min_len=args.track_min_len),
        foot_candidates_proxy=foot_candidates(out['kp2d'], out['kp3d_cam'], K, height),
        elapsed_sec=round(time.time() - t0, 1),
    )
    if fused_mask.any():
        stats['fingertip_agreement_proxy'] = fingertip_agreement(
            out['kp3d_cam'][fused_mask], K, out['kp2d_rtmpose'][fused_mask],
            out['kp2d_rtmpose_conf'][fused_mask], args.min_kpt_conf)

    stats_path = os.path.join(args.out, f'{stem}_fuse_stats.json')
    with open(stats_path, 'w') as handle:
        json.dump(stats, handle, indent=2, default=str)

    print(f"[fuse {stem}] {stats['hands']} hands {stats['source_mix']} "
          f"dupes={stats['dropped_dupes']} refit_fallback={int(out['n_pnp_fallback'])}")
    print(f"[fuse {stem}] reprojection: {json.dumps(stats['reprojection_px'])}")
    print(f"[fuse {stem}] depth: {json.dumps(stats['depth'])}")
    print(f'[fuse {stem}] wrote {npz_path}')
    print(f'[fuse {stem}] wrote {stats_path}')

    if args.overlay:
        if not args.video:
            print('[fuse] --overlay needs --video; skipping the comparison mp4')
        else:
            from fusion.render_comparison import render_comparison
            mp4 = os.path.join(args.out, f'{stem}_fusion_comparison.mp4')
            render_comparison(args.video, out, mp4)
            print(f'[fuse {stem}] wrote {mp4}')


if __name__ == '__main__':
    main()
