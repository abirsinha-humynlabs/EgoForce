#!/usr/bin/env python3
"""Stage 1b (replacement) - 21 metric 3D hand keypoints per hand via EgoForce.

This is a **drop-in replacement for the mono-pipeline's ``run_wilor_3d.py``**: same CLI shape, same
``<stem>_3d_keypoints.npz`` schema, same ``<stem>_3d_meta.json`` sidecar. Point the existing
``fuse_2d_3d.py`` / ``postprocess.py`` / ``render_overlay.py`` at its output and they work unchanged.

Why EgoForce instead of WiLoR, and what actually changes:

* **No focal-length hack.** WiLoR reconstructs with a virtual focal length and needs
  ``focal_length = f0 * 256 / max(W, H)`` passed in to land in the true camera. EgoForce's ray space
  solver consumes the real intrinsics directly (``core/rss.py::unproject_unit_rays``), so the true
  ``fx, fy, cx, cy`` are used as-is and anisotropic pixels are supported rather than averaged to
  ``f0``.
* **No principal-point rebase.** WiLoR pins the principal point to the image centre, so its 3D has
  to be rebased into the true-K camera afterwards. EgoForce never makes that assumption. We still
  write ``pp_rebase=True``, because downstream that flag means "``kp3d_cam`` is expressed in the
  camera described by the stored ``K``" - which holds here by construction. The run's
  ``reproj_median_px`` self-check proves it.
* **Handedness is structural, not guessed.** EgoForce runs a separate left and right hand/forearm
  crop pair, so ``is_right`` comes from which stream produced the row. WiLoR's and MediaPipe's
  per-crop handedness guess is the thing the existing pipeline has to repair with track-level
  geometric voting.
* **The forearm is an input.** When the wrist is occluded or out of frame the forearm crop still
  constrains the hand, and a hand-conditioned prior fills in when the forearm is missing too.

Extra keys beyond the WiLoR schema (all additive, all ignored by the existing loaders):
``kp2d_head``/``kp2d_conf`` (the network's own 2D keypoint head and its per-joint confidence, an
observation independent of the 3D lift), ``mano_betas``/``mano_global_orient``/``mano_hand_pose``/
``mano_transl`` (needed by ``fuse_egoforce_rtmpose.py`` to refit articulation), ``arm_kp3d``/
``arm_kp2d``/``arm_bbox``/``arm_visible``.

Input is an ALREADY-RECTIFIED pinhole video plus its true pinhole intrinsics, matching the existing
pipeline. ``--camera-model`` exists for the non-rectified case but is not what the mono pipeline
feeds.

Example
-------
    python fusion/run_egoforce_3d.py \
        --video /path/to/video_rectified.mp4 \
        --out /path/to/work \
        --K 736.6 736.6 928.57 540.0 \
        --overlay
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEMO_DIR = os.path.join(ROOT_DIR, 'demo')
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

import argparse
import contextlib
import io
import json
import time

import cv2
import numpy as np

from fusion.calibration import read_K
from fusion.topology import (NUM_ARM_JOINTS, NUM_HAND_JOINTS, draw_forearm, draw_hand, pinhole_K,
                             project_pinhole)
from fusion.video_io import OverlayVideoWriter, iter_frames, open_video

# EgoForce's demo detector applies a hardcoded RTMDet score floor here, which is what supplies the
# FOREARM boxes. --hand-conf below reaches the YOLO hand detector (the one that supplies the hand
# boxes); the forearm floor is not currently parameterised. See plan_of_action.md.
RTMDET_SCORE_FLOOR = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description='EgoForce 3D hand keypoints in the mono-pipeline stage-1b npz schema.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--video', required=True, help='rectified pinhole video')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--stem', default=None, help='output basename (default: video stem)')
    intr = parser.add_mutually_exclusive_group(required=True)
    intr.add_argument('--K', type=float, nargs=4, default=None, metavar=('FX', 'FY', 'CX', 'CY'),
                      help='true pinhole intrinsics of the rectified stream')
    intr.add_argument('--calib', default=None,
                      help='calibration.json to read fx,fy,cx,cy from (Standard Package v2 schema)')
    parser.add_argument('--calib-block', default='rectified', choices=['rectified', 'raw'])
    parser.add_argument('--eye', default='left', choices=['left', 'right'])

    parser.add_argument('--start-sec', type=float, default=0.0)
    parser.add_argument('--duration-sec', type=float, default=1e9)
    parser.add_argument('--sample-fps', type=float, default=0.0, help='0 = every frame (native fps)')
    parser.add_argument('--device', default='cuda', help='recorded for provenance; EgoForce needs CUDA')

    parser.add_argument('--hand-conf', type=float, default=0.25,
                        help="YOLO hand-detector confidence (Inference.yolo_track_cfg['conf'])")
    parser.add_argument('--max-misses', type=int, default=2,
                        help='frames a hand track survives without a detection before it is dropped. '
                             'Raise it to trade false positives for recall on fast motion.')

    parser.add_argument('--camera-model', default='pinhole',
                        choices=['pinhole', 'rational8', 'fisheye624'],
                        help='non-pinhole only makes sense on a NON-rectified stream')
    parser.add_argument('--distortion', type=float, nargs='+', default=None,
                        help='distortion coefficients for rational8 (8) or fisheye624 (12)')
    parser.add_argument('--no-undistort-inp', action='store_true',
                        help='feed distorted crops to the network (matches the eval ablation)')
    parser.add_argument('--no-kalman', action='store_true',
                        help='disable EgoForce translation smoothing (Kalman is on by default)')

    parser.add_argument('--overlay', action='store_true', help='also write a review mp4')
    parser.add_argument('--draw-forearm', action='store_true', help='draw the forearm chain too')
    parser.add_argument('--verbose', action='store_true', help='let per-frame timing prints through')
    return parser.parse_args()


def build_camera_model(args, width, height):
    # Imported here, not at module scope: camera_models/__init__.py pulls in the pytorch3d camera
    # wrappers, so an eager import makes even `--help` require pytorch3d. Deferring it lets the CLI
    # and its argument validation run on any machine.
    from camera_models import OVR624CameraModel, PinholeCameraModel, Rational8CameraModel

    fx, fy, cx, cy = args.K
    focal = np.array([fx, fy], dtype=np.float32)
    principal = np.array([cx, cy], dtype=np.float32)

    if args.camera_model == 'pinhole':
        if args.distortion:
            raise SystemExit('--distortion is meaningless with --camera-model pinhole')
        return PinholeCameraModel(focal, principal, width, height)

    if args.camera_model == 'rational8':
        dist = np.zeros(8, dtype=np.float32)
        if args.distortion:
            supplied = np.asarray(args.distortion, dtype=np.float32)
            if supplied.size > 8:
                raise SystemExit('rational8 takes at most 8 distortion coefficients')
            dist[:supplied.size] = supplied
        return Rational8CameraModel(focal, principal, dist, width, height)

    dist = np.zeros(12, dtype=np.float32)
    if args.distortion:
        supplied = np.asarray(args.distortion, dtype=np.float32)
        if supplied.size > 12:
            raise SystemExit('fisheye624 takes at most 12 distortion coefficients')
        dist[:supplied.size] = supplied
    return OVR624CameraModel(focal, principal, dist, width, height)


def main():
    args = parse_args()

    if args.calib:
        args.K, calib_block = read_K(args.calib, args.calib_block, args.eye)
        print(f'[3d] K from {args.calib} ({calib_block} block, {args.eye} eye): '
              f'fx={args.K[0]:.3f} fy={args.K[1]:.3f} cx={args.K[2]:.3f} cy={args.K[3]:.3f}')
    else:
        calib_block = 'cli'

    capture, info = open_video(args.video, args.start_sec, args.duration_sec, args.sample_fps)
    fx, fy, cx, cy = args.K
    K = pinhole_K(fx, fy, cx, cy)
    stem = args.stem or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    if abs(cx - info.width / 2.0) > 0.25 * info.width:
        print(f'[warn] cx={cx:.1f} is far from W/2={info.width / 2:.1f} - check the calibration '
              f'matches this video (right eye vs left eye?)')

    camera_model = build_camera_model(args, info.width, info.height)
    print(f'[3d {stem}] {info.width}x{info.height}@{info.fps:.2f} '
          f'fx={fx:.2f} fy={fy:.2f} pp=({cx:.1f},{cy:.1f}) '
          f'frames[{info.start_frame},{info.end_frame}) step={info.step} '
          f'camera={args.camera_model} undistort_inp={not args.no_undistort_inp} '
          f'kalman={not args.no_kalman}')

    # Imported here so --help and argument errors do not pay the TensorRT / mmdet import cost.
    from inference import Inference

    print(f'[3d {stem}] loading EgoForce (detectors + HALO; TensorRT compile on first run)...')
    inference = Inference(camera_model=camera_model, undistort_inp=not args.no_undistort_inp)
    inference.enable_kalman_filter = not args.no_kalman
    inference.set_kalman_filter_frequency(info.output_fps)
    inference.reset_runtime_state()
    inference.yolo_track_cfg['conf'] = float(args.hand_conf)
    inference.grouped_hand_track_max_misses = int(args.max_misses)
    device = inference.device
    if device.type != 'cuda':
        print('[warn] CUDA is not available; EgoForce is not supported on CPU (torch_tensorrt).')

    writer = None
    if args.overlay:
        writer = OverlayVideoWriter(os.path.join(args.out, f'{stem}_3d_overlay.mp4'),
                                    info.width, info.height, info.output_fps)
        print(f'[3d {stem}] overlay via {writer.backend}')

    rows = {k: [] for k in ('kp3d', 'kp2d', 'kp2d_head', 'kp2d_conf', 'fidx', 'isr', 'bbox', 'camt',
                            'betas', 'gorient', 'hpose', 'transl',
                            'arm_kp3d', 'arm_kp2d', 'arm_bbox', 'arm_visible')}
    processed, reproj_err, head_vs_proj, failures = [], [], [], 0
    n_frames = n_det = 0
    t0 = time.time()

    try:
        for frame_index, frame_bgr in iter_frames(capture, info):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed.append(frame_index)
            n_frames += 1

            try:
                if args.verbose:
                    outs = inference.run_outputs(rgb, device)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        outs = inference.run_outputs(rgb, device)
            except Exception as exc:
                failures += 1
                print(f'  [3d {stem}] frame {frame_index}: inference failed '
                      f'({type(exc).__name__}: {exc})')
                if writer is not None:
                    writer.write(frame_bgr)
                continue

            visible = np.asarray(outs['visible_hand'], dtype=bool).reshape(-1)
            vis = frame_bgr if writer is None else frame_bgr.copy()

            for hdx in range(len(visible)):
                if not visible[hdx]:
                    continue
                k3 = np.asarray(outs['pred_j3d'][hdx], dtype=np.float64)[:NUM_HAND_JOINTS]
                k2 = np.asarray(outs['pred_j2d'][hdx], dtype=np.float64)[:NUM_HAND_JOINTS]
                k2h = np.asarray(outs['pred_hand_j2d_head'][hdx], dtype=np.float64)[:NUM_HAND_JOINTS]

                # Self-check: project the stored 3D with the stored K and compare to the stored 2D.
                # This is the one test that proves kp3d_cam and K describe the same camera; a
                # principal-point or focal mismatch shows up here immediately.
                uv = project_pinhole(k3, K)
                reproj_err.append(float(np.median(np.linalg.norm(uv - k2, axis=1))))
                # And how far the network's own 2D head sits from its own 3D lift. Not a correctness
                # check - a genuine disagreement signal, useful for weighting the fusion.
                head_vs_proj.append(float(np.median(np.linalg.norm(k2h - uv, axis=1))))

                is_right = int(outs['hand_type'][hdx])
                rows['kp3d'].append(k3.astype(np.float32))
                rows['kp2d'].append(k2.astype(np.float32))
                rows['kp2d_head'].append(k2h.astype(np.float32))
                rows['kp2d_conf'].append(np.asarray(outs['pred_hand_kpt_conf'][hdx],
                                                    dtype=np.float32)[:NUM_HAND_JOINTS])
                rows['fidx'].append(frame_index)
                rows['isr'].append(is_right)
                rows['bbox'].append(np.asarray(outs['hand_bbox'][hdx], dtype=np.float32))
                rows['camt'].append(np.asarray(outs['pred_transl'][hdx], dtype=np.float32))
                rows['betas'].append(np.asarray(outs['pred_betas'][hdx], dtype=np.float32))
                rows['gorient'].append(np.asarray(outs['pred_global_orient'][hdx], dtype=np.float32))
                rows['hpose'].append(np.asarray(outs['pred_hand_pose'][hdx], dtype=np.float32))
                rows['transl'].append(np.asarray(outs['pred_transl'][hdx], dtype=np.float32))
                rows['arm_kp3d'].append(
                    np.asarray(outs['pred_arm_j3d'][hdx], dtype=np.float32)[:NUM_ARM_JOINTS])
                rows['arm_kp2d'].append(
                    np.asarray(outs['pred_arm_j2d'][hdx], dtype=np.float32)[:NUM_ARM_JOINTS])
                rows['arm_bbox'].append(np.asarray(outs['arm_bbox'][hdx], dtype=np.float32))
                rows['arm_visible'].append(bool(outs['visible_arm'][hdx]))
                n_det += 1

                if writer is not None:
                    if args.draw_forearm:
                        draw_forearm(vis, outs['pred_arm_j2d'][hdx])
                    draw_hand(vis, k2, 'right' if is_right else 'left')

            if writer is not None:
                cv2.putText(vis, f'f{frame_index} EgoForce hands={int(visible.sum())}', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                writer.write(vis)

            if n_frames % 200 == 0:
                print(f'  [3d {stem}] {n_frames} frames, {n_det} dets, '
                      f'{n_frames / max(time.time() - t0, 1e-6):.2f} fps', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.close()

    def stack(key, shape, dtype=np.float32):
        if rows[key]:
            return np.asarray(rows[key], dtype=dtype)
        return np.zeros((0, *shape), dtype=dtype)

    if reproj_err:
        r = float(np.median(reproj_err))
        print(f'[3d {stem}] reprojection self-check: median {r:.4f} px '
              f"({'OK' if r < 1.0 else 'HIGH - K/3D mismatch'})")
    if head_vs_proj:
        print(f'[3d {stem}] 2D head vs 3D lift disagreement: median '
              f'{float(np.median(head_vs_proj)):.2f} px')

    npz_path = os.path.join(args.out, f'{stem}_3d_keypoints.npz')
    np.savez_compressed(
        npz_path,
        # --- the WiLoR stage-1b schema, verbatim
        kp3d_cam=stack('kp3d', (NUM_HAND_JOINTS, 3)),
        kp2d=stack('kp2d', (NUM_HAND_JOINTS, 2)),
        frame_idx=np.asarray(rows['fidx'], dtype=np.int32),
        is_right=np.asarray(rows['isr'], dtype=np.int8),
        bbox=stack('bbox', (4,)),
        cam_t=stack('camt', (3,)),
        processed_frames=np.asarray(processed, dtype=np.int32),
        K=K, width=np.int32(info.width), height=np.int32(info.height),
        fps=np.float32(info.fps), step=np.int32(info.step),
        sample_fps=np.float32(info.sample_fps),
        start_frame=np.int32(info.start_frame), end_frame=np.int32(info.end_frame),
        # True in the sense the flag carries downstream: kp3d_cam is expressed in the camera the
        # stored K describes. EgoForce needs no rebase to get there.
        pp_rebase=np.bool_(True),
        model=np.array('egoforce'),
        # --- additive
        kp2d_head=stack('kp2d_head', (NUM_HAND_JOINTS, 2)),
        kp2d_conf=stack('kp2d_conf', (NUM_HAND_JOINTS,)),
        mano_betas=stack('betas', (10,)),
        mano_global_orient=stack('gorient', (6,)),
        mano_hand_pose=stack('hpose', (90,)),
        mano_transl=stack('transl', (3,)),
        mano_rot_format=np.array('rotation_6d'),
        arm_kp3d=stack('arm_kp3d', (NUM_ARM_JOINTS, 3)),
        arm_kp2d=stack('arm_kp2d', (NUM_ARM_JOINTS, 2)),
        arm_bbox=stack('arm_bbox', (4,)),
        arm_visible=np.asarray(rows['arm_visible'], dtype=bool),
        keypoint_order=np.array('OpenPose-21'),
    )

    z_all = np.asarray(rows['kp3d'])[..., 2] if rows['kp3d'] else None
    meta = dict(
        model='EgoForce (forearm-guided camera-space 3D hand), rectified pinhole',
        video=os.path.basename(args.video), width=info.width, height=info.height, fps=info.fps,
        step=info.step, start_frame=info.start_frame, end_frame=info.end_frame,
        frames_processed=n_frames, total_detections=n_det, inference_failures=failures,
        device=str(getattr(inference, 'device', args.device)),
        hand_conf=args.hand_conf, max_misses=args.max_misses,
        rtmdet_forearm_score_floor=RTMDET_SCORE_FLOOR,
        camera_model=args.camera_model, undistort_inp=not args.no_undistort_inp,
        kalman=not args.no_kalman, kalman_freq_hz=info.output_fps,
        K=K.tolist(), K_source=calib_block, pp_rebase=True,
        pp_rebase_note=('EgoForce solves translation against the true intrinsics, so no rebase is '
                        'applied; the flag records that kp3d_cam already lives in the stored-K '
                        'camera.'),
        reproj_median_px=float(np.median(reproj_err)) if reproj_err else None,
        head_vs_lift_median_px=float(np.median(head_vs_proj)) if head_vs_proj else None,
        depth_Z_m=(dict(min=round(float(z_all.min()), 6),
                        p1=round(float(np.percentile(z_all, 1)), 4),
                        median=round(float(np.median(z_all)), 4),
                        p99=round(float(np.percentile(z_all, 99)), 4),
                        max=round(float(z_all.max()), 4),
                        frac_below_5cm=round(float((z_all < 0.05).mean()), 6))
                   if z_all is not None and z_all.size else None),
        keypoint_order='OpenPose-21 (0=wrist;1-4 thumb;5-8 index;9-12 middle;13-16 ring;17-20 pinky)',
        handedness_source='structural (separate left/right crop streams), not a per-crop guess',
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(args.out, f'{stem}_3d_meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=2)

    print(f'[3d {stem}] DONE {n_frames} frames, {n_det} dets in {time.time() - t0:.0f}s')
    print(f'[3d {stem}] wrote {npz_path}')


if __name__ == '__main__':
    main()
