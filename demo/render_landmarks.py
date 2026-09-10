#!/usr/bin/env python3
"""Render 21-joint hand landmarks from a monocular RGB video.

Runs EgoForce frame by frame over a video file and produces:

  1. an overlay video with the 21-joint hand skeleton drawn on the input frames, and
  2. an ``.npz`` with the camera-space 3D joints, the projected 2D joints and
     per-hand visibility flags for every processed frame.

Unlike ``run_app.py`` this script needs no Gradio and no pytorch3d rasteriser --
the overlay is drawn with OpenCV from ``pred_j2d``, so only the ray space solver
(which does pull in the pytorch3d camera models) is required.

Camera intrinsics come either from AnyCalib on the first processed frame (the
default, mirroring the Gradio demo) or from explicit ``--focal``/``--principal``
values when the rig is already calibrated.

Examples
--------
Uncalibrated video, first 10 seconds, fisheye lens model::

    python demo/render_landmarks.py \
        --video /path/to/input.mp4 \
        --output-video _DATA/outputs/input_landmarks.mp4 \
        --output-keypoints _DATA/outputs/input_landmarks.npz \
        --duration-seconds 10

Known pinhole rig (skips AnyCalib entirely)::

    python demo/render_landmarks.py \
        --video /path/to/input.mp4 \
        --camera-model pinhole \
        --focal 736.6 736.6 \
        --principal 960.0 540.0
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from camera_models import OVR624CameraModel, PinholeCameraModel, Rational8CameraModel


# ---------------------------------------------------------------------------
# Joint layout
#
# ``models/mano_layer.py`` remaps the MANO joints through
# ``mano_joint_mapping = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]``
# with fingertips appended in the order thumb, index, middle, ring, pinky. That
# lands on the usual 21-joint hand layout: wrist first, then each finger from
# MCP out to the tip.
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'wrist',
    'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

# (name, joint chain, BGR colour)
FINGER_CHAINS = [
    ('thumb', (0, 1, 2, 3, 4), (60, 76, 231)),
    ('index', (0, 5, 6, 7, 8), (49, 176, 243)),
    ('middle', (0, 9, 10, 11, 12), (72, 201, 116)),
    ('ring', (0, 13, 14, 15, 16), (219, 152, 52)),
    ('pinky', (0, 17, 18, 19, 20), (182, 89, 155)),
]

PALM_CHAIN = (5, 9, 13, 17)
PALM_COLOR = (190, 190, 190)

HAND_ORDER = ['left', 'right']
HAND_LABEL_COLORS = {'left': (255, 214, 120), 'right': (140, 210, 255)}

NUM_HAND_JOINTS = 21
NUM_ARM_JOINTS = 3

ANYCALIB_LENS_SPECS = {
    'fisheye624': {
        'label': 'Fisheye',
        'model_id': 'anycalib_gen',
        'cam_id': 'simple_kb:4',
        'repo_camera': 'fisheye624',
    },
    'pinhole_distortion': {
        'label': 'Pinhole + Distortion',
        'model_id': 'anycalib_dist',
        'cam_id': 'radial:4',
        'repo_camera': 'rational8',
    },
    'pinhole': {
        'label': 'Pinhole',
        'model_id': 'anycalib_pinhole',
        'cam_id': 'pinhole',
        'repo_camera': 'pinhole',
    },
}


def skeleton_edges():
    """Return the (E, 2) joint-index pairs used to draw the hand skeleton."""
    edges = []
    for _, chain, _ in FINGER_CHAINS:
        edges.extend(zip(chain[:-1], chain[1:]))
    edges.extend(zip(PALM_CHAIN[:-1], PALM_CHAIN[1:]))
    return np.asarray(edges, dtype=np.int32)


# ---------------------------------------------------------------------------
# Camera model construction
# ---------------------------------------------------------------------------

def parse_anycalib_intrinsics(intrinsics, lens_mode):
    """Mirror of ``run_app.parse_anycalib_intrinsics``.

    Kept local so this script does not import the Gradio app (which pulls in
    ``gradio`` and the HF Spaces-only ``spaces`` module).
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(-1)

    if lens_mode == 'pinhole':
        if intrinsics.size >= 4:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
        elif intrinsics.size >= 3:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
        else:
            raise ValueError(f"Expected 3 or 4 intrinsics for pinhole, got {intrinsics.size}.")
        return focal, principal, None

    if lens_mode == 'pinhole_distortion':
        if intrinsics.size >= 8:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
            radial = intrinsics[4:8]
        elif intrinsics.size >= 7:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
            radial = intrinsics[3:7]
        else:
            raise ValueError(f"Expected 7 or 8 intrinsics for pinhole+distortion, got {intrinsics.size}.")

        distortion = np.zeros(8, dtype=np.float32)
        distortion[0] = radial[0]
        distortion[1] = radial[1]
        distortion[4] = radial[2]
        distortion[5] = radial[3]
        return focal, principal, distortion

    if lens_mode == 'fisheye624':
        if intrinsics.size >= 8:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
            kb = intrinsics[4:8]
        elif intrinsics.size >= 7:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
            kb = intrinsics[3:7]
        else:
            raise ValueError(f"Expected 7 or 8 intrinsics for fisheye624, got {intrinsics.size}.")

        distortion = np.zeros(12, dtype=np.float32)
        distortion[:4] = kb
        return focal, principal, distortion

    raise ValueError(f"Unsupported lens mode: {lens_mode}")


def build_camera_model(repo_camera, focal, principal, distortion, width, height):
    """Instantiate the repo camera model named by ``repo_camera``."""
    focal = np.asarray(focal, dtype=np.float32).reshape(-1)
    principal = np.asarray(principal, dtype=np.float32).reshape(-1)

    if repo_camera == 'pinhole':
        return PinholeCameraModel(focal, principal, width, height)

    if repo_camera == 'rational8':
        dist = np.zeros(8, dtype=np.float32)
        if distortion is not None:
            supplied = np.asarray(distortion, dtype=np.float32).reshape(-1)
            dist[:min(8, supplied.size)] = supplied[:8]
        return Rational8CameraModel(focal, principal, dist, width, height)

    if repo_camera == 'fisheye624':
        dist = np.zeros(12, dtype=np.float32)
        if distortion is not None:
            supplied = np.asarray(distortion, dtype=np.float32).reshape(-1)
            dist[:min(12, supplied.size)] = supplied[:12]
        return OVR624CameraModel(focal, principal, dist, width, height)

    raise ValueError(f"Unsupported camera model: {repo_camera}")


def camera_model_from_anycalib(rgb_frame, lens_mode):
    """Estimate intrinsics from a single RGB frame with AnyCalib."""
    from anycalib import AnyCalib

    spec = ANYCALIB_LENS_SPECS.get(lens_mode)
    if spec is None:
        raise ValueError(f"Unknown lens mode: {lens_mode}")

    if rgb_frame.ndim != 3 or rgb_frame.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame shaped (H, W, 3), got {tuple(rgb_frame.shape)}.")

    height, width = rgb_frame.shape[:2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = torch.tensor(rgb_frame, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    anycalib_model = None
    try:
        anycalib_model = AnyCalib(model_id=spec['model_id']).to(device)
        with torch.no_grad():
            prediction = anycalib_model.predict(image, cam_id=spec['cam_id'])
        intrinsics = prediction['intrinsics']
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().cpu().numpy()
    finally:
        del image
        if anycalib_model is not None:
            del anycalib_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    focal, principal, distortion = parse_anycalib_intrinsics(intrinsics, lens_mode)
    camera_model = build_camera_model(
        spec['repo_camera'], focal, principal, distortion, width, height
    )

    info = {
        'source': 'anycalib',
        'lens_mode': lens_mode,
        'anycalib_model_id': spec['model_id'],
        'anycalib_cam_id': spec['cam_id'],
        'repo_camera': spec['repo_camera'],
        'image_size': [int(width), int(height)],
        'focal': np.asarray(focal, dtype=np.float32).tolist(),
        'principal': np.asarray(principal, dtype=np.float32).tolist(),
        'distortion': None if distortion is None else np.asarray(distortion, dtype=np.float32).tolist(),
    }
    return camera_model, info


def camera_model_from_args(args, width, height):
    """Build the camera model from explicit CLI intrinsics."""
    camera_model = build_camera_model(
        args.camera_model, args.focal, args.principal, args.distortion, width, height
    )
    info = {
        'source': 'cli',
        'lens_mode': None,
        'repo_camera': args.camera_model,
        'image_size': [int(width), int(height)],
        'focal': [float(v) for v in args.focal],
        'principal': [float(v) for v in args.principal],
        'distortion': None if args.distortion is None else [float(v) for v in args.distortion],
    }
    return camera_model, info


# ---------------------------------------------------------------------------
# Video IO
# ---------------------------------------------------------------------------

class OverlayVideoWriter:
    """Write BGR frames to an mp4, preferring an ffmpeg h264 pipe.

    OpenCV's bundled ``mp4v`` encoder is used as a fallback when ffmpeg is not
    on PATH; it works but produces files some players refuse to scrub.
    """

    def __init__(self, path, width, height, fps):
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.proc = None
        self.writer = None

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is not None:
            cmd = [
                ffmpeg, '-y',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', f'{self.fps:.6f}',
                '-i', '-',
                '-an',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                self.path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.backend = 'ffmpeg/libx264'
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open an OpenCV VideoWriter for {self.path}.")
            self.backend = 'opencv/mp4v'

    def write(self, frame_bgr):
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if self.proc is not None:
            self.proc.stdin.write(frame_bgr.tobytes())
        else:
            self.writer.write(frame_bgr)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _finite_point(xy, width, height, margin=64):
    """Return an int pixel tuple, or None when the point is unusable."""
    if not np.all(np.isfinite(xy)):
        return None
    x, y = float(xy[0]), float(xy[1])
    if x < -margin or y < -margin or x > width + margin or y > height + margin:
        return None
    return int(round(x)), int(round(y))


def draw_hand(frame_bgr, j2d, hand_name, line_thickness=2, joint_radius=3, draw_palm=True):
    """Draw one 21-joint hand skeleton onto a BGR frame in place."""
    height, width = frame_bgr.shape[:2]
    points = [_finite_point(j2d[i], width, height) for i in range(NUM_HAND_JOINTS)]
    is_left = hand_name == 'left'

    if draw_palm:
        for a, b in zip(PALM_CHAIN[:-1], PALM_CHAIN[1:]):
            if points[a] is not None and points[b] is not None:
                cv2.line(frame_bgr, points[a], points[b], PALM_COLOR, max(1, line_thickness - 1), cv2.LINE_AA)

    for _, chain, color in FINGER_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            if points[a] is not None and points[b] is not None:
                cv2.line(frame_bgr, points[a], points[b], color, line_thickness, cv2.LINE_AA)

    for _, chain, color in FINGER_CHAINS:
        for idx in chain[1:]:
            pt = points[idx]
            if pt is None:
                continue
            cv2.circle(frame_bgr, pt, joint_radius, color, -1, cv2.LINE_AA)
            if not is_left:
                # Right-hand joints get an outer ring so the two hands stay
                # distinguishable where they overlap.
                cv2.circle(frame_bgr, pt, joint_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

    wrist = points[0]
    if wrist is not None:
        label_color = HAND_LABEL_COLORS[hand_name]
        cv2.circle(frame_bgr, wrist, joint_radius + 3, label_color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, wrist, joint_radius + 5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(
            frame_bgr, 'L' if is_left else 'R',
            (wrist[0] + 10, wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr, 'L' if is_left else 'R',
            (wrist[0] + 10, wrist[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA,
        )


def draw_forearm(frame_bgr, arm_j2d, line_thickness=2):
    """Draw the 3-joint forearm chain onto a BGR frame in place."""
    height, width = frame_bgr.shape[:2]
    points = [_finite_point(arm_j2d[i], width, height) for i in range(min(NUM_ARM_JOINTS, len(arm_j2d)))]
    color = (150, 150, 150)
    for a in range(len(points) - 1):
        if points[a] is not None and points[a + 1] is not None:
            cv2.line(frame_bgr, points[a], points[a + 1], color, line_thickness, cv2.LINE_AA)
    for pt in points:
        if pt is not None:
            cv2.circle(frame_bgr, pt, line_thickness + 1, color, -1, cv2.LINE_AA)


def draw_hud(frame_bgr, frame_index, timestamp_s, j3d, visible, failed=False):
    """Draw a small status readout: frame, time, per-hand wrist depth."""
    lines = [f'frame {frame_index}  t={timestamp_s:6.2f}s']
    if failed:
        lines.append('inference failed on this frame')
    else:
        for hdx, hand_name in enumerate(HAND_ORDER):
            if not visible[hdx]:
                lines.append(f'{hand_name:<5} not detected')
                continue
            wrist = j3d[hdx, 0]
            if np.all(np.isfinite(wrist)):
                depth_cm = float(np.linalg.norm(wrist)) * 100.0
                lines.append(f'{hand_name:<5} wrist {depth_cm:6.1f} cm')
            else:
                lines.append(f'{hand_name:<5} wrist n/a')

    x, y = 12, 26
    for line in lines:
        cv2.putText(frame_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 24


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run EgoForce on a video and render 21-joint hand landmarks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--video', required=True, help='Input video path.')
    parser.add_argument('--output-video', default=None,
                        help='Overlay mp4 path. Defaults to <video stem>_landmarks.mp4 next to --output-dir.')
    parser.add_argument('--output-keypoints', default=None,
                        help='Keypoint npz path. Defaults to <video stem>_landmarks.npz next to --output-dir.')
    parser.add_argument('--output-dir', default=os.path.join(ROOT_DIR, '_DATA', 'outputs'),
                        help='Directory used when --output-video/--output-keypoints are not given.')

    parser.add_argument('--start-seconds', type=float, default=0.0,
                        help='Seek this far into the video before processing.')
    parser.add_argument('--duration-seconds', type=float, default=None,
                        help='Process at most this many seconds of video.')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Process at most this many frames.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Process every Nth frame. Output fps is scaled to match.')

    parser.add_argument('--lens', default='fisheye624', choices=sorted(ANYCALIB_LENS_SPECS.keys()),
                        help='AnyCalib lens model used when intrinsics are not supplied.')
    parser.add_argument('--camera-model', default=None,
                        choices=['pinhole', 'rational8', 'fisheye624'],
                        help='Skip AnyCalib and use this repo camera model with --focal/--principal.')
    parser.add_argument('--focal', type=float, nargs=2, default=None, metavar=('FX', 'FY'),
                        help='Focal length in pixels. Required with --camera-model.')
    parser.add_argument('--principal', type=float, nargs=2, default=None, metavar=('CX', 'CY'),
                        help='Principal point in pixels. Required with --camera-model.')
    parser.add_argument('--distortion', type=float, nargs='+', default=None,
                        help='Distortion coefficients for rational8 (8) or fisheye624 (12).')

    parser.add_argument('--no-undistort-inp', action='store_true',
                        help='Feed distorted crops to the network (matches the --no-undistort-inp ablation).')
    parser.add_argument('--no-kalman', action='store_true',
                        help='Disable the translation Kalman filter.')
    parser.add_argument('--draw-forearm', action='store_true',
                        help='Also draw the predicted 3-joint forearm chain.')
    parser.add_argument('--no-hud', action='store_true', help='Hide the frame/depth readout.')
    parser.add_argument('--line-thickness', type=int, default=2, help='Skeleton line thickness in pixels.')
    parser.add_argument('--joint-radius', type=int, default=3, help='Joint marker radius in pixels.')
    parser.add_argument('--skip-video', action='store_true',
                        help='Only dump keypoints, do not render the overlay video.')
    parser.add_argument('--verbose', action='store_true',
                        help='Let the per-frame EgoForce timing prints through.')
    return parser.parse_args()


def validate_args(args):
    if args.camera_model is not None:
        if args.focal is None or args.principal is None:
            raise SystemExit('--camera-model requires both --focal and --principal.')
        if args.camera_model == 'rational8' and args.distortion is not None and len(args.distortion) > 8:
            raise SystemExit('rational8 takes at most 8 distortion coefficients.')
        if args.camera_model == 'fisheye624' and args.distortion is not None and len(args.distortion) > 12:
            raise SystemExit('fisheye624 takes at most 12 distortion coefficients.')
    elif args.focal is not None or args.principal is not None or args.distortion is not None:
        raise SystemExit('--focal/--principal/--distortion require --camera-model.')

    if args.stride < 1:
        raise SystemExit('--stride must be >= 1.')


def resolve_outputs(args):
    stem = Path(args.video).stem.replace(' ', '_')
    out_dir = Path(args.output_dir)
    video_path = Path(args.output_video) if args.output_video else out_dir / f'{stem}_landmarks.mp4'
    kpts_path = Path(args.output_keypoints) if args.output_keypoints else out_dir / f'{stem}_landmarks.npz'
    video_path.parent.mkdir(parents=True, exist_ok=True)
    kpts_path.parent.mkdir(parents=True, exist_ok=True)
    return video_path, kpts_path


def main():
    args = parse_args()
    validate_args(args)

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f'Input video not found: {video_path}')

    out_video_path, out_kpts_path = resolve_outputs(args)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f'Could not open video: {video_path}')

    src_fps = capture.get(cv2.CAP_PROP_FPS)
    if not src_fps or not np.isfinite(src_fps) or src_fps <= 0:
        print('WARNING: could not read fps from the container, assuming 30.')
        src_fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)

    ok, first_bgr = capture.read()
    if not ok or first_bgr is None:
        raise SystemExit('Could not read a frame at the requested start offset.')

    height, width = first_bgr.shape[:2]
    first_frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    print(f'Input : {video_path}')
    print(f'        {width}x{height} @ {src_fps:.3f} fps, {total_frames} frames total')
    print(f'        starting at frame {first_frame_index} ({args.start_seconds:.2f}s), stride {args.stride}')

    # --- camera model -----------------------------------------------------
    if args.camera_model is not None:
        camera_model, calib_info = camera_model_from_args(args, width, height)
        print(f'Camera: {args.camera_model} from CLI intrinsics')
    else:
        print(f'Camera: estimating intrinsics with AnyCalib (lens={args.lens})...')
        camera_model, calib_info = camera_model_from_anycalib(
            cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB), args.lens
        )
    print(f'        focal={calib_info["focal"]} principal={calib_info["principal"]}')
    if calib_info['distortion'] is not None:
        print(f'        distortion={calib_info["distortion"]}')

    # --- model ------------------------------------------------------------
    # Imported here so --help and argument errors do not pay for the
    # TensorRT / mmdet import cost.
    from inference import Inference

    print('Loading EgoForce (detectors + HALO, TensorRT compile on first run)...')
    inference = Inference(camera_model=camera_model, undistort_inp=not args.no_undistort_inp)
    inference.enable_kalman_filter = not args.no_kalman
    inference.set_kalman_filter_frequency(src_fps / args.stride)
    inference.reset_runtime_state()
    device = inference.device
    print(f'Device: {device}')

    # --- frame budget -----------------------------------------------------
    budget = None
    if args.duration_seconds is not None:
        budget = int(np.floor(args.duration_seconds * src_fps / args.stride))
    if args.max_frames is not None:
        budget = args.max_frames if budget is None else min(budget, args.max_frames)

    writer = None
    if not args.skip_video:
        writer = OverlayVideoWriter(out_video_path, width, height, src_fps / args.stride)
        print(f'Output: {out_video_path} via {writer.backend}')
    print(f'Output: {out_kpts_path}')

    all_j3d, all_j2d = [], []
    all_arm_j3d, all_arm_j2d = [], []
    all_visible, all_frame_index, all_timestamp, all_failed = [], [], [], []

    frame_bgr = first_bgr
    frame_index = first_frame_index
    processed = 0
    failures = 0
    start_time = time.time()

    progress = tqdm(total=budget, unit='frame', desc='EgoForce', disable=args.verbose)
    try:
        while frame_bgr is not None:
            if budget is not None and processed >= budget:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            j3d = np.full((2, NUM_HAND_JOINTS, 3), np.nan, dtype=np.float32)
            j2d = np.full((2, NUM_HAND_JOINTS, 2), np.nan, dtype=np.float32)
            arm_j3d = np.full((2, NUM_ARM_JOINTS, 3), np.nan, dtype=np.float32)
            arm_j2d = np.full((2, NUM_ARM_JOINTS, 2), np.nan, dtype=np.float32)
            visible = np.zeros((2,), dtype=bool)
            failed = False

            try:
                if args.verbose:
                    outs = inference.run_outputs(rgb, device)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        outs = inference.run_outputs(rgb, device)

                j3d[:] = np.asarray(outs['pred_j3d'], dtype=np.float32)[:, :NUM_HAND_JOINTS]
                j2d[:] = np.asarray(outs['pred_j2d'], dtype=np.float32)[:, :NUM_HAND_JOINTS]
                visible[:] = np.asarray(outs['visible_hand'], dtype=bool).reshape(-1)[:2]

                pred_arm_j3d = np.asarray(outs['pred_arm_j3d'], dtype=np.float32)
                pred_arm_j2d = np.asarray(outs['pred_arm_j2d'], dtype=np.float32)
                n_arm = min(NUM_ARM_JOINTS, pred_arm_j3d.shape[1])
                arm_j3d[:, :n_arm] = pred_arm_j3d[:, :n_arm]
                arm_j2d[:, :n_arm] = pred_arm_j2d[:, :n_arm]
            except Exception as exc:  # keep the render going past a bad frame
                failed = True
                failures += 1
                tqdm.write(f'frame {frame_index}: inference failed ({type(exc).__name__}: {exc})')

            timestamp_s = frame_index / src_fps

            if writer is not None:
                overlay = frame_bgr.copy()
                for hdx, hand_name in enumerate(HAND_ORDER):
                    if not visible[hdx]:
                        continue
                    if args.draw_forearm:
                        draw_forearm(overlay, arm_j2d[hdx], line_thickness=args.line_thickness)
                    draw_hand(
                        overlay, j2d[hdx], hand_name,
                        line_thickness=args.line_thickness,
                        joint_radius=args.joint_radius,
                    )
                if not args.no_hud:
                    draw_hud(overlay, frame_index, timestamp_s, j3d, visible, failed=failed)
                writer.write(overlay)

            all_j3d.append(j3d)
            all_j2d.append(j2d)
            all_arm_j3d.append(arm_j3d)
            all_arm_j2d.append(arm_j2d)
            all_visible.append(visible)
            all_frame_index.append(frame_index)
            all_timestamp.append(timestamp_s)
            all_failed.append(failed)

            processed += 1
            progress.update(1)

            for _ in range(args.stride):
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    frame_bgr = None
                    break
                frame_index += 1
    finally:
        progress.close()
        capture.release()
        if writer is not None:
            writer.close()

    if processed == 0:
        raise SystemExit('No frames were processed.')

    elapsed = time.time() - start_time
    np.savez_compressed(
        out_kpts_path,
        j3d=np.stack(all_j3d),
        j2d=np.stack(all_j2d),
        arm_j3d=np.stack(all_arm_j3d),
        arm_j2d=np.stack(all_arm_j2d),
        visible=np.stack(all_visible),
        frame_index=np.asarray(all_frame_index, dtype=np.int32),
        timestamp_s=np.asarray(all_timestamp, dtype=np.float32),
        failed=np.asarray(all_failed, dtype=bool),
        hand_order=np.asarray(HAND_ORDER),
        joint_names=np.asarray(JOINT_NAMES),
        skeleton_edges=skeleton_edges(),
        source_fps=np.float32(src_fps),
        output_fps=np.float32(src_fps / args.stride),
        image_size=np.asarray([width, height], dtype=np.int32),
        camera_type_id=np.int32(camera_model.TYPE_ID),
        calibration=np.asarray(json.dumps(calib_info)),
        units=np.asarray('j3d/arm_j3d in metres, camera space; j2d/arm_j2d in source-image pixels'),
    )

    detected = np.stack(all_visible)
    print()
    print(f'Processed {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} fps)')
    print(f'  left hand detected : {int(detected[:, 0].sum())}/{processed} frames')
    print(f'  right hand detected: {int(detected[:, 1].sum())}/{processed} frames')
    if failures:
        print(f'  frames where inference raised: {failures}')
    if writer is not None:
        print(f'Overlay video: {out_video_path}')
    print(f'Keypoints    : {out_kpts_path}')


if __name__ == '__main__':
    main()
