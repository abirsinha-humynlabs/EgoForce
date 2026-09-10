"""Side-by-side review overlay: what EgoForce predicted, what the refit changed, what RTMPose saw.

This exists because "which overlay looks smoothest" is the wrong question. To judge whether the
articulated refit is helping you have to see all three things on the same pixels at once:

* **grey, thin**  - EgoForce's raw 3D, projected. The prior.
* **coloured**    - the row that was actually written (refit when accepted, raw EgoForce otherwise).
* **white dots**  - RTMPose's independent 2D observation, the evidence being fitted to.

When the coloured skeleton has moved off the grey one and onto the dots, the refit did its job. When
it has moved off both, something is wrong and the reject gates in ``mano_refit.accept_refit`` did not
catch it. The per-hand HUD prints the source label and the numbers those gates used.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import collections

import cv2
import numpy as np

from fusion.topology import (HAND_EDGES, NUM_HAND_JOINTS, _finite_point, draw_hand, draw_points,
                             project_pinhole)
from fusion.video_io import OverlayVideoWriter, iter_frames, open_video

SOURCE_TAG = {
    'fused': 'EgoForce+RTMPose (refit)',
    'wilor_pnpfail': 'EgoForce (refit rejected)',
    'wilor': 'EgoForce only',
    'lifted_2d': 'RTMPose only (borrowed depth)',
}


def _draw_faint_skeleton(frame_bgr, kp2d, color=(120, 120, 120), thickness=1):
    height, width = frame_bgr.shape[:2]
    points = [_finite_point(kp2d[i], width, height) for i in range(NUM_HAND_JOINTS)]
    for a, b in HAND_EDGES:
        if points[a] is not None and points[b] is not None:
            cv2.line(frame_bgr, points[a], points[b], color, thickness, cv2.LINE_AA)


def render_comparison(video_path, fused, out_path, max_frames=None):
    """Render the comparison mp4.

    ``fused`` is the dict written by ``fuse_egoforce_rtmpose.py`` (or a loaded npz of it).
    """
    K = np.asarray(fused['K'], dtype=np.float64)
    width, height = int(fused['width']), int(fused['height'])
    step = int(fused['step'])

    by_frame = collections.defaultdict(list)
    for i, f in enumerate(np.asarray(fused['frame_idx'])):
        by_frame[int(f)].append(i)

    frames_present = sorted(by_frame)
    if not frames_present:
        raise SystemExit('nothing to render: the fused npz has no detections')

    capture, info = open_video(video_path, 0.0, 1e9, 0.0)
    if info.width != width or info.height != height:
        capture.release()
        raise SystemExit(f'video is {info.width}x{info.height} but the npz was built on '
                         f'{width}x{height}')
    info.step = step
    info.start_frame = frames_present[0]
    info.end_frame = frames_present[-1] + 1

    raw3d = np.asarray(fused['kp3d_cam_wilor_raw'])
    obs2d = np.asarray(fused.get('kp2d_rtmpose', np.full_like(np.asarray(fused['kp2d']), np.nan)))
    source = np.asarray(fused['source'])
    residual = np.asarray(fused['fuse_residual_px'], dtype=np.float64)
    bone_change = np.asarray(fused.get('refit_bone_change',
                                       np.full(len(source), np.nan)), dtype=np.float64)
    is_right = np.asarray(fused['is_right_wilor'], dtype=np.int64)
    is_right_2d = np.asarray(fused['is_right_mp'], dtype=np.int64)

    written = 0
    with OverlayVideoWriter(out_path, width, height, info.output_fps) as writer:
        try:
            for frame_index, frame_bgr in iter_frames(capture, info):
                vis = frame_bgr.copy()
                lines = [f'f{frame_index}']

                for row in by_frame.get(frame_index, []):
                    hand = is_right[row] if is_right[row] >= 0 else is_right_2d[row]
                    hand_name = 'right' if hand == 1 else 'left'

                    if np.isfinite(raw3d[row]).all():
                        _draw_faint_skeleton(vis, project_pinhole(raw3d[row], K))
                    if np.isfinite(obs2d[row]).any():
                        draw_points(vis, obs2d[row], (255, 255, 255), radius=2)
                    draw_hand(vis, np.asarray(fused['kp2d'])[row], hand_name)

                    tag = SOURCE_TAG.get(str(source[row]), str(source[row]))
                    detail = f'{hand_name[0].upper()}: {tag}'
                    if np.isfinite(residual[row]):
                        detail += f'  res={residual[row]:.1f}px'
                    if np.isfinite(bone_change[row]):
                        detail += f'  dbone={bone_change[row] * 100:.1f}%'
                    lines.append(detail)

                if not by_frame.get(frame_index):
                    lines.append('no detections')

                y = 28
                for line in lines:
                    cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (245, 245, 245), 1, cv2.LINE_AA)
                    y += 24

                legend = 'grey=EgoForce raw   colour=written   dots=RTMPose obs'
                cv2.putText(vis, legend, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(vis, legend, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (245, 245, 245), 1, cv2.LINE_AA)

                writer.write(vis)
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
        finally:
            capture.release()

    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Render the fusion comparison overlay.')
    parser.add_argument('--fused', required=True, help='_hand21_keypoints.npz')
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True, help='output mp4 path')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    fused = dict(np.load(args.fused, allow_pickle=False))
    render_comparison(args.video, fused, args.out, args.max_frames)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
