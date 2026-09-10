#!/usr/bin/env python3
"""Stage 1a (replacement) - 21 2D hand landmarks via RTMPose-m Hand5.

A **drop-in replacement for the mono-pipeline's ``run_mediapipe_2d.py``**: same
``<stem>_2d_keypoints.npz`` schema, so the existing ``fuse_2d_3d.py`` accepts it unchanged.

Verified facts this stage depends on (checked against the mmpose repo, not assumed):

* Config ``configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py``,
  256x256 input, 21 output channels, reported 96.4 PCK@0.2 / 83.9 AUC / 5.06 EPE on Hand5.
* It is trained against ``configs/_base_/datasets/coco_wholebody_hand.py``, whose keypoint order is
  ``wrist, thumb1..4, forefinger1..4, middle_finger1..4, ring_finger1..4, pinky_finger1..4`` -
  identical to OpenPose-21 and therefore to both MediaPipe and EgoForce. **No remapping.**
* The model's ``data_preprocessor`` sets ``bgr_to_rgb=True``, so frames are handed to
  ``inference_topdown`` as **BGR** (unlike the MediaPipe stage, which needs RGB).

Box sourcing - RTMPose is top-down, so it needs boxes:

``--boxes npz`` (default)
    Reuse the hand boxes EgoForce already detected, read from its ``_3d_keypoints.npz``. This is the
    tightest coupling available: every 2D row corresponds 1:1 to a 3D row on the same frame and the
    same physical hand, so the downstream fusion needs no wrist matching and cannot mis-pair. It also
    guarantees ``width``/``height``/``step`` agree with the 3D stage, which is the check
    ``fuse_2d_3d.py`` performs.

``--boxes mmdet``
    Run an independent mmdet hand detector. Use this for the "is the detector or the pose model the
    bottleneck?" comparison - in particular for the recurring foot false positives, where sharing
    EgoForce's detector would propagate the same error into both streams.

Example
-------
    python fusion/run_rtmpose_2d.py \
        --video /path/to/video_rectified.mp4 \
        --out /path/to/work \
        --boxes npz --boxes-npz /path/to/work/video_rectified_3d_keypoints.npz \
        --rtmpose-config _DATA/rtmpose_hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
        --rtmpose-checkpoint _DATA/rtmpose_hand5/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth
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

import cv2
import numpy as np

from fusion.topology import NUM_HAND_JOINTS, draw_hand
from fusion.video_io import OverlayVideoWriter, iter_frames, open_video

DEFAULT_CONFIG = os.path.join(ROOT_DIR, '_DATA', 'rtmpose_hand5',
                              'rtmpose-m_8xb256-210e_hand5-256x256.py')
DEFAULT_CHECKPOINT = os.path.join(
    ROOT_DIR, '_DATA', 'rtmpose_hand5',
    'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth')


def parse_args():
    parser = argparse.ArgumentParser(
        description='RTMPose-m Hand5 2D landmarks in the mono-pipeline stage-1a npz schema.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--video', required=True, help='rectified pinhole video')
    parser.add_argument('--out', required=True, help='output directory')
    parser.add_argument('--stem', default=None, help='output basename (default: video stem)')

    parser.add_argument('--rtmpose-config', default=DEFAULT_CONFIG)
    parser.add_argument('--rtmpose-checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--boxes', default='npz', choices=['npz', 'mmdet'],
                        help='where hand boxes come from')
    parser.add_argument('--boxes-npz', default=None,
                        help='EgoForce _3d_keypoints.npz (required for --boxes npz)')
    parser.add_argument('--det-config', default=None, help='mmdet hand detector config')
    parser.add_argument('--det-checkpoint', default=None, help='mmdet hand detector checkpoint')
    parser.add_argument('--det-score', type=float, default=0.3, help='mmdet score floor')
    parser.add_argument('--det-cat-ids', type=int, nargs='+', default=None,
                        help='restrict mmdet labels to these category ids (default: all)')
    parser.add_argument('--bbox-pad', type=float, default=1.0,
                        help='scale factor applied to every input box about its centre before pose '
                             'estimation. 1.0 keeps the box as given; RTMPose also applies its own '
                             'GetBBoxCenterScale padding on top.')

    # Only used with --boxes mmdet; with --boxes npz these come from the 3D npz so the two stages
    # cannot disagree.
    parser.add_argument('--start-sec', type=float, default=0.0)
    parser.add_argument('--duration-sec', type=float, default=1e9)
    parser.add_argument('--sample-fps', type=float, default=0.0, help='0 = every frame')

    parser.add_argument('--min-kpt-conf', type=float, default=0.0,
                        help='drop a whole detection whose MEAN keypoint confidence is below this')
    parser.add_argument('--overlay', action='store_true', help='also write a review mp4')
    return parser.parse_args()


def scale_box(box, factor):
    """Scale an xyxy box about its centre."""
    if factor == 1.0:
        return box
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * factor, 0.5 * (y2 - y1) * factor
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)


def load_boxes_from_npz(path):
    """Return ``(boxes_by_frame, provenance)`` from an EgoForce stage-1b npz.

    ``boxes_by_frame[frame] = [(bbox_xyxy, is_right), ...]``
    """
    data = np.load(path, allow_pickle=False)
    for key in ('frame_idx', 'bbox', 'is_right', 'width', 'height', 'step'):
        if key not in data:
            raise SystemExit(f'{path} is missing "{key}" - is it a stage-1b 3D npz?')

    boxes = collections.defaultdict(list)
    for i in range(len(data['frame_idx'])):
        boxes[int(data['frame_idx'][i])].append(
            (np.asarray(data['bbox'][i], dtype=np.float32), int(data['is_right'][i])))

    provenance = dict(
        width=int(data['width']), height=int(data['height']), step=int(data['step']),
        fps=float(data['fps']), start_frame=int(data['start_frame']),
        end_frame=int(data['end_frame']), sample_fps=float(data['sample_fps']),
        processed_frames=np.asarray(data['processed_frames'], dtype=np.int32),
        source_model=str(data['model']) if 'model' in data else 'unknown',
    )
    return boxes, provenance


def build_detector(args):
    from mmdet.apis import init_detector

    if not args.det_config or not args.det_checkpoint:
        raise SystemExit('--boxes mmdet requires --det-config and --det-checkpoint')
    return init_detector(args.det_config, args.det_checkpoint, device=args.device)


def detect_boxes(detector, frame_bgr, score_thr, cat_ids):
    from mmdet.apis import inference_detector

    result = inference_detector(detector, frame_bgr)
    inst = result.pred_instances
    bboxes = inst.bboxes.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    labels = inst.labels.cpu().numpy()

    keep = scores >= score_thr
    if cat_ids is not None:
        keep &= np.isin(labels, np.asarray(cat_ids))
    return [(np.asarray(b, dtype=np.float32), -1) for b in bboxes[keep]]


def main():
    args = parse_args()

    stem = args.stem or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    boxes_by_frame = None
    if args.boxes == 'npz':
        if not args.boxes_npz:
            raise SystemExit('--boxes npz requires --boxes-npz <EgoForce _3d_keypoints.npz>')
        boxes_by_frame, prov = load_boxes_from_npz(args.boxes_npz)
        # Inherit the 3D stage's frame selection exactly, so fuse_2d_3d.py's width/height/step
        # guard passes by construction rather than by the operator remembering to match flags.
        capture, info = open_video(args.video, 0.0, 1e9, 0.0)
        if info.width != prov['width'] or info.height != prov['height']:
            raise SystemExit(f"video is {info.width}x{info.height} but {args.boxes_npz} was built "
                             f"on {prov['width']}x{prov['height']} - different streams")
        info.step = prov['step']
        info.start_frame = prov['start_frame']
        info.end_frame = prov['end_frame']
        info.sample_fps = prov['sample_fps']
        print(f'[2d {stem}] boxes from {os.path.basename(args.boxes_npz)} '
              f"(model={prov['source_model']}, {sum(len(v) for v in boxes_by_frame.values())} boxes "
              f'over {len(boxes_by_frame)} frames)')
    else:
        capture, info = open_video(args.video, args.start_sec, args.duration_sec, args.sample_fps)

    print(f'[2d {stem}] {info.width}x{info.height}@{info.fps:.2f} '
          f'frames[{info.start_frame},{info.end_frame}) step={info.step} boxes={args.boxes}')

    for path, what in ((args.rtmpose_config, 'config'), (args.rtmpose_checkpoint, 'checkpoint')):
        if not os.path.exists(path):
            raise SystemExit(f'RTMPose {what} not found: {path}\n'
                             f'Run: bash scripts/download_rtmpose_hand5.sh')

    from mmpose.apis import inference_topdown, init_model

    print(f'[2d {stem}] loading RTMPose-m Hand5 on {args.device}...')
    pose_model = init_model(args.rtmpose_config, args.rtmpose_checkpoint, device=args.device)
    detector = build_detector(args) if args.boxes == 'mmdet' else None

    writer = None
    if args.overlay:
        writer = OverlayVideoWriter(os.path.join(args.out, f'{stem}_2d_overlay.mp4'),
                                    info.width, info.height, info.output_fps)

    kp2d_all, conf_all, fidx, isr, score_all, bbox_all, processed = [], [], [], [], [], [], []
    n_frames = n_det = n_low = 0
    t0 = time.time()

    try:
        for frame_index, frame_bgr in iter_frames(capture, info):
            processed.append(frame_index)
            n_frames += 1

            if boxes_by_frame is not None:
                boxes = boxes_by_frame.get(frame_index, [])
            else:
                boxes = detect_boxes(detector, frame_bgr, args.det_score, args.det_cat_ids)

            vis = frame_bgr if writer is None else frame_bgr.copy()

            if boxes:
                padded = np.stack([scale_box(b, args.bbox_pad) for b, _ in boxes])
                # BGR in: the model's data_preprocessor has bgr_to_rgb=True.
                results = inference_topdown(pose_model, frame_bgr, padded, bbox_format='xyxy')
                for (box, is_right), sample in zip(boxes, results):
                    inst = sample.pred_instances
                    kpts = np.asarray(inst.keypoints, dtype=np.float32).reshape(-1, 2)
                    confs = np.asarray(inst.keypoint_scores, dtype=np.float32).reshape(-1)
                    if kpts.shape[0] != NUM_HAND_JOINTS:
                        raise SystemExit(f'RTMPose returned {kpts.shape[0]} keypoints, expected '
                                         f'{NUM_HAND_JOINTS} - wrong config?')
                    mean_conf = float(np.mean(confs))
                    if mean_conf < args.min_kpt_conf:
                        n_low += 1
                        continue
                    kp2d_all.append(kpts)
                    conf_all.append(confs)
                    fidx.append(frame_index)
                    isr.append(int(is_right))
                    score_all.append(mean_conf)
                    bbox_all.append(np.asarray(box, dtype=np.float32))
                    n_det += 1
                    if writer is not None:
                        draw_hand(vis, kpts, 'right' if is_right == 1 else 'left')

            if writer is not None:
                cv2.putText(vis, f'f{frame_index} RTMPose hands={len(boxes)}', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                writer.write(vis)

            if n_frames % 200 == 0:
                print(f'  [2d {stem}] {n_frames} frames, {n_det} hands, '
                      f'{n_frames / max(time.time() - t0, 1e-6):.1f} fps', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.close()

    def stack(rows, shape, dtype=np.float32):
        return np.asarray(rows, dtype=dtype) if rows else np.zeros((0, *shape), dtype=dtype)

    npz_path = os.path.join(args.out, f'{stem}_2d_keypoints.npz')
    np.savez_compressed(
        npz_path,
        # --- the MediaPipe stage-1a schema, verbatim
        kp2d=stack(kp2d_all, (NUM_HAND_JOINTS, 2)),
        frame_idx=np.asarray(fidx, dtype=np.int32),
        is_right=np.asarray(isr, dtype=np.int8),
        # NOTE: the MediaPipe stage stores a HANDEDNESS score here. RTMPose has no handedness head,
        # so this is the mean per-joint keypoint confidence instead - which is what a fusion should
        # weight by anyway. postprocess.py only uses it as a handedness-vote weight.
        score=np.asarray(score_all, dtype=np.float32),
        processed_frames=np.asarray(processed, dtype=np.int32),
        model=np.array('rtmpose-m_hand5'),
        width=np.int32(info.width), height=np.int32(info.height),
        fps=np.float32(info.fps), step=np.int32(info.step),
        # --- additive
        kp2d_conf=stack(conf_all, (NUM_HAND_JOINTS,)),
        bbox=stack(bbox_all, (4,)),
        box_source=np.array(args.boxes),
        keypoint_order=np.array('OpenPose-21'),
        score_semantics=np.array('mean_keypoint_confidence'),
    )

    meta = dict(
        model='RTMPose-m Hand5 (mmpose)', config=args.rtmpose_config,
        checkpoint=os.path.basename(args.rtmpose_checkpoint),
        video=os.path.basename(args.video), width=info.width, height=info.height, fps=info.fps,
        step=info.step, start_frame=info.start_frame, end_frame=info.end_frame,
        frames_processed=n_frames, total_detections=n_det,
        dropped_low_conf=n_low, min_kpt_conf=args.min_kpt_conf,
        box_source=args.boxes, boxes_npz=args.boxes_npz, bbox_pad=args.bbox_pad,
        det_config=args.det_config, det_score=args.det_score, device=args.device,
        keypoint_order='OpenPose-21 (0=wrist;1-4 thumb;5-8 index;9-12 middle;13-16 ring;17-20 pinky)',
        keypoint_order_note='COCO-WholeBody-hand order == OpenPose-21; no remapping applied',
        mean_kpt_conf=(round(float(np.mean(score_all)), 4) if score_all else None),
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(args.out, f'{stem}_2d_meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=2)

    print(f'[2d {stem}] DONE {n_frames} frames, {n_det} hands in {time.time() - t0:.0f}s')
    print(f'[2d {stem}] wrote {npz_path}')


if __name__ == '__main__':
    main()
