#!/usr/bin/env python3
"""Build the cross-case comparison table from a set of fused npz files.

Every metric is recomputed **from the fused npz** rather than read out of the per-case stats sidecar.
That matters: test case ``T2a`` is produced by the mono-pipeline's own ``fuse_2d_3d.py``, whose
sidecar has a different shape and does not carry coverage or track metrics at all. Recomputing here
is the only way the rigid and articulated fusions end up on the same axes.

The columns are the ones the plan asks to compare on - missed-hand duration, foot false positives,
fingertip error, identity switches, recovery delay - with two honest substitutions:

* ``tip_disagree_px`` stands in for fingertip error. It is 2D disagreement between the written 3D's
  projection and RTMPose's independent observation, so it only exists for cases that ran RTMPose, and
  it ranks variants rather than measuring accuracy. See ``metrics.fingertip_agreement``.
* ``foot_lowframe`` is a review shortlist, not a foot count. ``foot_degenerate`` IS a real defect
  count. See ``metrics.foot_candidates``.

Usage
-----
    python fusion/compare_runs.py --manifest _DATA/runs/runs_manifest.json --out _DATA/runs
    python fusion/compare_runs.py --fused a/x_hand21_keypoints.npz b/y_hand21_keypoints.npz --out /tmp
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import csv
import glob
import json

import numpy as np

from fusion.metrics import (bone_consistency, coverage_metrics, depth_qc, fingertip_agreement,
                            foot_candidates, reprojection_qc, track_metrics)

# Column order for the CSV and the markdown table. Keep the identity columns first so a wide table
# is still readable after truncation.
COLUMNS = [
    'clip', 'case', 'hands', 'src_fused', 'src_ego_only', 'src_refit_rejected', 'src_lifted_2d',
    'cov_left', 'cov_right', 'gap_max_s_left', 'gap_max_s_right',
    'recovery_med_s_left', 'recovery_med_s_right',
    'tracks_kept', 'tracks_dropped_short', 'dets_lost_short', 'label_flips',
    'frag_left', 'frag_right',
    'reproj_med_px', 'depth_implausible_frac', 'bone_cv_med',
    'tip_disagree_px', 'joint_disagree_px',
    'foot_lowframe', 'foot_degenerate',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare fused runs on the same axes.')
    parser.add_argument('--manifest', default=None,
                        help='runs_manifest.json written by run_testcases.py')
    parser.add_argument('--fused', nargs='+', default=None,
                        help='explicit list of _hand21_keypoints.npz files')
    parser.add_argument('--runs-root', default=None,
                        help='glob <root>/*/*/*_hand21_keypoints.npz instead of a manifest')
    parser.add_argument('--out', required=True, help='directory for the comparison outputs')
    parser.add_argument('--track-min-len', type=int, default=10,
                        help="mirror the mono-pipeline's --min-len so the table shows what it drops")
    parser.add_argument('--min-kpt-conf', type=float, default=0.3)
    return parser.parse_args()


def discover_runs(args):
    """Return a list of ``(clip, case, fused_path)``."""
    runs = []
    if args.manifest:
        with open(args.manifest) as handle:
            manifest = json.load(handle)
        for entry in manifest.get('runs', []):
            if entry.get('status') == 'ok' and entry.get('fused'):
                runs.append((entry['clip'], entry['case'], entry['fused']))
    if args.runs_root:
        for path in sorted(glob.glob(os.path.join(args.runs_root, '*', '*',
                                                  '*_hand21_keypoints.npz'))):
            case = os.path.basename(os.path.dirname(path))
            clip = os.path.basename(os.path.dirname(os.path.dirname(path)))
            runs.append((clip, case, path))
    if args.fused:
        for path in args.fused:
            case = os.path.basename(os.path.dirname(path))
            clip = os.path.basename(os.path.dirname(os.path.dirname(path)))
            runs.append((clip, case, path))

    seen, unique = set(), []
    for clip, case, path in runs:
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((clip, case, path))
    if not unique:
        raise SystemExit('no fused npz files found; pass --manifest, --runs-root or --fused')
    return unique


def processed_frames_for(fused_path, fused):
    """Prefer the producer's own processed-frame list; fall back to the detected range.

    The fallback understates gaps, because a frame nobody processed cannot be a miss. It is used only
    when the sibling 3D npz is absent - the mono-pipeline's own fused npz does not carry the list.
    """
    sibling = glob.glob(os.path.join(os.path.dirname(fused_path), '*_3d_keypoints.npz'))
    if sibling:
        data = np.load(sibling[0], allow_pickle=False)
        if 'processed_frames' in data and len(data['processed_frames']):
            return np.asarray(data['processed_frames']), 'producer'

    frames = np.asarray(fused['frame_idx'], dtype=np.int64)
    if frames.size == 0:
        return frames, 'empty'
    step = max(1, int(fused['step']))
    return np.arange(frames.min(), frames.max() + 1, step), 'inferred_from_range'


def row_for(clip, case, fused_path, args):
    fused = dict(np.load(fused_path, allow_pickle=False))
    K = np.asarray(fused['K'], dtype=np.float64)
    width, height = int(fused['width']), int(fused['height'])
    step, fps = int(fused['step']), float(fused['fps'])

    source = np.asarray(fused['source']).astype(str)
    counts = {s: int((source == s).sum()) for s in set(source.tolist())}

    hand_label = np.where(np.asarray(fused['is_right_wilor']) >= 0,
                          np.asarray(fused['is_right_wilor']),
                          np.asarray(fused['is_right_mp']))

    processed, processed_origin = processed_frames_for(fused_path, fused)
    coverage = coverage_metrics(fused['frame_idx'], hand_label, processed, fps, step)
    tracks = track_metrics(fused['frame_idx'], fused['kp2d'], hand_label, width, fps, step,
                           min_len=args.track_min_len)
    reproj = reprojection_qc(fused['kp3d_cam'], fused['kp2d'], K, source)
    _, depth = depth_qc(fused['kp3d_cam'])
    measured = np.asarray(fused['depth_measured'], dtype=bool)
    bones = bone_consistency(fused['kp3d_cam'][measured]) if measured.any() else {}
    feet = foot_candidates(fused['kp2d'], fused['kp3d_cam'], K, height)

    agreement = {}
    if 'kp2d_rtmpose' in fused:
        obs_mask = np.isfinite(np.asarray(fused['kp2d_rtmpose'])).all(axis=(1, 2))
        if obs_mask.any():
            agreement = fingertip_agreement(
                np.asarray(fused['kp3d_cam'])[obs_mask], K,
                np.asarray(fused['kp2d_rtmpose'])[obs_mask],
                np.asarray(fused['kp2d_rtmpose_conf'])[obs_mask]
                if 'kp2d_rtmpose_conf' in fused else None,
                args.min_kpt_conf)

    def dig(mapping, *keys, default=None):
        for key in keys:
            if not isinstance(mapping, dict) or key not in mapping:
                return default
            mapping = mapping[key]
        return mapping

    row = {
        'clip': clip, 'case': case, 'hands': int(len(source)),
        'src_fused': counts.get('fused', 0),
        'src_ego_only': counts.get('wilor', 0),
        'src_refit_rejected': counts.get('wilor_pnpfail', 0),
        'src_lifted_2d': counts.get('lifted_2d', 0),
        'cov_left': dig(coverage, 'left', 'coverage_within_span'),
        'cov_right': dig(coverage, 'right', 'coverage_within_span'),
        'gap_max_s_left': dig(coverage, 'left', 'longest_gap_s'),
        'gap_max_s_right': dig(coverage, 'right', 'longest_gap_s'),
        'recovery_med_s_left': dig(coverage, 'left', 'median_recovery_s'),
        'recovery_med_s_right': dig(coverage, 'right', 'median_recovery_s'),
        'tracks_kept': tracks.get('n_tracks_kept'),
        'tracks_dropped_short': tracks.get('n_tracks_dropped_short'),
        'dets_lost_short': tracks.get('detections_lost_to_short_tracks'),
        'label_flips': tracks.get('label_flips'),
        'frag_left': tracks.get('fragments_left'),
        'frag_right': tracks.get('fragments_right'),
        'reproj_med_px': dig(reproj, 'ALL', 'median_px'),
        'depth_implausible_frac': depth.get('implausible_frac'),
        'bone_cv_med': bones.get('bone_cv_median'),
        'tip_disagree_px': dig(agreement, 'fingertips', 'median_px'),
        'joint_disagree_px': dig(agreement, 'all_joints', 'median_px'),
        'foot_lowframe': feet.get('low_in_frame'),
        'foot_degenerate': feet.get('degenerate_span'),
    }
    detail = dict(row, coverage=coverage, tracks=tracks, reprojection=reproj, depth=depth,
                  bone_consistency=bones, foot_candidates_proxy=feet,
                  fingertip_agreement_proxy=agreement,
                  processed_frames_origin=processed_origin,
                  fused_npz=os.path.abspath(fused_path))
    return row, detail


def markdown_table(rows):
    header = '| ' + ' | '.join(COLUMNS) + ' |'
    rule = '| ' + ' | '.join('---' for _ in COLUMNS) + ' |'
    lines = [header, rule]
    for row in rows:
        cells = []
        for col in COLUMNS:
            value = row.get(col)
            cells.append('' if value is None else str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    runs = discover_runs(args)
    os.makedirs(args.out, exist_ok=True)

    rows, details = [], []
    for clip, case, path in runs:
        print(f'[compare] {clip} / {case}: {path}')
        try:
            row, detail = row_for(clip, case, path, args)
        except Exception as exc:
            print(f'  FAILED to read: {type(exc).__name__}: {exc}')
            continue
        rows.append(row)
        details.append(detail)

    if not rows:
        raise SystemExit('nothing comparable was read')

    rows.sort(key=lambda r: (r['clip'], r['case']))

    csv_path = os.path.join(args.out, 'comparison.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out, 'comparison.json')
    with open(json_path, 'w') as handle:
        json.dump(details, handle, indent=2, default=str)

    md_path = os.path.join(args.out, 'comparison.md')
    with open(md_path, 'w') as handle:
        handle.write('# EgoForce test-case comparison\n\n')
        handle.write('`tip_disagree_px` and `foot_lowframe` are PROXIES, not ground-truth error or '
                     'a foot count - see `fusion/metrics.py`.\n\n')
        handle.write(markdown_table(rows))
        handle.write('\n')

    print()
    print(markdown_table(rows))
    print()
    for path in (csv_path, json_path, md_path):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
