#!/usr/bin/env python3
"""The three metrics a delivered run is judged on.

Everything else in ``fusion/metrics.py`` is diagnostic. These three are the headline numbers, chosen
to map one-to-one onto the three defects the pipeline audit actually found, and constrained by the
fact that **there is no ground truth** for this footage:

    M1  Hand coverage & recovery   -> "hands missing from the output"      EXACT
    M2  Geometric validity         -> "wrong finger / wrist geometry"      EXACT
    M3  Fingertip agreement        -> finger accuracy                      PROXY

M1 and M2 are exact properties of the output: they need no annotation and no second model, and a
regression in either is a real regression. **M3 is a proxy** - it measures where the written 3D and
an independent 2D model disagree, which ranks variants against each other but is not accuracy. Two
models that agree may both be wrong. It is labelled ``is_proxy`` everywhere it appears.

Fairness note for M3: both run types are scored against the **same** RTMPose 2D npz. Type 1 never
consumed RTMPose, so scoring it against the observation produced during the type 2 run is what makes
the comparison apples-to-apples rather than measuring type 2 against its own input.

Usage
-----
    python fusion/evaluate_run.py \
        --run type1=~/egoforce_runs/episode_047/type1_egoforce_only/out \
        --run type2=~/egoforce_runs/episode_047/type2_egoforce_rtmpose/out \
        --rtmpose ~/egoforce_runs/episode_047/type2_egoforce_rtmpose/out/episode_047_2d_keypoints.npz \
        --out ~/egoforce_runs/episode_047/evaluation

Runs on numpy alone - no GPU, no model weights.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import glob
import json

import numpy as np

from fusion.metrics import (bone_consistency, coverage_metrics, depth_qc, fingertip_agreement,
                            foot_candidates, reprojection_qc)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Score one or more runs on the three delivery metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                        help='a run to score; repeatable. DIR holds the *_hand21_keypoints.npz')
    parser.add_argument('--rtmpose', default=None,
                        help='RTMPose *_2d_keypoints.npz used as the common independent 2D '
                             'reference for M3. Without it M3 is skipped.')
    parser.add_argument('--out', required=True, help='directory for report.json / report.md')
    parser.add_argument('--min-kpt-conf', type=float, default=0.3,
                        help='RTMPose landmarks below this are ignored in M3')
    parser.add_argument('--max-hand-m', type=float, default=0.30,
                        help='a real hand is no wider than this; used by the degenerate-span check')
    return parser.parse_args()


def find_fused(run_dir):
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(run_dir), '*_hand21_keypoints.npz')))
    if not hits:
        raise SystemExit(f'no *_hand21_keypoints.npz in {run_dir}')
    if len(hits) > 1:
        raise SystemExit(f'{run_dir} has {len(hits)} fused npz files; expected exactly one')
    return hits[0]


def processed_frames_for(fused_path, fused):
    """Prefer the producer's own processed-frame list; fall back to the detected range.

    The fallback understates gaps - a frame nobody processed cannot be a miss - so it is used only
    when the sibling 3D npz is absent, and the report records which was used.
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


def score(label, run_dir, rtm, args):
    path = find_fused(run_dir)
    fused = dict(np.load(path, allow_pickle=False))
    K = np.asarray(fused['K'], dtype=np.float64)
    height = int(fused['height'])
    fps, step = float(fused['fps']), int(fused['step'])

    source = np.asarray(fused['source']).astype(str)
    hand_label = np.where(np.asarray(fused['is_right_wilor']) >= 0,
                          np.asarray(fused['is_right_wilor']),
                          np.asarray(fused['is_right_mp']))
    processed, origin = processed_frames_for(path, fused)

    # ---------------------------------------------------------------- M1: coverage & recovery
    cov = coverage_metrics(fused['frame_idx'], hand_label, processed, fps, step)
    m1 = {
        'is_proxy': False,
        'measures': 'how much of the clip each hand is actually present in the output',
        'processed_frames': cov.get('processed_frames'),
        'processed_duration_s': cov.get('processed_duration_s'),
        'processed_frames_origin': origin,
    }
    for side in ('left', 'right'):
        s = cov.get(side, {})
        m1[side] = {
            'coverage_within_span': s.get('coverage_within_span'),
            'frames_detected': s.get('frames_detected'),
            'longest_gap_s': s.get('longest_gap_s'),
            'total_gap_s': s.get('total_gap_s'),
            'median_recovery_s': s.get('median_recovery_s'),
        }

    # ---------------------------------------------------------------- M2: geometric validity
    reproj = reprojection_qc(fused['kp3d_cam'], fused['kp2d'], K, source)
    _, depth = depth_qc(fused['kp3d_cam'])
    measured = np.asarray(fused['depth_measured'], dtype=bool)
    bones = bone_consistency(fused['kp3d_cam'][measured]) if measured.any() else {}
    feet = foot_candidates(fused['kp2d'], fused['kp3d_cam'], K, height, max_hand_m=args.max_hand_m)
    m2 = {
        'is_proxy': False,
        'measures': 'is the delivered 3D internally consistent and physically possible',
        'reproj_median_px': (reproj.get('ALL') or {}).get('median_px'),
        'reproj_p95_px': (reproj.get('ALL') or {}).get('p95_px'),
        'depth_implausible_frac': depth.get('implausible_frac'),
        'bone_cv_median': bones.get('bone_cv_median'),
        'bone_cv_max': bones.get('bone_cv_max'),
        'degenerate_span': feet.get('degenerate_span'),
        'wrist_Z_median_m': (depth.get('wrist_Z_m') or {}).get('median'),
    }

    # ---------------------------------------------------------------- M3: fingertip agreement
    m3 = {'is_proxy': True, 'skipped': 'no --rtmpose reference supplied'}
    if rtm is not None:
        # Score only the rows this run and the reference share a frame on, so the two run types are
        # compared over the same evidence rather than over whatever each happened to detect.
        ref_by_frame = {}
        for i, f in enumerate(np.asarray(rtm['frame_idx'])):
            ref_by_frame.setdefault(int(f), []).append(i)

        rows, refs = [], []
        for i, f in enumerate(np.asarray(fused['frame_idx'])):
            cands = ref_by_frame.get(int(f), [])
            if not cands:
                continue
            wrist = np.asarray(fused['kp2d'])[i, 0]
            j = min(cands, key=lambda c: float(np.hypot(*(rtm['kp2d'][c][0] - wrist))))
            if float(np.hypot(*(rtm['kp2d'][j][0] - wrist))) <= 120.0:
                rows.append(i)
                refs.append(j)

        if rows:
            conf = (np.asarray(rtm['kp2d_conf'])[refs] if 'kp2d_conf' in rtm else None)
            agree = fingertip_agreement(np.asarray(fused['kp3d_cam'])[rows], K,
                                        np.asarray(rtm['kp2d'])[refs], conf, args.min_kpt_conf)
            m3 = {
                'is_proxy': True,
                'measures': 'px disagreement between this run\'s 3D projected through K and an '
                            'independent RTMPose 2D observation - ranks variants, is NOT accuracy',
                'rows_compared': len(rows),
                'fingertips_median_px': (agree.get('fingertips') or {}).get('median_px'),
                'fingertips_p95_px': (agree.get('fingertips') or {}).get('p95_px'),
                'all_joints_median_px': (agree.get('all_joints') or {}).get('median_px'),
                'non_fingertips_median_px': (agree.get('non_fingertips') or {}).get('median_px'),
            }
        else:
            m3 = {'is_proxy': True, 'skipped': 'no frames shared with the RTMPose reference'}

    return {
        'label': label,
        'fused_npz': os.path.abspath(path),
        'detections': int(len(source)),
        'source_mix': {s: int((source == s).sum()) for s in sorted(set(source.tolist()))},
        'M1_coverage_and_recovery': m1,
        'M2_geometric_validity': m2,
        'M3_fingertip_agreement_proxy': m3,
    }


def fmt(v, nd=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def markdown(results):
    lines = ['# Run evaluation — three metrics', '']
    lines += ['**M1** and **M2** are exact properties of the output. **M3 is a proxy**: it measures '
              'disagreement with an independent 2D model, which ranks variants but is not accuracy — '
              'two models that agree may both be wrong.', '']

    labels = [r['label'] for r in results]
    head = '| Metric | ' + ' | '.join(labels) + ' |'
    rule = '| --- | ' + ' | '.join('---' for _ in labels) + ' |'

    def row(name, get, nd=3):
        return f'| {name} | ' + ' | '.join(fmt(get(r), nd) for r in results) + ' |'

    lines += ['## M1 — Hand coverage & recovery  *(exact)*', '',
              'Targets the "hands missing from the output" defect.', '', head, rule]
    for side in ('left', 'right'):
        lines += [
            row(f'{side} coverage within span', lambda r, s=side: r['M1_coverage_and_recovery'][s]['coverage_within_span'], 4),
            row(f'{side} longest gap (s)', lambda r, s=side: r['M1_coverage_and_recovery'][s]['longest_gap_s']),
            row(f'{side} median recovery (s)', lambda r, s=side: r['M1_coverage_and_recovery'][s]['median_recovery_s']),
        ]
    lines += [row('detections', lambda r: r['detections'])]

    lines += ['', '## M2 — Geometric validity  *(exact)*', '',
              'Targets the "wrong finger / wrist geometry" defect. `reproj_median_px` must be < 1 — '
              'above that the 3D and the intrinsics describe different cameras and nothing else here '
              'means anything. `bone_cv_median` catches a hand whose bone lengths drift frame to '
              'frame, which reads as a pulsing hand in the overlay.', '', head, rule]
    for name, key, nd in (('reproj median (px)', 'reproj_median_px', 4),
                          ('reproj p95 (px)', 'reproj_p95_px', 4),
                          ('depth implausible frac', 'depth_implausible_frac', 5),
                          ('bone length CV (median)', 'bone_cv_median', 5),
                          ('bone length CV (max)', 'bone_cv_max', 5),
                          ('degenerate span count', 'degenerate_span', 0),
                          ('wrist Z median (m)', 'wrist_Z_median_m', 3)):
        lines += [row(name, lambda r, k=key: r['M2_geometric_validity'].get(k), nd)]

    lines += ['', '## M3 — Fingertip agreement  *(PROXY, not accuracy)*', '',
              'Both runs are scored against the **same** RTMPose 2D observation, so the comparison is '
              'apples-to-apples. Lower is better only in the sense of "closer to the independent '
              'observation".', '', head, rule]
    for name, key in (('rows compared', 'rows_compared'),
                      ('fingertips median (px)', 'fingertips_median_px'),
                      ('fingertips p95 (px)', 'fingertips_p95_px'),
                      ('all joints median (px)', 'all_joints_median_px'),
                      ('non-fingertips median (px)', 'non_fingertips_median_px')):
        lines += [row(name, lambda r, k=key: r['M3_fingertip_agreement_proxy'].get(k))]

    lines += ['', '## Source mix', '', head, rule]
    for src in sorted({s for r in results for s in r['source_mix']}):
        lines += [row(f'`{src}`', lambda r, s=src: r['source_mix'].get(s, 0))]

    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    runs = []
    for spec in args.run:
        if '=' not in spec:
            raise SystemExit(f'--run expects LABEL=DIR, got {spec!r}')
        label, _, path = spec.partition('=')
        runs.append((label, path))

    rtm = None
    if args.rtmpose:
        rtm = dict(np.load(os.path.expanduser(args.rtmpose), allow_pickle=False))
        print(f'M3 reference: {args.rtmpose}  ({len(rtm["frame_idx"])} rows)')
    else:
        print('M3 skipped: no --rtmpose reference supplied')

    results = [score(label, path, rtm, args) for label, path in runs]

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'report.json'), 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    md = markdown(results)
    with open(os.path.join(out, 'report.md'), 'w') as fh:
        fh.write(md)

    print()
    print(md)
    print(f"wrote {os.path.join(out, 'report.json')}")
    print(f"wrote {os.path.join(out, 'report.md')}")


if __name__ == '__main__':
    main()
