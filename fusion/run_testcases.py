#!/usr/bin/env python3
"""Run the test matrix in ``fusion/testcases.yaml`` and collect the comparison.

Each case is a small pipeline of stages, run per clip into its own directory:

    <runs-root>/<clip name>/<case id>/
        <stem>_3d_keypoints.npz      EgoForce
        <stem>_2d_keypoints.npz      RTMPose        (cases that use it)
        <stem>_hand21_keypoints.npz  fused          <- the deliverable, mono-pipeline schema
        <stem>_fuse_stats.json       per-case QC + metrics
        *.mp4                        review overlays, when overlay: true

``--dry-run`` prints every command without executing anything. Use it first: it is the cheapest way
to confirm the paths, the intrinsics and the flags are what you meant before committing GPU time.

Nothing here needs to be re-run to re-tune tracking or filtering. Once a case has written its
``_hand21_keypoints.npz``, the mono-pipeline's own stage 3/4 can re-run over it on CPU:

    python run_clip.py --from-npz <that file> --video <clip> --out <dir> --min-len 6 --foot-filter
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import json
import shlex
import subprocess
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGE_SCRIPTS = {
    'egoforce_3d': os.path.join(HERE, 'run_egoforce_3d.py'),
    'rtmpose_2d': os.path.join(HERE, 'run_rtmpose_2d.py'),
    'fuse': os.path.join(HERE, 'fuse_egoforce_rtmpose.py'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the EgoForce test matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', default=os.path.join(HERE, 'testcases.yaml'))
    parser.add_argument('--runs-root', default=os.path.join(ROOT_DIR, '_DATA', 'runs'))
    parser.add_argument('--case', action='append', default=None,
                        help='only run these case ids (repeatable)')
    parser.add_argument('--clip', action='append', default=None,
                        help='only run these clip names (repeatable)')
    parser.add_argument('--dry-run', action='store_true', help='print commands, run nothing')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='keep going to the next case when one fails')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip a case whose fused npz already exists')

    parser.add_argument('--mono-pipeline', default=None,
                        help='path to hand_labelling_21kp/ (needed by cases using fuse_rigid)')
    parser.add_argument('--rtmpose-config', default=None, help='overrides the stage default')
    parser.add_argument('--rtmpose-checkpoint', default=None)
    parser.add_argument('--det-config', default=None, help='mmdet hand detector, for T2d')
    parser.add_argument('--det-checkpoint', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-compare', action='store_true',
                        help='skip building the comparison table at the end')
    return parser.parse_args()


def flag(name):
    """`hand_conf` -> `--hand-conf`."""
    return '--' + name.replace('_', '-')


def as_flags(params):
    """Turn a YAML param dict into CLI flags. ``True`` becomes a bare switch, ``False``/``None`` is
    dropped, lists are expanded."""
    out = []
    for key, value in (params or {}).items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag(key))
        elif isinstance(value, (list, tuple)):
            out.append(flag(key))
            out.extend(str(v) for v in value)
        else:
            out.append(flag(key))
            out.append(str(value))
    return out


def clip_intrinsic_flags(clip):
    if clip.get('calib'):
        return ['--calib', str(clip['calib'])]
    if clip.get('K'):
        K = clip['K']
        if len(K) != 4:
            raise SystemExit(f"clip {clip['name']}: K must be [fx, fy, cx, cy], got {K}")
        return ['--K', *[str(v) for v in K]]
    raise SystemExit(f"clip {clip['name']}: needs either `calib:` or `K: [fx, fy, cx, cy]`")


def validate_clips(clips, only=None):
    resolved = []
    for clip in clips:
        if only and clip['name'] not in only:
            continue
        if 'EDIT_ME' in clip['name'] or not os.path.exists(str(clip.get('video', ''))):
            raise SystemExit(
                f"clip {clip['name']}: video not found at {clip.get('video')!r}.\n"
                f'Edit the `clips:` block in the config before running - the shipped entries are '
                f'placeholders.')
        if clip.get('calib') and not os.path.exists(str(clip['calib'])):
            raise SystemExit(f"clip {clip['name']}: calibration not found at {clip['calib']!r}")
        resolved.append(clip)
    if not resolved:
        raise SystemExit('no clips selected; check --clip and the config')
    return resolved


def build_commands(args, config, clip, case, out_dir):
    """Return a list of ``(label, argv)`` for one clip x case."""
    stem = os.path.splitext(os.path.basename(clip['video']))[0]
    defaults = config.get('defaults', {}) or {}
    common_window = []
    for key in ('start_sec', 'duration_sec', 'sample_fps'):
        value = clip.get(key, defaults.get(key))
        if value is not None:
            common_window += [flag(key), str(value)]

    overlay = clip.get('overlay', defaults.get('overlay', False))
    commands = []

    for stage in case['stages']:
        params = dict(case.get(stage, {}) or {})

        if stage == 'egoforce_3d':
            argv = [PY, STAGE_SCRIPTS[stage], '--video', clip['video'], '--out', out_dir,
                    '--stem', stem, '--device', args.device]
            argv += clip_intrinsic_flags(clip) + common_window + as_flags(params)
            if overlay:
                argv.append('--overlay')
            commands.append(('egoforce_3d', argv))

        elif stage == 'rtmpose_2d':
            argv = [PY, STAGE_SCRIPTS[stage], '--video', clip['video'], '--out', out_dir,
                    '--stem', stem, '--device', args.device]
            if params.get('boxes', 'npz') == 'npz':
                argv += ['--boxes', 'npz',
                         '--boxes-npz', os.path.join(out_dir, f'{stem}_3d_keypoints.npz')]
                params.pop('boxes', None)
            else:
                if not (args.det_config and args.det_checkpoint):
                    raise SystemExit(
                        f"case {case['id']} uses `boxes: mmdet` but --det-config/--det-checkpoint "
                        f'were not given')
                argv += ['--boxes', 'mmdet',
                         '--det-config', args.det_config,
                         '--det-checkpoint', args.det_checkpoint]
                params.pop('boxes', None)
                argv += common_window
            if args.rtmpose_config:
                argv += ['--rtmpose-config', args.rtmpose_config]
            if args.rtmpose_checkpoint:
                argv += ['--rtmpose-checkpoint', args.rtmpose_checkpoint]
            argv += as_flags(params)
            if overlay:
                argv.append('--overlay')
            commands.append(('rtmpose_2d', argv))

        elif stage == 'fuse':
            argv = [PY, STAGE_SCRIPTS[stage],
                    '--egoforce', os.path.join(out_dir, f'{stem}_3d_keypoints.npz'),
                    '--out', out_dir, '--stem', stem]
            if params.get('mode', 'articulated') == 'articulated':
                argv += ['--rtmpose', os.path.join(out_dir, f'{stem}_2d_keypoints.npz')]
            argv += as_flags(params)
            if overlay:
                argv += ['--overlay', '--video', clip['video']]
            commands.append(('fuse', argv))

        elif stage == 'fuse_rigid':
            if not args.mono_pipeline:
                raise SystemExit(
                    f"case {case['id']} needs the existing rigid fusion; pass "
                    f'--mono-pipeline /path/to/hand_labelling_21kp')
            script = os.path.join(args.mono_pipeline, 'fuse_2d_3d.py')
            if not os.path.exists(script):
                raise SystemExit(f'not found: {script}')
            argv = [PY, script,
                    '--mp2d', os.path.join(out_dir, f'{stem}_2d_keypoints.npz'),
                    '--wilor', os.path.join(out_dir, f'{stem}_3d_keypoints.npz'),
                    '--out', out_dir, '--stem', stem]
            argv += as_flags(params)
            commands.append(('fuse_rigid', argv))

        else:
            raise SystemExit(f"case {case['id']}: unknown stage {stage!r}")

    return commands, stem


def main():
    args = parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    cases = config.get('cases') or []
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c['id'] in wanted]
        missing = wanted - {c['id'] for c in cases}
        if missing:
            raise SystemExit(f'unknown case id(s): {sorted(missing)}')
    if not cases:
        raise SystemExit('no cases selected')

    clips = validate_clips(config.get('clips') or [], set(args.clip) if args.clip else None)

    plan, skipped = [], []
    for clip in clips:
        for case in cases:
            if case.get('requires_mono_pipeline') and not args.mono_pipeline:
                skipped.append((clip['name'], case['id'], 'needs --mono-pipeline'))
                continue
            if case.get('requires_detector') and not (args.det_config and args.det_checkpoint):
                skipped.append((clip['name'], case['id'], 'needs --det-config/--det-checkpoint'))
                continue
            out_dir = os.path.join(args.runs_root, clip['name'], case['id'])
            commands, stem = build_commands(args, config, clip, case, out_dir)
            fused = os.path.join(out_dir, f'{stem}_hand21_keypoints.npz')
            if args.skip_existing and os.path.exists(fused):
                skipped.append((clip['name'], case['id'], 'fused npz exists (--skip-existing)'))
                continue
            plan.append(dict(clip=clip['name'], case=case['id'], out_dir=out_dir, stem=stem,
                             commands=commands, question=case.get('question', '').strip(),
                             fused=fused))

    print(f'{len(plan)} clip x case run(s) planned, {len(skipped)} skipped')
    for clip_name, case_id, why in skipped:
        print(f'  SKIP {clip_name} / {case_id}: {why}')
    print()

    if args.dry_run:
        for item in plan:
            print(f"=== {item['clip']} / {item['case']}")
            if item['question']:
                print(f"    question: {item['question']}")
            print(f"    out: {item['out_dir']}")
            for label, argv in item['commands']:
                print(f'    [{label}] {shlex.join(argv)}')
            print()
        print('dry run: nothing executed')
        return

    results = []
    for item in plan:
        os.makedirs(item['out_dir'], exist_ok=True)
        print(f"=== {item['clip']} / {item['case']} -> {item['out_dir']}", flush=True)
        started = time.time()
        status, failed_stage, message = 'ok', None, None
        for label, argv in item['commands']:
            print(f'  [{label}] {shlex.join(argv)}', flush=True)
            proc = subprocess.run(argv, cwd=HERE)
            if proc.returncode != 0:
                status, failed_stage = 'failed', label
                message = f'{label} exited {proc.returncode}'
                print(f'  FAILED: {message}')
                break
        elapsed = round(time.time() - started, 1)
        results.append(dict(clip=item['clip'], case=item['case'], out_dir=item['out_dir'],
                            stem=item['stem'], status=status, failed_stage=failed_stage,
                            message=message, elapsed_sec=elapsed,
                            fused=item['fused'] if status == 'ok' else None))
        if status == 'failed' and not args.continue_on_error:
            break

    os.makedirs(args.runs_root, exist_ok=True)
    manifest = os.path.join(args.runs_root, 'runs_manifest.json')
    with open(manifest, 'w') as handle:
        json.dump(dict(config=os.path.abspath(args.config), runs=results, skipped=skipped),
                  handle, indent=2)
    print(f'\nwrote {manifest}')

    ok = [r for r in results if r['status'] == 'ok']
    print(f'{len(ok)}/{len(results)} run(s) completed')
    for r in results:
        if r['status'] != 'ok':
            print(f"  FAILED {r['clip']} / {r['case']}: {r['message']}")

    if ok and not args.no_compare:
        argv = [PY, os.path.join(HERE, 'compare_runs.py'),
                '--manifest', manifest, '--out', args.runs_root]
        print(f'\n[compare] {shlex.join(argv)}', flush=True)
        subprocess.run(argv, cwd=HERE)


if __name__ == '__main__':
    main()
