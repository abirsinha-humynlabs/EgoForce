"""Read pinhole intrinsics out of a mono Standard Package ``calibration.json``.

Deliberately the same schema handling and the same warnings as the mono-pipeline's
``run_clip.read_K``, so a normalized clip directory can be fed to these stages directly. Making the
operator transcribe ``fx fy cx cy`` by hand is a real source of error: the whole reason that repo
warns when ``cx`` is far from ``W/2`` is that a left-eye video with right-eye intrinsics silently
biases every 3D point.
"""

import json


def read_K(calib_path, prefer='rectified', eye='left'):
    """Return ``([fx, fy, cx, cy], which_block)``.

    Accepts the Standard Package v2 schema
    ``{"rectified": {"camera": {"fx":..,"fy":..,"cx":..,"cy":..}}, "raw": {...}}``, the stereo
    per-eye schema ``{"rectified": {"left": {...}}}``, and a flat
    ``{"fx":..,"fy":..,"cx":..,"cy":..}``.

    The RECTIFIED block is the right one: these stages consume an already-rectified pinhole stream.
    Falling back to ``raw`` is allowed but warned about loudly - using raw intrinsics on a rectified
    video is a real, if small, error rather than a formality.
    """
    with open(calib_path) as handle:
        calib = json.load(handle)

    if all(k in calib for k in ('fx', 'fy', 'cx', 'cy')):
        return [float(calib['fx']), float(calib['fy']),
                float(calib['cx']), float(calib['cy'])], 'flat'

    block, used = calib.get(prefer), prefer
    if block is None:
        block, used = calib.get('raw'), 'raw'
        print(f"[warn] calibration has no '{prefer}' block; falling back to 'raw' intrinsics. "
              f'If the video IS rectified this introduces a small principal-point/focal error.')
    if block is None:
        raise SystemExit(f"calibration.json has neither '{prefer}' nor 'raw'")

    entry = block.get(eye) or block.get('camera') or block.get('left')
    if entry is None:
        raise SystemExit(f"calibration '{used}' block has no '{eye}'/'camera' entry")

    if ('distortion' in entry and used == 'raw'
            and any(abs(float(d)) > 1e-6 for d in entry['distortion'])):
        print(f"[warn] using RAW intrinsics that carry non-zero distortion {entry['distortion']}: "
              f'these stages expect an undistorted/rectified stream')

    return [float(entry['fx']), float(entry['fy']),
            float(entry['cx']), float(entry['cy'])], used
