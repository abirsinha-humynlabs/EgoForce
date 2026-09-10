"""Comparison metrics for the EgoForce experiments.

The point of these is the comparison the plan actually calls for - **missed-hand duration, foot false
positives, fingertip error, identity switches and recovery delay on the same difficult clips** - not
which overlay looks smoothest.

Read this before trusting a number:

* ``depth_qc`` and ``reprojection_qc`` are re-implementations of the checks already in the
  mono-pipeline's ``fuse_2d_3d.py``, kept here so this repo's stages can emit the same sidecar stats
  without importing that repo. They are exact QC, not proxies.
* ``coverage_metrics`` and ``track_metrics`` are exact measurements of what the pipeline output
  contains: coverage, gap durations, recovery delays, track fragmentation, handedness flips.
* **``fingertip_agreement`` is a PROXY, not fingertip error.** With no ground truth, all it measures
  is where two models disagree. Disagreement bounds error from below in the sense that if the two
  agree they are either both right or both wrong the same way - it cannot tell you which. Every
  function that returns a proxy says so in its name or docstring, and ``foot_candidates`` is the
  weakest of them.
* Real fingertip error needs annotation. ``fingertip_error`` takes a ground-truth array and is the
  function to use once difficult clips are labelled; until then it has nothing to consume.
"""

import collections

import numpy as np

from fusion.topology import TIPS, WRIST, bone_lengths, project_pinhole

MEASURED_SOURCES = ('fused', 'wilor', 'wilor_pnpfail')


# ------------------------------------------------------------------ QC (exact)


def depth_qc(kp3d_cam, zmin=0.05, zmax=3.0):
    """Flag physically impossible depths over ALL 21 joints, not just the wrist.

    Mirrors ``fuse_2d_3d.depth_qc``. Checking only the wrist misses the real failures: a wrist at a
    plausible depth with a fingertip at 0.2 mm is a 13 cm-deep "hand". Nothing is dropped here - rows
    are flagged and counted, so a systematic depth failure (wrong K, changed convention) shows up as
    a large implausible fraction instead of hiding inside a healthy-looking median.
    """
    kp3d_cam = np.asarray(kp3d_cam)
    if kp3d_cam.shape[0] == 0:
        return np.zeros(0, bool), {}
    Z = kp3d_cam[..., 2].astype(np.float64)
    ok = np.isfinite(Z).all(1) & (Z >= zmin).all(1) & (Z <= zmax).all(1)
    w = Z[:, WRIST]
    stats = dict(
        zmin_m=zmin, zmax_m=zmax, checked='all 21 joints',
        implausible_rows=int((~ok).sum()), implausible_frac=round(float((~ok).mean()), 5),
        implausible_keypoints=int(((Z < zmin) | (Z > zmax)).sum()),
        wrist_Z_m=dict(p1=round(float(np.percentile(w, 1)), 4),
                       median=round(float(np.median(w)), 4),
                       p99=round(float(np.percentile(w, 99)), 4),
                       min=round(float(w.min()), 6), max=round(float(w.max()), 4)),
        any_joint_Z_m=dict(min=round(float(Z.min()), 6), max=round(float(Z.max()), 4)),
    )
    return ok, stats


def reprojection_qc(kp3d_cam, kp2d, K, source=None):
    """Project the 3D through the stored K and compare to the stored 2D, per source.

    Mirrors ``fuse_2d_3d.reprojection_qc``. This is the single check that proves the 3D and the
    intrinsics describe the same camera - it catches a principal-point or focal-length mismatch
    immediately.
    """
    kp3d_cam = np.asarray(kp3d_cam)
    if kp3d_cam.shape[0] == 0:
        return {}
    proj = project_pinhole(kp3d_cam, np.asarray(K, dtype=np.float64))
    err = np.linalg.norm(proj - np.asarray(kp2d, dtype=np.float64), axis=-1)

    def summarise(e):
        return dict(n=int(e.shape[0]), median_px=round(float(np.median(e)), 3),
                    p95_px=round(float(np.percentile(e, 95)), 3))

    out = {}
    if source is not None:
        source = np.asarray(source)
        for s in sorted(set(source.tolist())):
            out[str(s)] = summarise(err[source == s])
    out['ALL'] = summarise(err)
    return out


def bone_consistency(kp3d_cam):
    """Per-bone length spread across detections. An articulated refit that "breathes" shows up here.

    A MANO-parameterised hand cannot breathe within one frame, but nothing stops the shape parameters
    drifting frame to frame, which reads as a pulsing hand in the overlay. Reported as the
    coefficient of variation per bone, aggregated.
    """
    kp3d_cam = np.asarray(kp3d_cam)
    if kp3d_cam.shape[0] < 2:
        return {}
    lengths = bone_lengths(kp3d_cam)                     # (N, 20)
    mean = lengths.mean(axis=0)
    cv = lengths.std(axis=0) / np.clip(mean, 1e-9, None)
    return dict(n=int(lengths.shape[0]),
                bone_len_mean_cm=round(float(mean.mean() * 100), 3),
                bone_cv_median=round(float(np.median(cv)), 5),
                bone_cv_max=round(float(cv.max()), 5),
                worst_bone_index=int(np.argmax(cv)))


# ------------------------------------------------------------------ coverage / recovery (exact)


def coverage_metrics(frame_idx, hand_label, processed_frames, fps, step=1):
    """Detection coverage, gap durations and recovery delays, per hand.

    ``hand_label``: 0 = left, 1 = right, anything else is ignored (e.g. -1 "unknown", 2 "other").

    A "gap" is a run of PROCESSED frames between the first and last detection of that hand in which
    the hand was not detected. Frames before the first and after the last detection are excluded on
    purpose: the hand may genuinely be out of frame, and counting that as a miss would reward a model
    that hallucinates hands. The gap durations are exactly the recovery delays.
    """
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    hand_label = np.asarray(hand_label, dtype=np.int64)
    processed = np.unique(np.asarray(processed_frames, dtype=np.int64))
    fps = float(fps) if fps and fps > 0 else 30.0

    out = dict(processed_frames=int(processed.size),
               processed_duration_s=round(float(processed.size * step / fps), 3))
    for label, name in ((0, 'left'), (1, 'right')):
        seen = np.unique(frame_idx[hand_label == label])
        entry = dict(detections=int((hand_label == label).sum()), frames_detected=int(seen.size))
        if seen.size == 0:
            entry.update(coverage=0.0, n_gaps=0, longest_gap_s=None, total_gap_s=None,
                         median_recovery_s=None,
                         note='never detected; gap stats undefined without a first detection')
            out[name] = entry
            continue

        span = processed[(processed >= seen[0]) & (processed <= seen[-1])]
        missing = np.setdiff1d(span, seen, assume_unique=False)
        entry['coverage_within_span'] = round(float(seen.size / max(span.size, 1)), 4)
        entry['span_s'] = round(float(span.size * step / fps), 3)

        gaps = []
        if missing.size:
            # Consecutive PROCESSED frames are `step` apart, so a run break is a jump of more
            # than one processed slot. Working in slot indices keeps this stride-correct.
            pos = np.searchsorted(span, missing)
            breaks = np.where(np.diff(pos) > 1)[0]
            for run in np.split(pos, breaks + 1):
                if run.size:
                    gaps.append(run.size * step / fps)
        entry.update(n_gaps=len(gaps),
                     longest_gap_s=round(float(max(gaps)), 3) if gaps else 0.0,
                     total_gap_s=round(float(sum(gaps)), 3) if gaps else 0.0,
                     median_recovery_s=round(float(np.median(gaps)), 3) if gaps else 0.0)
        out[name] = entry
    return out


def greedy_wrist_tracks(frame_idx, kp2d, gate_px, maxgap_slots=10, step=1):
    """Minimal greedy wrist-proximity tracker, deliberately the same algorithm the mono-pipeline's
    ``postprocess.build_tracks`` uses, so identity metrics computed here are comparable to what that
    stage will report. Returns a list of row-index lists, one per track.
    """
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    kp2d = np.asarray(kp2d)
    maxgap_abs = int(maxgap_slots) * int(step)
    tracks = []
    for row in np.argsort(frame_idx, kind='stable'):
        f = int(frame_idx[row])
        xy = kp2d[row, WRIST]
        best, best_d = None, gate_px
        for track in tracks:
            if 0 < f - track['last_frame'] <= maxgap_abs:
                d = float(np.hypot(*(track['last_xy'] - xy)))
                if d < best_d:
                    best, best_d = track, d
        if best is None:
            tracks.append(dict(last_xy=xy, last_frame=f, rows=[row]))
        else:
            best['last_xy'] = xy
            best['last_frame'] = f
            best['rows'].append(row)
    return [t['rows'] for t in tracks]


def track_metrics(frame_idx, kp2d, hand_label, width, fps, step=1, gate_frac=0.08,
                  maxgap_slots=10, min_len=1):
    """Track fragmentation and identity switches.

    * ``label_flips`` counts consecutive detections WITHIN one track whose handedness label differs.
      One physical hand does not change identity mid-track, so every flip is an error somewhere.
    * ``fragments_per_hand`` counts how many separate tracks carry each label. Fragmentation is the
      mechanism behind "the hand was detected but the output is missing": the mono-pipeline discards
      tracks shorter than ``--min-len`` (10 detections by default), so three 6-detection fragments
      become nothing while one 18-detection track survives.
    """
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    hand_label = np.asarray(hand_label, dtype=np.int64)
    gate_px = gate_frac * float(width) * max(1, int(step))
    tracks = greedy_wrist_tracks(frame_idx, kp2d, gate_px, maxgap_slots, step)

    kept = [rows for rows in tracks if len(rows) >= min_len]
    flips = 0
    for rows in kept:
        order = sorted(rows, key=lambda r: int(frame_idx[r]))
        labels = [int(hand_label[r]) for r in order if int(hand_label[r]) in (0, 1)]
        flips += sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)

    per_hand = collections.Counter()
    for rows in kept:
        labels = [int(hand_label[r]) for r in rows if int(hand_label[r]) in (0, 1)]
        if labels:
            per_hand[int(np.round(np.mean(labels)))] += 1

    lengths = [len(rows) for rows in kept]
    return dict(
        gate_px=round(float(gate_px), 1), maxgap_slots=int(maxgap_slots), min_len=int(min_len),
        n_tracks_raw=len(tracks), n_tracks_kept=len(kept),
        n_tracks_dropped_short=int(sum(1 for rows in tracks if len(rows) < min_len)),
        detections_lost_to_short_tracks=int(sum(len(rows) for rows in tracks
                                                if len(rows) < min_len)),
        label_flips=int(flips),
        fragments_left=int(per_hand.get(0, 0)), fragments_right=int(per_hand.get(1, 0)),
        track_len_median=int(np.median(lengths)) if lengths else 0,
        track_len_max=int(max(lengths)) if lengths else 0,
    )


# ------------------------------------------------------------------ agreement (PROXY)


def fingertip_agreement(kp3d_cam, K, other_kp2d, other_conf=None, min_conf=0.3):
    """PROXY for fingertip accuracy: 2D distance between the fitted 3D's projection and an
    independent 2D observation, split into fingertips / all joints.

    This is disagreement, not error. Two models that agree may both be wrong. Use it to RANK
    variants against each other on the same clip, never as an absolute accuracy figure.
    """
    kp3d_cam = np.asarray(kp3d_cam)
    other_kp2d = np.asarray(other_kp2d, dtype=np.float64)
    if kp3d_cam.shape[0] == 0 or other_kp2d.shape[0] == 0:
        return {}
    proj = project_pinhole(kp3d_cam, np.asarray(K, dtype=np.float64))
    err = np.linalg.norm(proj - other_kp2d, axis=-1)              # (N, 21)

    mask = np.isfinite(err)
    if other_conf is not None:
        mask &= np.asarray(other_conf) >= min_conf

    def summarise(e, m):
        vals = e[m]
        if vals.size == 0:
            return dict(n=0, median_px=None, p95_px=None)
        return dict(n=int(vals.size), median_px=round(float(np.median(vals)), 3),
                    p95_px=round(float(np.percentile(vals, 95)), 3))

    tips = np.zeros(err.shape[1], bool)
    tips[TIPS] = True
    return dict(is_proxy=True,
                measures='2D disagreement between the fitted 3D projection and an independent 2D model',
                all_joints=summarise(err, mask),
                fingertips=summarise(err[:, tips], mask[:, tips]),
                non_fingertips=summarise(err[:, ~tips], mask[:, ~tips]))


def fingertip_error(kp3d_cam, gt_kp3d_cam, valid=None):
    """Real fingertip error in millimetres against ground truth. Nothing to consume yet.

    ``gt_kp3d_cam`` must be (N,21,3) camera-frame metres, row-aligned with ``kp3d_cam``.
    Reports both absolute (camera-space) and root-relative error, because a pipeline can be right
    about articulation and wrong about placement, and those are separate defects.
    """
    kp3d_cam = np.asarray(kp3d_cam, dtype=np.float64)
    gt = np.asarray(gt_kp3d_cam, dtype=np.float64)
    if kp3d_cam.shape != gt.shape:
        raise ValueError(f'shape mismatch: prediction {kp3d_cam.shape} vs gt {gt.shape}')
    if kp3d_cam.shape[0] == 0:
        return {}

    mask = np.isfinite(kp3d_cam).all(-1) & np.isfinite(gt).all(-1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)

    abs_err = np.linalg.norm(kp3d_cam - gt, axis=-1) * 1000.0
    rel_err = np.linalg.norm((kp3d_cam - kp3d_cam[:, WRIST:WRIST + 1])
                             - (gt - gt[:, WRIST:WRIST + 1]), axis=-1) * 1000.0
    tips = np.zeros(abs_err.shape[1], bool)
    tips[TIPS] = True

    def summarise(e, m):
        vals = e[m]
        if vals.size == 0:
            return dict(n=0, mean_mm=None, median_mm=None, p95_mm=None)
        return dict(n=int(vals.size), mean_mm=round(float(vals.mean()), 2),
                    median_mm=round(float(np.median(vals)), 2),
                    p95_mm=round(float(np.percentile(vals, 95)), 2))

    return dict(is_proxy=False,
                camera_space=dict(all_joints=summarise(abs_err, mask),
                                  fingertips=summarise(abs_err[:, tips], mask[:, tips])),
                root_relative=dict(all_joints=summarise(rel_err, mask),
                                   fingertips=summarise(rel_err[:, tips], mask[:, tips])))


def foot_candidates(kp2d, kp3d_cam, K, height, wrist_y_frac=0.75, max_hand_m=0.30):
    """WEAK PROXY for foot false positives. Geometry cannot separate a foot from a hand.

    ``hand_shape_score`` in the mono-pipeline's postprocess scores a synthetic FOOT higher than a
    real hand, which is exactly why that repo added a CLIP image classifier. So this returns only two
    honest things:

    * ``low_in_frame`` - detections whose wrist sits below ``wrist_y_frac`` of the image height.
      In head-mounted video your own feet appear at the bottom. This is a SHORTLIST to review or feed
      to CLIP, not a count of feet: a hand resting in your lap lands here too.
    * ``degenerate_span`` - detections spanning more pixels than a ``max_hand_m`` hand could at their
      own depth. These are real defects and the count IS meaningful.

    Counting actual foot false positives needs the image classifier or a human. Treat
    ``low_in_frame`` as a work queue.
    """
    kp2d = np.asarray(kp2d, dtype=np.float64)
    kp3d_cam = np.asarray(kp3d_cam, dtype=np.float64)
    if kp2d.shape[0] == 0:
        return {}
    K = np.asarray(K, dtype=np.float64)
    f0 = float((K[0, 0] + K[1, 1]) / 2.0)

    wrist_y = kp2d[:, WRIST, 1] / float(height)
    low = wrist_y > wrist_y_frac

    span = np.maximum(kp2d[..., 0].max(1) - kp2d[..., 0].min(1),
                      kp2d[..., 1].max(1) - kp2d[..., 1].min(1))
    z = np.clip(kp3d_cam[:, WRIST, 2], 1e-3, None)
    degenerate = span > (max_hand_m * f0 / z)

    return dict(is_proxy=True, n=int(kp2d.shape[0]),
                low_in_frame=int(low.sum()), low_in_frame_frac=round(float(low.mean()), 5),
                wrist_y_frac_threshold=wrist_y_frac,
                degenerate_span=int(degenerate.sum()),
                degenerate_span_frac=round(float(degenerate.mean()), 5),
                max_hand_m=max_hand_m,
                note='low_in_frame is a review shortlist, not a foot count; run the CLIP '
                     'hand-vs-foot classifier or a human over it to get true positives')
