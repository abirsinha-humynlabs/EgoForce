# Provenance of the vendored stabilisation passes

`temporal_smooth.py`, `rigidify.py` and `kinematic.py` in this directory are **vendored verbatim**
(header comment aside) from:

    https://github.com/Maiemdiab/egocentric-hand-stabilisation
    commit 2ca531b

They are kept unmodified so they stay diffable against upstream. Adaptation to this repo's npz
schema lives in `fusion/stabilise_run.py`, not in these files.

## Licence status — UNRESOLVED

**The upstream repository carries no LICENCE file.** Absent an explicit licence, the default is
all-rights-reserved. This code was vendored on explicit instruction to use it, but before any of
this ships outside the team, someone needs to obtain a licence grant from the author or reimplement
the passes independently. Recording it here rather than leaving it implicit.

## Why these three, and not the other passes

Upstream ships four passes. Two are used here:

| pass | used | why |
| --- | --- | --- |
| `temporal_smooth.py` | yes | Zero-phase, gap-aware, robust smoothing. Upstream measured per-frame estimator noise at **83%** of visible jitter, which is the defect we have. |
| `rigidify.py` (+ `kinematic.py`) | yes | Exact bone rigidity + articulation smoothed in pose space. Our baseline bone-length CV was 5.1% median / 11.1% max - the same defect upstream measured at 12.32%. |
| `rescue_handedness.py` | no | Recovers detections that were found but never given a left/right label. EgoForce's handedness is **structural** (separate left/right crop streams), so there is no unlabelled pool to rescue. Revisit if a run ever emits `is_right == -1`. |
| `mint_bridge2.py` | no | Fills gaps using a second model's (MINT's) motion. We have no MINT output. This is the obvious next lever for the 0.67 right-hand coverage, and type 2's RTMPose stream is a candidate second model - but RTMPose gives no depth, so the depth-interpolation half would need rethinking. |
