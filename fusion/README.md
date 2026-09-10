# `fusion/` — EgoForce as a drop-in for the WiLoR + MediaPipe hand pipeline

These stages evaluate EgoForce against the existing `mono-pipeline-v2/hand_labelling_21kp` pipeline,
and add a fusion that jointly optimises articulation rather than rigidly re-posing a fixed skeleton.

**Nothing here has been run.** It is written, linted, and its GPU-independent logic is covered by
`fusion/selftest.py` (34 checks, all passing). EgoForce inference, RTMPose inference and the MANO
forward pass are untested — see [`../plan_of_action.md`](../plan_of_action.md).

## The one fact that makes this work

All three models emit **the same 21 keypoints in the same order** — wrist, then thumb, index, middle,
ring, pinky, each running MCP → PIP → DIP → tip. Verified, not assumed:

| Model | Evidence |
| --- | --- |
| EgoForce | `mano_joint_mapping` in [`models/mano_layer.py`](../models/mano_layer.py) + fingertips appended thumb→pinky |
| RTMPose-m Hand5 | trained against mmpose's `configs/_base_/datasets/coco_wholebody_hand.py` |
| MediaPipe | the order the existing pipeline's `hand_topology.py` documents |

So there is **no remapping anywhere**. `fusion/selftest.py` asserts the derivation and asserts our
edge list is byte-identical to `hand_topology.HAND_EDGES`, so the two repos cannot silently drift.

## Stages

```
                       ┌─ run_egoforce_3d.py ──► <stem>_3d_keypoints.npz  (stage-1b schema)
rectified pinhole video┤                                    │
        + K            └─ run_rtmpose_2d.py ───► <stem>_2d_keypoints.npz  (stage-1a schema)
                              (boxes from ──────────────────┘
                               the 3D npz)
                                   │
                      ┌────────────┴─────────────┐
                      ▼                          ▼
        fuse_egoforce_rtmpose.py      the existing fuse_2d_3d.py
        (articulated MANO refit)      (rigid PnP, unmodified)
                      │                          │
                      └────────────┬─────────────┘
                                   ▼
                    <stem>_hand21_keypoints.npz  (fused schema)
                                   │
                                   ▼
              run_clip.py --from-npz  →  tracking / filtering / handedness / overlay
                                          (CPU, no GPU needed)
```

| File | Role |
| --- | --- |
| `run_egoforce_3d.py` | Drop-in for `run_wilor_3d.py`. Emits the identical stage-1b npz, plus EgoForce's MANO params and its own 2D head. |
| `run_rtmpose_2d.py` | Drop-in for `run_mediapipe_2d.py`. Top-down RTMPose-m Hand5, boxes reused from the 3D stage (`--boxes npz`) or from an independent mmdet detector (`--boxes mmdet`). |
| `fuse_egoforce_rtmpose.py` | The articulated fusion, plus `--mode egoforce-only` for the Experiment 1 baseline. Writes the mono-pipeline's fused schema. |
| `mano_refit.py` | The refit itself: confidence-weighted reprojection + depth/pose/shape priors + accept-reject gates. Torch only, runs on CPU. |
| `metrics.py` | Coverage, gap/recovery, track fragmentation, identity flips, depth/reprojection QC, and clearly-labelled proxies for fingertip error and foot false positives. |
| `compare_runs.py` | Recomputes every metric from the fused npz so the rigid and articulated fusions land on the same axes, then writes `comparison.{csv,json,md}`. |
| `run_testcases.py` + `testcases.yaml` | The test matrix. `--dry-run` prints every command without executing. |
| `render_comparison.py` | Review overlay: grey = EgoForce raw, colour = what was written, white dots = RTMPose's observation. |
| `topology.py`, `video_io.py`, `calibration.py` | Shared keypoint layout/drawing, frame-selection arithmetic, `calibration.json` reading. |
| `selftest.py` | Everything testable without a GPU or a checkpoint. Run it first. |

## Why the articulated refit

The existing `fuse_2d_3d.py::pnp_refit` keeps WiLoR's root-relative skeleton **rigid** and solves one
rotation + translation to land it on MediaPipe's pixels. That guarantees MANO-consistent bone lengths
— but a wrongly bent finger stays wrongly bent no matter how confident the 2D is. Two models run and
their finger estimates are never jointly optimised.

`mano_refit.py` optimises the MANO **parameters** instead (global orientation, 15 joint rotations,
shape, translation), so articulation can move to satisfy the image evidence while every candidate is
still a MANO sample. Two things are load-bearing rather than decorative:

- **The depth prior.** A 2D reprojection loss is scale–depth degenerate: a hand twice as large at
  twice the depth projects identically. RTMPose contributes nothing to depth, so translation stays
  anchored to EgoForce, which is the model that actually resolved it.
- **The reject gates.** Because articulation is now free, a refit can "win" on reprojection by
  stretching bones or sliding in depth. `accept_refit` rejects on bone-length change, depth change,
  reprojection ceiling, and failure to improve on EgoForce; rejected rows fall back to raw EgoForce
  3D and are labelled so downstream can see it.

The **rigid** fusion is deliberately not reimplemented — the existing script accepts our two producer
npz files unchanged, so the rigid baseline is free and stays bit-identical to the historical one.
`run_testcases.py` wires it up as `T2a`, which is what isolates the model swap from the fusion change.

## Source labels

The fused npz uses the mono-pipeline's **existing** source vocabulary so `postprocess.py` works
untouched. An additive `producer` array records the real model names.

| `source` | Meaning here | `depth_measured` |
| --- | --- | --- |
| `fused` | EgoForce + RTMPose, refit accepted | ✅ |
| `wilor_pnpfail` | matched, refit rejected → raw EgoForce 3D | ✅ |
| `wilor` | EgoForce only, RTMPose had no detection | ✅ |
| `lifted_2d` | RTMPose only → depth **borrowed**, not measured | ❌ |

## Running

```bash
# 0. always first, needs no GPU
python fusion/selftest.py

# 1. see exactly what would run
python fusion/run_testcases.py --config fusion/testcases.yaml --dry-run

# 2. one clip, one case
python fusion/run_egoforce_3d.py --video clip.mp4 --out work --calib calibration.json --overlay
python fusion/run_rtmpose_2d.py  --video clip.mp4 --out work \
    --boxes npz --boxes-npz work/clip_3d_keypoints.npz --overlay
python fusion/fuse_egoforce_rtmpose.py --egoforce work/clip_3d_keypoints.npz \
    --rtmpose work/clip_2d_keypoints.npz --out work --video clip.mp4 --overlay

# 3. the whole matrix, then the comparison table
python fusion/run_testcases.py --config fusion/testcases.yaml \
    --mono-pipeline /path/to/hand_labelling_21kp
```

Prerequisites: the `egoforce` conda env, `_DATA/` weights, `mim install "mmpose>=1.3.2"`, and
`bash scripts/download_rtmpose_hand5.sh`. See [`../plan_of_action.md`](../plan_of_action.md) for the
GPU requirement and the full pending list.
