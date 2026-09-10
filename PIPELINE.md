# The EgoForce hand pipeline — design and engineering notes

**What this document is.** The full account of what was built in `fusion/` and `demo/render_landmarks.py`,
why each piece is shaped the way it is, how the internals actually work, and what was verified versus
assumed. It is the document to read before changing any of it.

Companion documents, deliberately not duplicated here:

| Document | Answers |
| --- | --- |
| [`fusion/README.md`](fusion/README.md) | "How do I run this?" — quick orientation and the stage map |
| [`plan_of_action.md`](plan_of_action.md) | "What's next?" — pending steps, GPU requirement, open issues |
| **this file** | "What is it and why is it like this?" |

**Status.** All code is written, linted and committed on `feat/egoforce-rtmpose-fusion`. The
GPU-independent logic is covered by `fusion/selftest.py` (34 checks, passing). **No stage has been
run on real footage** — there was no GPU on the development machine. §9 is the honest inventory of
what that leaves unverified.

---

## 1. The problem

An existing pipeline (`mono-pipeline-v2/hand_labelling_21kp`) produces 21 hand keypoints per hand
from rectified egocentric video using WiLoR for 3D and MediaPipe for 2D, fused by rigid PnP. A
human-in-the-loop audit found three defects. They have three different causes, and the single most
important design decision in this work was refusing to treat them as one problem:

| Defect | Actual cause | Can a better pose model fix it? |
| --- | --- | --- |
| A foot tracked as a hand | **Detection.** 21 keypoints cannot separate a foot from a hand — the existing `hand_shape_score` scores a synthetic foot *higher* than a real hand | **No.** Needs detector fine-tuning with feet/shoes as negatives, or an image classifier |
| Hands missing from the output | **Association and rejection rules.** Detections often exist but are discarded — tracks under 10 detections are dropped (`postprocess.py:63`, `run_clip.py:177`) | **Only indirectly**, by producing detections that fragment less |
| Wrong finger and wrist geometry | **Pose estimation and fusion** | **Yes.** This is the only row a model swap addresses |

So this work is scoped to the third row. The metrics (§5.5) are built to *expose* the first two
rather than quietly absorb them — because a pipeline that drops a foot false positive and a pipeline
that never detected the foot produce the same output, and only one of them has been fixed.

A second, subtler defect sits underneath the third. The existing fusion
(`fuse_2d_3d.py:102::pnp_refit`) keeps WiLoR's root-relative skeleton **rigid** and solves a single
rotation and translation that lands it on MediaPipe's pixels. That is a sound choice for guaranteeing
MANO-consistent bone lengths, and it is why the naive alternative — moving each keypoint along its
own ray — was rejected there. But it means **a wrongly bent finger stays wrongly bent** no matter how
confident the 2D observation is. Two models run, and their finger estimates are never jointly
optimised. Fixing that is what §5.3 is for.

---

## 2. Design constraints

Six constraints shaped every decision. Most of the non-obvious choices later in this document follow
directly from one of them.

1. **Slot into the existing pipeline, don't replace it.** Its stage 3/4 (tracking, filtering,
   handedness voting, overlay) encodes real hard-won knowledge — the left/right geometric voting, the
   degenerate-span gate, the wrist-gated duplicate suppression. Rewriting that would discard it.
2. **EgoForce requires CUDA and TensorRT; the fusion must not.** `demo/inference.py:8` imports
   `torch_tensorrt` at module scope. That is unavoidable for inference, but it must not contaminate
   the parts that could otherwise be iterated on a laptop.
3. **The input contract is a rectified pinhole stream plus its true intrinsics.** That is what the
   existing pipeline consumes, so the new stages consume it too. Uncalibrated footage is a separate
   path (§5.7).
4. **The comparison must isolate the model swap from the fusion change.** If EgoForce+RTMPose beats
   WiLoR+MediaPipe, that alone does not say whether the model or the fusion did the work. This
   requires a rigid-fusion row in the matrix, which in turn requires the new producers to be
   consumable by the *old* fusion code.
5. **There is no ground truth.** No annotated fingertips, no labelled feet. Every metric must
   therefore be either an exact property of the output, or a clearly-labelled proxy. Presenting
   disagreement as accuracy would be the most damaging thing this work could do.
6. **The development machine had no GPU.** So anything that *can* be made testable without one
   must be, or the code ships entirely unexercised.

---

## 3. Architecture

Constraint 1 and constraint 4 together forced the central decision: **emit the existing pipeline's
exact npz schemas.** Once the new producers are byte-compatible with `run_wilor_3d.py` and
`run_mediapipe_2d.py`, three things fall out for free — the old rigid fusion becomes a test case,
the old stage 3/4 runs unchanged on our output, and the old QC scripts keep working.

```
rectified pinhole video + K (or calibration.json)
        │
        ├──────────────► fusion/run_egoforce_3d.py ──► <stem>_3d_keypoints.npz   [stage-1b schema]
        │                    (CUDA + TensorRT)              │  + MANO params
        │                                                   │  + its own 2D head
        │                                                   ▼
        └──────────────► fusion/run_rtmpose_2d.py  ──► <stem>_2d_keypoints.npz   [stage-1a schema]
                             (mmpose)                       │
                             boxes reused from the 3D npz ──┘
                                                            │
                      ┌─────────────────────────────────────┴──────────────┐
                      ▼                                                    ▼
       fusion/fuse_egoforce_rtmpose.py                    the EXISTING fuse_2d_3d.py
       articulated MANO refit  (T2b)                      rigid PnP, unmodified  (T2a)
       also --mode egoforce-only (T1)                                     │
                      │                                                    │
                      └─────────────────────────┬──────────────────────────┘
                                                ▼
                              <stem>_hand21_keypoints.npz   [fused schema]
                                                │
                   ┌────────────────────────────┼────────────────────────────┐
                   ▼                            ▼                            ▼
   run_clip.py --from-npz          fusion/compare_runs.py      fusion/render_comparison.py
   tracking / filtering /          comparison.{csv,json,md}    review overlay
   handedness / overlay  (CPU)
```

### File map

| File | Lines | Role |
| --- | --- | --- |
| `fusion/topology.py` | 176 | The 21-keypoint layout, skeleton, drawing, pinhole projection. Single source of truth. |
| `fusion/video_io.py` | 161 | Frame-selection arithmetic and the overlay writer. |
| `fusion/calibration.py` | 50 | `calibration.json` reading, same schemas and warnings as `run_clip.read_K`. |
| `fusion/run_egoforce_3d.py` | 369 | Stage 1b producer. |
| `fusion/run_rtmpose_2d.py` | 311 | Stage 1a producer. |
| `fusion/mano_refit.py` | 327 | The articulated refit. Torch only — no CUDA needed. |
| `fusion/fuse_egoforce_rtmpose.py` | 572 | Fusion orchestration: matching, source labelling, depth borrowing, dedup, QC. |
| `fusion/metrics.py` | 344 | Exact QC and coverage metrics; clearly-labelled proxies. |
| `fusion/compare_runs.py` | 257 | Cross-case comparison table. |
| `fusion/run_testcases.py` | 307 | Test matrix runner with `--dry-run`. |
| `fusion/render_comparison.py` | 150 | Three-layer review overlay. |
| `fusion/selftest.py` | 621 | 34 GPU-free checks. |
| `fusion/testcases.yaml` | 112 | The matrix definition. |
| `demo/render_landmarks.py` | 548 | Standalone viewer for uncalibrated footage (AnyCalib intrinsics). |
| `scripts/download_rtmpose_hand5.sh` | 95 | Config + checkpoint fetch, two routes. |

---

## 4. The fact that makes fusion possible

All three models emit **the same 21 keypoints in the same order**: wrist, then thumb, index, middle,
ring, pinky, each running MCP → PIP → DIP → tip. So there is **no remapping anywhere** in this
pipeline. That is a strong claim, so here is how each was established rather than assumed.

**EgoForce.** Derived, not read off a doc. `models/mano_layer.py:33` defines

```python
mano_joint_mapping = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

applied as `out_landmarks = out_landmarks[:, self.joint_mapper]`. The tensor being indexed is MANO's
own 16 joints (0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb) with five
fingertip vertices appended at 16-20 in the order given by `MANO_FINGERTIP_VERT_INDICES`
(`models/mano_layer.py:39`): thumb 744, index 320, middle 443, ring 554, pinky 671. Working the
mapping through: index 0 → wrist; 1,2,3 → MANO 13,14,15 = thumb; 4 → extra[0] = thumb tip; 5,6,7 →
MANO 1,2,3 = index; 8 → extra[1] = index tip; and so on. That is exactly OpenPose-21.

**RTMPose-m Hand5.** Its config sets
`from_file='configs/_base_/datasets/coco_wholebody_hand.py'`, whose keypoint names are
`wrist, thumb1..4, forefinger1..4, middle_finger1..4, ring_finger1..4, pinky_finger1..4`. Same order.

**MediaPipe.** The order the existing `hand_topology.py` already documents and relies on.

Because a silent divergence here would mis-wire every fused row while still producing plausible
pictures, `fusion/selftest.py` asserts **both** the EgoForce derivation above (it reconstructs
`JOINT_NAMES` from `mano_joint_mapping` independently) **and** that `fusion/topology.HAND_EDGES` is
byte-identical to the existing pipeline's `hand_topology.HAND_EDGES`. If either repo ever reorders,
a test fails instead of the output quietly rotting.

---

## 5. How each stage works

### 5.1 EgoForce 3D producer — `fusion/run_egoforce_3d.py`

Wraps `demo/inference.py::Inference` and writes the stage-1b schema. It is a drop-in replacement for
`run_wilor_3d.py`, and three of that script's complications simply disappear:

- **No focal-length hack.** WiLoR/HaMeR reconstruct with a virtual focal length, so the old stage has
  to pass `focal_length = f0 * 256 / max(W, H)` to make `scaled_focal_length == f0` and then assert
  at runtime that the trick took. EgoForce's ray-space solver consumes the real intrinsics directly
  (`core/rss.py::unproject_unit_rays`), so `fx, fy, cx, cy` are used as-is. **Anisotropic pixels are
  supported** rather than averaged to a single `f0`, which the old stage warns about and cannot fix.
- **No principal-point rebase.** WiLoR pins the principal point to the image centre, so the old stage
  applies `X += (W/2 - cx)·Z/f` to move the 3D into the true-K camera. EgoForce never makes that
  assumption. We still write `pp_rebase=True`, because downstream that flag means "`kp3d_cam` is
  expressed in the camera the stored `K` describes" — which holds here by construction, and is proven
  per run by the reprojection self-check.
- **Handedness is structural.** EgoForce runs a separate left and right hand/forearm crop pair, so
  `is_right` records *which stream produced the row*, not a per-crop guess. This matters a lot: the
  existing pipeline's biggest reported complaint was left/right swapping, and its track-level
  geometric voting exists precisely to repair the unreliable per-crop guess that WiLoR and MediaPipe
  provide. EgoForce removes the need at the source.

**Self-checks written to `_3d_meta.json`.** Two numbers, with different jobs:

- `reproj_median_px` — projects the stored `kp3d_cam` through the stored `K` and compares to the
  stored `kp2d`. This is a *correctness* check: it must be < 1 px, and if it is not, the 3D and the
  intrinsics describe different cameras and everything downstream is meaningless.
- `head_vs_lift_median_px` — how far EgoForce's own 2D keypoint head sits from its own 3D lift. This
  is *new information*, not a correctness check. A large value says the 3D lift is fighting the 2D
  head, which is worth understanding before weighting an external 2D model against it.

**Additive outputs.** Beyond the WiLoR schema it also writes the MANO parameters (needed by the
refit), the network's own 2D head with per-joint confidence (an observation independent of the 3D
lift), and the forearm chain. All additive, all ignored by the old loaders.

**Exposed knobs that matter for the missing-hand defect.** `--hand-conf` reaches the YOLO hand
detector (`Inference.yolo_track_cfg['conf']`) and `--max-misses` controls how long a hand track
survives without a detection (`grouped_hand_track_max_misses`). These are the two levers that trade
false positives for recall, which is what test case `T1c` probes. Note the RTMDet score floor that
supplies the *forearm* boxes is hardcoded at 0.3 in `demo/inference.py::detect_bounding_boxes` and is
not parameterised — recorded rather than silently worked around.

### 5.2 RTMPose 2D producer — `fusion/run_rtmpose_2d.py`

Drop-in for `run_mediapipe_2d.py`. RTMPose is **top-down**, so it needs boxes, and where those come
from is a real design choice with a real trade-off:

- **`--boxes npz` (default)** reads the hand boxes EgoForce already detected out of the 3D npz. Every
  2D row then corresponds 1:1 to a 3D row on the same frame and the same physical hand, so the fusion
  needs no wrist matching and *cannot* mis-pair. It also makes the `width`/`height`/`step` agreement
  that `fuse_2d_3d.py` checks true by construction rather than by the operator remembering to pass
  matching flags.
- **`--boxes mmdet`** runs an independent hand detector. The cost of the default is that both streams
  inherit EgoForce's detection errors — a foot detected once is a foot in both — so this mode exists
  to answer whether detector choice matters (`T2d`).

**Two conventions worth flagging**, both easy to get silently wrong:

1. **Frames are handed over as BGR.** The model's `data_preprocessor` sets `bgr_to_rgb=True`. The
   MediaPipe stage this replaces needs RGB. Getting this backwards does not crash — it degrades
   accuracy quietly.
2. **`score` means something different.** The MediaPipe stage stores a *handedness* score there.
   RTMPose has no handedness head, so we store the mean per-joint keypoint confidence — which is what
   a fusion should weight by anyway. `postprocess.py` only uses this field as a handedness-vote
   weight, so the substitution is safe, and `score_semantics` records it in the npz.

When boxes come from the 3D npz, `is_right` is inherited from the EgoForce row (structural). With an
independent detector there is no handedness, so `-1` is written — which `postprocess.py:374` already
treats as "no vote".

### 5.3 The articulated refit — `fusion/mano_refit.py`

This is the intellectual core, and the part that does something the rigid fusion structurally cannot.

Instead of re-posing a fixed skeleton, it optimises the MANO **parameters** — global orientation, the
15 joint rotations, shape and translation — so articulation itself can move to satisfy the image
evidence, while every candidate remains a MANO sample and therefore a plausible hand by construction.

**The objective**, per detection *i*:

```
L_i =  w_reproj  · Σ_j c_ij · huber(‖π(J_ij) − x_ij‖, δ) / Σ_j c_ij     confidence-weighted image evidence
     + w_depth   · ‖t_i − t⁰_i‖²                                        keep EgoForce's metric placement
     + w_pose    · mean_k (θ_ik − θ⁰_ik)²                                stay near EgoForce's articulation
     + w_orient  · mean_k (r_ik − r⁰_ik)²
     + w_beta    · mean_k (β_ik − β⁰_ik)²                                keep EgoForce's bone lengths
     + w_limit   · mean_j relu(‖ω_ij‖ − ω_max)²                          reject implausible joint rotations
```

where `x_ij`/`c_ij` are the RTMPose landmark and its confidence, `π` is the pinhole projection through
the same `K` the 3D stage stored, `ω_ij` is joint *j*'s axis-angle vector, and superscript `⁰` is
EgoForce's prediction — used as both initialisation and prior.

**Two terms are load-bearing, not regularisation to be tuned away:**

- **`w_depth`.** A 2D reprojection loss is *scale–depth degenerate*: a hand twice as large at twice
  the depth projects to identical pixels. RTMPose contributes nothing to depth. So if translation is
  not anchored to EgoForce — the model that actually resolved depth — the fit will happily slide along
  the viewing ray and destroy the metric placement that was the whole reason to use EgoForce.
- **The reject gates** (below). Because articulation is now free, a refit can "win" on reprojection by
  stretching bones or sliding in depth. Rigid PnP never needed this guard; this does.

**On the joint limits, honestly:** `w_limit` is a coarse cap on each joint's total rotation
*magnitude* (default 1.75 rad ≈ 100°), **not** a per-DOF anatomical range. MANO's per-joint axis-angle
frames are not clean flexion axes, so a real anatomical limit set would have to be derived and
validated separately. The load-bearing plausibility constraint is the MANO parameterisation plus
`w_pose` pulling toward EgoForce's already-plausible prediction. Do not read `w_limit` as anatomy.

**Parameterisation.** Rotations are optimised in the 6D representation (Zhou et al.) and converted
with `rotation_6d_to_axis_angle_direct` — the same conversion `models/limb_model.py` uses, so the
numerics match the forward pass EgoForce itself ran. 6D avoids the singularities and wrap-around that
make axis-angle a poor optimisation space.

**Mechanics.** Adam over `[betas, go6, hp6, transl]`, lr 0.02, 80 iterations, batched over all
detections at once in chunks of 256 (`--chunk`). MANO's `forward_kinematics` splits left and right
internally by mask, so mixed-handedness batches work. Batching matters: a per-detection loop over a
10k-detection clip would be dominated by Python overhead.

**Accept / reject — `accept_refit`.** A refit is kept only if all of:

| Gate | Default | Why |
| --- | --- | --- |
| joints finite, all `z > 0` | — | a hand behind the camera is not a hand |
| weighted-median reprojection | ≤ 20 px | same role as the old `--pnp-max-px` |
| max bone-length change | ≤ 15 % | the hand must still be the same hand |
| wrist depth change | ≤ 5 cm | it must not have slid along the ray |
| reprojection improved on EgoForce | required | if the refit made things worse, it has no case |

Rejected rows fall back to **raw EgoForce 3D** and are labelled, so a high reject rate is visible in
the stats rather than hidden. The per-reason breakdown is written to `_fuse_stats.json`, and
`selftest.py` asserts that the reason counts partition the input exactly — otherwise a rejection could
be miscounted and the diagnosis would be wrong.

**This module needs no CUDA.** Torch + smplx + the MANO files only. That is deliberate (constraint 2):
once the producer npz files exist, refit weights can be re-tuned on a laptop.

### 5.4 Fusion orchestration — `fusion/fuse_egoforce_rtmpose.py`

**Matching.** Exact-bbox first (`atol=0.51` px, same frame), which pairs everything when
`--boxes npz` was used, since each RTMPose row was produced *from* an EgoForce box. Wrist proximity
within `--match-dist` is the fallback for independently-detected boxes, matching the old pipeline's
behaviour. Pairs never cross frames.

**Source labels.** This is the ugliest compromise in the design and worth stating plainly. The old
`postprocess.py` branches on exact source strings, so emitting honest new names like
`egoforce+rtmpose` would break it. We therefore reuse its vocabulary:

| `source` | What it means here | `depth_measured` |
| --- | --- | --- |
| `fused` | EgoForce + RTMPose, refit accepted | ✅ |
| `wilor_pnpfail` | matched, refit rejected → raw EgoForce 3D | ✅ |
| `wilor` | EgoForce only, RTMPose had no detection | ✅ |
| `lifted_2d` | RTMPose only → depth **borrowed**, not measured | ❌ |

An additive `producer` array carries the real model names, because a row labelled `wilor` contains no
WiLoR. **This compromise has a cost that will bias the results** — see §11.

**Depth borrowing for 2D-only rows.** Mirrors the old pipeline: look at the five temporally nearest
frames with measured depth, take the candidate whose wrist is closest, accept it if within
`0.15·W` px, else fall back to the clip-wide median per-joint depth profile. Either way
`depth_measured=False`, so a metric consumer can exclude fabricated depth. `selftest.py` checks both
that the backprojection inverts the projection exactly and that the wrist-proximity gate actually
rejects a far-away donor.

**Duplicate suppression.** Same algorithm and thresholds as the old `dedup`, including the part that
is easy to get wrong: a candidate is suppressed only if it overlaps a kept detection **and their
wrists are close**. Without the wrist gate, the two-handed-manipulation case — one hand steadying a
part while the other reaches across it — gives IoMin ≈ 1.0 for two genuinely different hands and one
gets deleted. Two `fused` rows never suppress each other, since both are cross-model corroborated.
`selftest.py` covers both directions of this.

**`NOT_COMPUTED` is NaN, never 0.0.** Inherited reasoning, and worth preserving: a consumer filtering
`fuse_residual_px < 5` would otherwise keep every *uncorroborated* single-model row and drop the
*cross-model-corroborated* ones — exactly inverted.

### 5.5 Metrics — `fusion/metrics.py`

Constraint 5 (no ground truth) makes this section the one most likely to mislead, so every function
declares which kind it is.

**Exact — properties of the output, no interpretation:**

| Function | Measures |
| --- | --- |
| `depth_qc` | Physically impossible depths across **all 21 joints**, not just the wrist. Inherited insight: a plausible wrist with a 0.2 mm fingertip is a 13 cm-deep "hand", and a wrist-only check misses it. |
| `reprojection_qc` | Projects the 3D through the stored `K` and compares to the stored 2D, split by source. The one check that proves 3D and intrinsics describe the same camera. |
| `bone_consistency` | Per-bone length spread across detections. A MANO hand cannot breathe within a frame, but nothing stops shape parameters drifting between frames — which reads as a pulsing hand. This is the metric that catches an over-loose refit. |
| `coverage_metrics` | Coverage, gap count, longest gap, total gap, median recovery — per hand. |
| `track_metrics` | Track fragmentation, tracks dropped as short, detections lost to that, handedness flips within a track. |

Two details in `coverage_metrics` are easy to get wrong and are tested:

1. **Gaps are counted only *between* the first and last detection of that hand.** Frames before the
   first and after the last are excluded, because the hand may genuinely be out of frame — counting
   that as a miss would reward a model that hallucinates hands.
2. **The arithmetic is stride-aware.** Consecutive processed frames are `step` apart, so gap runs are
   computed in *processed-slot* space and converted with `slots · step / fps`. A stride-unaware
   version reports a 0.7 s gap as 0.23 s at `--sample-fps 10`. `selftest.py` pins both numbers.

`track_metrics` deliberately reimplements the old pipeline's greedy wrist tracker rather than
inventing a better one, so identity numbers computed here are comparable to what that stage will
report. `tracks_dropped_short` and `dets_lost_short` exist to make the missing-hand mechanism
visible: three 6-detection fragments become nothing under `--min-len 10` while one 18-detection track
survives.

**Proxies — labelled as such, in the function name or the returned `is_proxy` flag:**

| Function | What it really measures | What it does *not* tell you |
| --- | --- | --- |
| `fingertip_agreement` | 2D distance between the written 3D's projection and RTMPose's independent observation | Accuracy. Two models that agree may both be wrong. Use it to *rank* variants on one clip. |
| `foot_candidates` | `low_in_frame`: detections whose wrist sits low in the frame. `degenerate_span`: detections wider than a 30 cm hand could be at their own depth | Whether anything is a foot. `low_in_frame` is a **review queue** — a hand resting in your lap lands there too. Only `degenerate_span` is a real defect count. |

`fingertip_error` is the real thing — camera-space and root-relative error in millimetres, reported
separately because a pipeline can be right about articulation and wrong about placement. It is
written and tested against a synthetic known offset. **It has nothing to consume until clips are
annotated**, which is the honest state of affairs rather than a gap to paper over.

### 5.6 Test matrix and comparison

`fusion/testcases.yaml` defines seven cases (T1, T1b, T1c, T2a, T2b, T2c, T2d) each carrying the
*question it answers*, so a reader can tell why a row exists. `T2a` — the old rigid fusion on the new
producers — is the row that satisfies constraint 4: without it, a `T2b` win cannot be attributed.

`fusion/run_testcases.py` shells out to the stages. `--dry-run` prints every command without
executing, which given the GPU cost is the intended first move. The runner refuses to start while the
shipped placeholder clips are unedited, and **never guesses intrinsics** — either `calib:` or an
explicit `K`.

`fusion/compare_runs.py` **recomputes every metric from the fused npz** rather than reading the
per-case stats sidecar. That is not redundancy: `T2a` is produced by the old `fuse_2d_3d.py`, whose
sidecar has a different shape and carries no coverage or track metrics at all. Recomputing is the only
way the rigid and articulated fusions land on the same axes.

### 5.7 Overlays, and the uncalibrated path

`fusion/render_comparison.py` draws three layers on the same pixels, because "which overlay looks
smoothest" is the wrong question: **grey thin** = EgoForce's raw 3D projected (the prior), **coloured**
= the row actually written, **white dots** = RTMPose's observation (the evidence). When the coloured
skeleton has moved off the grey and onto the dots, the refit did its job; when it has moved off both,
the reject gates missed something. The HUD prints the source label and the gate numbers.

`demo/render_landmarks.py` is a separate, simpler tool for **uncalibrated** footage — a YouTube
egocentric clip has no rectified pinhole `K`, so it does not belong in the matrix. It estimates
intrinsics from one frame with AnyCalib (the same mapping the Gradio demo uses) and writes an overlay
plus a keypoint npz. It deliberately avoids the pytorch3d *rasteriser* by drawing from `pred_j2d`
with OpenCV, and it reimplements the AnyCalib→camera-model mapping locally rather than importing
`run_app.py`, which does `import spaces` — an HF-Spaces-only module absent from
`scripts/requirements.txt`.

---

## 6. Data contracts

Key counts are from an AST extraction of the actual `savez_compressed` calls, not from memory.

### Stage 1b — `<stem>_3d_keypoints.npz` (29 keys)

The first 17 are the WiLoR schema verbatim; the rest are additive and ignored by the old loaders.

| Key | Shape | Notes |
| --- | --- | --- |
| `kp3d_cam` | (N,21,3) | metres, camera frame, true-K convention |
| `kp2d` | (N,21,2) | `kp3d_cam` projected through `K` |
| `frame_idx` | (N,) | **absolute** source frame number |
| `is_right` | (N,) int8 | structural, not a guess |
| `bbox`, `cam_t` | (N,4), (N,3) | detector box; root translation |
| `processed_frames` | (M,) | frames fed to the model, for coverage accounting |
| `K` | (3,3) float64 | the intrinsics `kp3d_cam` is expressed in |
| `width`, `height`, `fps`, `step`, `sample_fps`, `start_frame`, `end_frame` | scalars | run provenance; `fuse_2d_3d.py` guards on the first four |
| `pp_rebase`, `model` | scalars | see §5.1 |
| `kp2d_head`, `kp2d_conf` | (N,21,2), (N,21) | the network's own 2D head and confidence |
| `mano_betas`, `mano_global_orient`, `mano_hand_pose`, `mano_transl`, `mano_rot_format` | (N,10), (N,6), (N,90), (N,3), str | refit inputs; rotations are 6D |
| `arm_kp3d`, `arm_kp2d`, `arm_bbox`, `arm_visible` | (N,3,3), (N,3,2), (N,4), (N,) | forearm chain |
| `keypoint_order` | str | `'OpenPose-21'` |

### Stage 1a — `<stem>_2d_keypoints.npz` (15 keys)

`kp2d`, `frame_idx`, `is_right`, `score`, `processed_frames`, `model`, `width`, `height`, `fps`,
`step` are the MediaPipe schema verbatim. Additive: `kp2d_conf` (N,21), `bbox` (N,4), `box_source`,
`keypoint_order`, `score_semantics`. See §5.2 on what `score` means.

### Fused — `<stem>_hand21_keypoints.npz` (28 keys + `depth_plausible`)

`kp3d_cam`, `kp3d_cam_wilor_raw`, `kp2d`, `frame_idx`, `is_right_wilor`, `is_right_mp`, `mp_score`,
`source`, `fuse_residual_px`, `depth_measured`, `K`, `n_dropped_dupes`, `n_pnp_fallback`, `width`,
`height`, `fps`, `step`, `fusion_mode`, `pp_rebase`, `depth_plausible` are the old fused schema
verbatim. Additive: `producer`, `is_right_egoforce`, `is_right_rtmpose`, `rtmpose_score`,
`kp2d_rtmpose`, `kp2d_rtmpose_conf`, `refit_bone_change`, `refit_depth_change_m`, `keypoint_order`.

The old field names are kept even where they now hold something else (`is_right_wilor` carries
EgoForce's handedness, `n_pnp_fallback` counts refit rejections) because `postprocess.py` reads those
exact names.

---

## 7. Changes to existing code

Kept minimal on purpose — the point was to add capability, not to refactor a working repo.

**`demo/inference.py` — additive only.** The `infer()` return dict gains eleven keys: the MANO
parameters, the 2D head mapped from crop space to image pixels, per-joint confidences, the boxes,
forearm visibility and hand type. Nothing existing was renamed or changed. Verified safe by checking
every consumer — `renderer.py:178-179`, `run_aria.py:97,127` and `run_app.py` all index by explicit
key, none iterate the dict or unpack positionally.

One line of new computation was added: `pred_hand_j2d_head`, using the repo's own
`get_j2d_from_kpt2d(..., pred_type='hand')` — the same helper already used for the arm. It is exact
for a rectified pinhole input; the crop-to-image map is linear, so for a distorted camera it is an
approximation, which the comment says.

**`demo/render_landmarks.py` — deduplicated.** It originally carried its own copies of the joint
layout, the drawing helpers and the video writer. Those now come from `fusion/topology.py` and
`fusion/video_io.py`, removing ~120 duplicated lines and, more importantly, removing the possibility
of the viewer and the fusion stages disagreeing about keypoint order.

**Nothing in `hand_labelling_21kp` was touched.** The issues found there (§11) are recorded, not
fixed — that is a separate change with its own review.

**`scripts/install.sh` was not modified.** mmpose is documented as a separate
`mim install "mmpose>=1.3.2"` rather than added to the shared installer, because mmpose/mmcv version
compatibility is a real risk that could not be tested here and would affect everyone using the repo,
including people not running this experiment.

---

## 8. Verification — what was actually checked

**`fusion/selftest.py`, 34 checks, all passing.** Runs on numpy + CPU torch + OpenCV; no GPU, no
checkpoints, no MANO files.

| Group | Checks | Examples of what is pinned |
| --- | --- | --- |
| topology | 5 | joint order re-derived from `mano_joint_mapping`; edge list byte-identical to the old repo's; projection against the closed form |
| metrics | 10 | planted gap of exactly 1.0 s recovered; stride-awareness (0.7 s, not 0.23 s); shallow fingertip inside a plausible wrist flagged; `reprojection_qc` ≈ 0 for consistent input and > 30 px for a `cx` off by 40 |
| fuse | 7 | bbox matching survives shuffled row order; wrist gate rejects beyond `--match-dist`; duplicate suppressed **and** two overlapping real hands both kept; backprojection inverts projection |
| refit | 7 | torch and numpy projections agree; projection differentiable with `du/dx = fx/z` checked numerically; Huber continuous at δ; 6D↔axis-angle round-trip; reject reasons partition the input exactly |
| calibration / video_io / config | 5 | all three `calibration.json` schemas; frame-selection arithmetic; drawing survives NaN and 1e9 coordinates; every YAML case names a known stage |

**Static analysis.** `ruff --select F821,F811,F401,F841` clean over `fusion/` and
`demo/render_landmarks.py` — in particular **zero undefined names**, which matters most for code that
has never been executed. All files byte-compile; the shell script passes `bash -n`.

**CLI construction.** All six entry points build their parsers and validate flags on a machine with
no pytorch3d. `run_egoforce_3d.py` needed a local fix for this: `camera_models/__init__.py` imports
the pytorch3d wrappers at package init, so the camera-model import was deferred into
`build_camera_model()`. Without that, even `--help` required a CUDA-only dependency.

**External facts checked against source, not memory.** The RTMPose config filename (a first search
result conflated it with the coco-wholebody-hand config; the actual directory listing settled it),
the checkpoint URL (HTTP 200, 55,287,475 bytes), the `bgr_to_rgb=True` preprocessor setting, the
`inference_topdown` signature and return structure, and the `coco_wholebody_hand` keypoint names.

**Two self-test failures on first run were both my test fixtures, not the code.** Constant-`y`
synthetic hands produced zero-height bboxes, so IoU was 0 by definition and nothing could ever be
suppressed; and one test bypassed `__init__` to avoid needing MANO files, then read an attribute
`__init__` sets. Both fixtures were fixed rather than the code weakened.

---

## 9. What remains unverified

A green self-test means *the plumbing and the arithmetic are right*. It does not mean the pipeline
produces good hands.

| Surface | Why untested | Risk |
| --- | --- | --- |
| EgoForce inference | Needs CUDA + TensorRT + weights | Never executed end to end |
| RTMPose inference | Needs mmpose + checkpoint | Call shape and BGR convention verified against source, never run |
| MANO forward + refit convergence | Needs the MANO pkl files | **No optimisation step has ever been run.** The objective is written and differentiable by construction; whether 80 Adam iterations at lr 0.02 converge is unknown |
| Refit weight defaults | No footage to tune on | Reasoned, not fitted. Expect to iterate — `plan_of_action.md` §5 lists what to watch |
| `--bbox-pad` | Untunable without running | Interacts with RTMPose's own `GetBBoxCenterScale` padding; EgoForce boxes may be tighter or looser than RTMPose expects |
| `download_rtmpose_hand5.sh` | Not executed | URL and config name verified; which of its two routes fires is unknown |
| Overlay encoders | Not exercised | ffmpeg h264 pipe with an OpenCV `mp4v` fallback |

---

## 10. Design decisions, and the alternatives rejected

| Decision | Alternative rejected | Reasoning |
| --- | --- | --- |
| Emit the old npz schemas exactly | A cleaner new format | Makes the old fusion, QC and stage 3/4 work unchanged, and `T2a` free. Cost: we inherit its vocabulary quirks (§11) |
| Reuse the old `source` labels | Honest new names like `egoforce+rtmpose` | `postprocess.py` branches on those exact strings. `producer` carries the truth |
| Don't reimplement rigid PnP | A local reimplementation for symmetry | The old one already accepts our files, and staying bit-identical is what makes `T2a` a valid control |
| Refit MANO parameters | Per-keypoint 3D optimisation | Moving keypoints independently makes bone lengths breathe and the output stops being a hand — the same reason the old pipeline chose rigid PnP over ray backprojection |
| Anchor translation to EgoForce | Let the 2D determine everything | 2D reprojection is scale–depth degenerate; without the anchor the fit slides along the ray |
| Boxes from the 3D npz by default | Always detect independently | Gives exact 1:1 pairing and guarantees the stride/size agreement the old fusion checks. `T2d` covers the other question |
| Recompute metrics in `compare_runs.py` | Read the per-case sidecars | The old fusion's sidecar has a different shape and no coverage/track metrics |
| Reimplement the old greedy tracker in `metrics.py` | Write a better tracker | Identity numbers must be comparable to what the old stage will report |
| Label proxies as proxies | Report them as accuracy | There is no ground truth. Presenting disagreement as error would be the most damaging possible outcome |
| Coarse rotation-magnitude cap | Per-DOF anatomical limits | MANO's per-joint axis-angle frames are not clean flexion axes; a real limit set needs separate derivation and validation |
| Leave `install.sh` alone | Add mmpose to it | mmpose/mmcv compatibility is untestable here and would affect everyone |
| Don't fix the other repo | Patch `mp_support` while we're in there | Separate change, separate review. Recorded in §11 |

---

## 11. Failure modes and where they surface

| Symptom | Where it shows | What it means / what to do |
| --- | --- | --- |
| `reproj_median_px` ≥ 1 px | `_3d_meta.json` | 3D and `K` describe different cameras. **Stop** — everything downstream is meaningless. Check `calib-block` and eye. |
| `cx` far from `W/2` warning | 3D stage stdout | Calibration may not match this video (right eye vs left). |
| High `wilor_pnpfail` count | `_fuse_stats.json` → `refit.reasons` | Read the per-reason breakdown before loosening gates. `rejected_bone_change` and `rejected_depth_change` mean the refit is deforming the hand; `rejected_no_improvement` means RTMPose is not adding information. |
| `bone_cv_med` grew vs `T1` | `comparison.csv` | The refit is buying reprojection accuracy by letting shape drift between frames. Raise `w_beta`, tighten `max_bone_change`. |
| Coverage dropped vs the old pipeline | `comparison.csv` `cov_*` | Check `source_mix` first — see the `mp_support` row below. |
| Many `dets_lost_short` | `comparison.csv` | Detections exist but fragment. The fix is association, not the pose model. Compare `T1c`. |
| `label_flips` > 0 | `comparison.csv` | Handedness changing mid-track. EgoForce's handedness is structural, so nonzero here suggests the tracker is merging two hands. |
| High `foot_lowframe` | `comparison.csv` | A **review queue**, not a foot count. Run the CLIP classifier or a human over it. |
| `depth_implausible_frac` high | `comparison.csv` | Systematic depth failure — wrong `K` or a changed convention — rather than a few bad frames. |

**The one that will bias `T2b` specifically.** `postprocess.py:138` computes
`mp_support = mean(source in ('fused', 'lifted_2d'))`, which **excludes `wilor_pnpfail`** — even
though `MEASURED_SOURCES` at `postprocess.py:34` includes it and those rows *were* 2D-matched. Our
fusion emits `wilor_pnpfail` for every rejected refit, so a track that is mostly refit-rejected reads
as uncorroborated and can be dropped as `wilor_only` under misleading single-model evidence. Check
`source_mix` before concluding anything from a coverage drop. Not fixable from this repo.

---

## 12. How to extend it

**Add a different 2D model.** Write a producer that emits the stage-1a schema (§6) — that is the whole
contract. If its keypoint order differs from OpenPose-21, remap it *in the producer* and say so in
`keypoint_order`; do not push the remap downstream.

**Add a metric.** Put it in `fusion/metrics.py`, returning a dict. If it is a proxy, say so in the
name or an `is_proxy` key, and document what it does *not* establish. Add a column to
`compare_runs.COLUMNS` and a synthetic check to `selftest.py` with a hand-computed expected value —
every existing metric has one.

**Add a test case.** Append to `cases:` in `testcases.yaml` with an `id`, a `question`, and `stages`
drawn from `{egoforce_3d, rtmpose_2d, fuse, fuse_rigid}`. Per-stage keys become CLI flags
(`hand_conf: 0.15` → `--hand-conf 0.15`; `no_kalman: true` → `--no-kalman`). The self-test validates
that every case names known stages.

**Wire up ground truth.** `metrics.fingertip_error(pred, gt)` already exists and is tested. It needs
`(N,21,3)` camera-frame metres row-aligned with the predictions. Once annotations exist, call it from
`compare_runs.row_for` and the proxy columns can be retired.

**Change the refit objective.** Everything is in `mano_refit.ManoRefiner._refit_chunk`. Keep the
depth anchor unless you have replaced it with another depth cue, and re-check `accept_refit`'s gates —
a new term can make a previously-rejected fit pass for the wrong reason.

---

## 13. Commands

```bash
python fusion/selftest.py                                              # 34 checks, no GPU
python fusion/run_testcases.py --config fusion/testcases.yaml --dry-run
python fusion/run_testcases.py --config fusion/testcases.yaml \
       --mono-pipeline /path/to/hand_labelling_21kp
python fusion/compare_runs.py --runs-root _DATA/runs --out _DATA/runs
python demo/render_landmarks.py --video <uncalibrated.mp4> --duration-seconds 10
```

Environment, weights and the GPU requirement: [`plan_of_action.md`](plan_of_action.md) §3.
