# Plan of action — EgoForce evaluation for the egocentric hand pipeline

**Last updated:** 2026-09-10
**Branch:** `feat/egoforce-rtmpose-fusion`
**Status:** all code written and linted; GPU-independent logic self-tested. **Nothing has been run
on real footage.** No GPU was available on the development machine (Apple M3, and EgoForce is
hard-pinned to CUDA 12.6 + TensorRT).

---

## 1. What we are trying to find out

Three separate defects were observed in the current WiLoR + MediaPipe pipeline, and they have three
separate causes. Conflating them is the main risk to this evaluation:

| Defect | Most likely cause | What a better pose model can do about it |
| --- | --- | --- |
| A foot tracked as a hand | **Detection.** The 21 keypoints cannot separate a foot from a hand — the existing `hand_shape_score` scores a synthetic foot *higher* than a real hand | Nothing. This needs detector fine-tuning with feet/shoes as negatives, or the CLIP image classifier |
| Hands missing from the output | **Association and rejection rules**, not recall alone. Detections exist but get discarded: tracks shorter than 10 detections are dropped (`postprocess.py:63`, `run_clip.py:177`) | Only indirectly, by producing more consistent detections that fragment less |
| Wrong finger / wrist geometry | **Pose estimation and fusion** | This is the part a model swap and an articulated fusion can actually fix |

So the experiments are scoped to the third row, and the metrics are instrumented to *show* the first
two rather than pretend to fix them.

### The experiments

| ID | Pipeline | Question it answers |
| --- | --- | --- |
| `T1` | EgoForce alone | Does swapping WiLoR for EgoForce improve difficult poses and camera-space placement at all? Baseline for everything else. |
| `T1b` | EgoForce, Kalman off | How much of EgoForce's apparent stability is the translation filter rather than the model? The filter also delays reacquisition, which shows up as recovery delay. |
| `T1c` | EgoForce, loosened detector + longer track survival | Are missing hands a detection problem or a pose problem? If coverage jumps here, stop working on the pose model. |
| `T2a` | EgoForce 3D + RTMPose 2D through the **existing rigid PnP** | Isolates the *model swap* from the *fusion change*. Without this row, a T2b win is uninterpretable. |
| `T2b` | EgoForce 3D + RTMPose 2D through the **articulated MANO refit** | The actual proposal. The only variant that can correct a wrongly bent finger. |
| `T2c` | `T2b` with weaker priors | Sensitivity check. If accuracy improves *and* bone-length spread stays flat, the default priors are too stiff. |
| `T2d` | `T2b` with an independent hand detector | Does detector choice change the foot false positives and recall? With shared boxes, a foot detected once is a foot in both streams. |

---

## 2. What is done

Everything is on `feat/egoforce-rtmpose-fusion`. See [`fusion/README.md`](fusion/README.md) for how
the stages connect.

| Component | File | Notes |
| --- | --- | --- |
| Landmark video/keypoint viewer | `demo/render_landmarks.py` | Standalone; AnyCalib intrinsics; for uncalibrated footage |
| EgoForce 3D producer | `fusion/run_egoforce_3d.py` | Drop-in for `run_wilor_3d.py`, identical npz schema |
| RTMPose 2D producer | `fusion/run_rtmpose_2d.py` | Drop-in for `run_mediapipe_2d.py`, identical npz schema |
| Articulated fusion | `fusion/fuse_egoforce_rtmpose.py`, `fusion/mano_refit.py` | Writes the mono-pipeline fused schema |
| Metrics | `fusion/metrics.py` | Coverage, gaps, recovery, fragmentation, identity flips, depth/reprojection QC, labelled proxies |
| Comparison table | `fusion/compare_runs.py` | Recomputes from the fused npz so rigid and articulated land on the same axes |
| Test matrix | `fusion/run_testcases.py`, `fusion/testcases.yaml` | `--dry-run` prints every command |
| Review overlay | `fusion/render_comparison.py` | grey = EgoForce raw, colour = written, dots = RTMPose observation |
| Self-test | `fusion/selftest.py` | **34/34 passing** |
| RTMPose fetch | `scripts/download_rtmpose_hand5.sh` | Verified config name + checkpoint URL |
| MANO params exposed | `demo/inference.py` | Purely additive keys; `run_app.py` / `run_aria.py` / `renderer.py` unaffected |

### Verified facts (checked, not assumed)

- **All three models share the same 21-keypoint order.** EgoForce's comes from `mano_joint_mapping`
  in `models/mano_layer.py:33` plus fingertips appended thumb→pinky; RTMPose Hand5 trains against
  mmpose's `coco_wholebody_hand.py`; MediaPipe matches. **No remapping anywhere.** `selftest.py`
  asserts the derivation *and* asserts our edge list is byte-identical to the existing
  `hand_topology.HAND_EDGES`, so the repos cannot drift silently.
- **RTMPose-m Hand5:** config `rtmpose-m_8xb256-210e_hand5-256x256.py`, checkpoint
  `rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth` (55 MB, URL returns 200),
  256×256 input, 21 channels, reported 96.4 PCK@0.2 / 83.9 AUC / 5.06 EPE.
- **It expects BGR** — its `data_preprocessor` sets `bgr_to_rgb=True`. The MediaPipe stage it
  replaces needs RGB. Easy silent-accuracy-loss bug; handled.
- **EgoForce needs no focal hack and no principal-point rebase.** Its ray-space solver consumes the
  real intrinsics (`core/rss.py::unproject_unit_rays`), so anisotropic `fx≠fy` is supported instead
  of being averaged to `f0`, and there is no image-centre assumption to undo.
- **Handedness is structural, not guessed** — EgoForce runs separate left/right crop streams, so
  `is_right` says which stream produced the row. This is the input the existing track-level
  geometric handedness voting was built to repair.

---

## 3. Pending — in execution order

### Step 1 — get a GPU box and install (blocking everything)

```bash
conda create -n egoforce python=3.10 -y && conda activate egoforce
bash scripts/install.sh                    # CUDA 12.6, torch 2.8, TensorRT, mmcv, pytorch3d, AnyCalib
bash scripts/download_model_weights.sh     # _DATA/: model_weights.pth, detectors, MANO
mim install "mmpose>=1.3.2"                # NOT in scripts/install.sh
bash scripts/download_rtmpose_hand5.sh
python fusion/selftest.py                  # must stay 34/34 in the real env
```

**GPU requirement.** An NVIDIA GPU with a CUDA 12.6-capable driver is mandatory:
`demo/inference.py` imports `torch_tensorrt` at module scope and compiles both the detector and HALO;
`scripts/install.sh` pins `torch==2.8.0+cu126` / `torch_tensorrt==2.8.0+cu126`. There is no macOS or
CPU path. Per-frame work is small and fixed (RTMDet-tiny + a YOLO pose detector on the full frame,
then HALO on four 224×224 crops), so VRAM is dominated by the TensorRT and inductor `max-autotune`
compile rather than inference. The repo documents no VRAM figure; **~12 GB should be comfortable, but
treat that as an estimate, not a measured requirement.** Expect a one-off compile before frame one.

The **articulated refit alone needs no GPU** — `fusion/mano_refit.py` is torch + smplx + the MANO
files. Once the producer npz files exist, refit weights can be re-tuned on a laptop.

### Step 2 — choose the clips and fill in `fusion/testcases.yaml`

The shipped `clips:` block is placeholders and the runner refuses to start until it is edited.
Intrinsics are **not** guessed: give `calib:` (a Standard Package `calibration.json`, read with the
same schema handling as `run_clip.read_K`) or `K: [fx, fy, cx, cy]`.

Pick clips where the **current pipeline visibly fails** — a comparison on clips that already work
tells you nothing. Wanted:

1. fast hand motion (tests recall and recovery delay)
2. a hand gripping a tool, seen from the back (the case MediaPipe misses and the shape gate deletes)
3. both hands crossing / two-handed manipulation (tests identity switches and duplicate suppression)
4. **the clip that produced the foot false positive**
5. a clip that currently produces missing output despite visible hands

Then:

```bash
python fusion/run_testcases.py --config fusion/testcases.yaml --dry-run
```

Read the printed commands before spending GPU time. Confirm the clip paths, the `K` source and the
frame windows are what you meant.

### Step 3 — smoke-run one clip, one case

```bash
python fusion/run_testcases.py --config fusion/testcases.yaml --case T1_egoforce_only --clip <name>
```

Check, in this order:

- `_3d_meta.json` → `reproj_median_px` **must be < 1 px**. If it is not, `kp3d_cam` and the stored
  `K` describe different cameras and every downstream number is meaningless.
- `_3d_meta.json` → `depth_Z_m.frac_below_5cm` should be ~0.
- `_3d_meta.json` → `head_vs_lift_median_px`. This is new information: how far EgoForce's own 2D
  keypoint head sits from its own 3D lift. A large value means the 3D lift is fighting the 2D head
  and is worth understanding before trusting the fusion.
- the `_3d_overlay.mp4` — does the skeleton sit on the hands at all?

### Step 4 — run the matrix and read the comparison

```bash
python fusion/run_testcases.py --config fusion/testcases.yaml \
    --mono-pipeline /path/to/hand_labelling_21kp
```

Produces `_DATA/runs/comparison.{csv,json,md}`. Read it in this order:

1. **`T1` vs the current pipeline's own numbers** — did the model swap help?
2. **`T2a` vs `T2b`** — did the *articulated* fusion help beyond the model swap? If `T2b` ≈ `T2a`,
   the refit is not earning its complexity and should be dropped.
3. **`T1c` vs `T1`** — if coverage jumps, the missing-hand work belongs in detection/association.
4. **`bone_cv_med` and `refit_depth_change_m` on `T2b`** — if these grew, the refit is buying
   reprojection accuracy by deforming the hand, and the gates need tightening.

### Step 5 — tune the refit (no GPU needed)

The defaults in `fusion/mano_refit.py::RefitWeights` are **reasoned but untuned**. Nothing has been
fitted to real footage. Expect to iterate:

| Knob | Default | What to watch |
| --- | --- | --- |
| `w_depth` | 250.0 | Load-bearing. 2D reprojection is scale–depth degenerate, so this is what keeps the fit metric. Lower it and depth will drift. |
| `w_pose` / `w_orient` | 4.0 / 8.0 | Too high → the refit cannot fix the bent finger (the whole point). Too low → implausible poses. `T2c` probes this. |
| `w_beta` | 2.0 | Guards bone lengths. Watch `bone_cv_med`. |
| `min_conf` | 0.30 | RTMPose landmarks below this contribute nothing. Needs calibrating against RTMPose's actual confidence distribution on *this* footage. |
| `max_bone_change` | 0.15 | Reject gate. If the reject rate is high, read the `refit.reasons` breakdown before loosening it. |
| `iters` / `lr` | 80 / 0.02 | Convergence is unverified. Check `residual_px` vs `residual_px_init` actually falls. |

### Step 6 — hand back to the existing pipeline

The fused npz is in the mono-pipeline's schema, so tracking/filtering/handedness/overlay re-run on
**CPU** with no changes to that repo:

```bash
python run_clip.py --from-npz <case>/<stem>_hand21_keypoints.npz \
    --video <clip> --out <dir> --min-len 6 --foot-filter
```

### Step 7 — only then, the things this evaluation cannot settle

- **Foot rejection: fine-tune the detector.** Feet, shoes and confusing objects as negatives, plus
  difficult real hands as positives. No keypoint model is a semantic foot filter. `T2d` only tells
  you whether the *choice* of detector matters.
- **Real fingertip error needs annotation.** `fusion/metrics.py::fingertip_error` is written and
  tested and takes a ground-truth array — it just has nothing to consume yet. Until then
  `tip_disagree_px` is cross-model *disagreement*, which ranks variants but is not accuracy.
- **Stereo (POEM-v2).** Out of scope here. Note EgoForce has **no stereo support at all** — for a
  rectified stereo pair you would run it per view and get two independent estimates.
- **Temporal model.** Only worth considering if jitter survives everything above.

---

## 4. Untested surfaces — what could still be wrong

`fusion/selftest.py` covers topology, the stride-aware coverage/gap arithmetic, duplicate
suppression, 2D/3D matching, projection maths, the Huber/weighted-median helpers and the
accept/reject accounting. It explicitly does **not** cover:

| Surface | Risk |
| --- | --- |
| EgoForce inference | Needs CUDA + TensorRT + weights. Never executed. |
| RTMPose inference | Needs mmpose + checkpoint. The `inference_topdown` call shape and BGR convention were verified against the mmpose source, but never run. |
| MANO forward + refit convergence | Needs the MANO pkl files. The objective is written and differentiable-by-construction, but **no optimisation has ever been stepped**. |
| `_2d_keypoints.npz` box padding | `--bbox-pad` interacts with RTMPose's own `GetBBoxCenterScale` padding. Untuned; EgoForce's hand boxes may be tighter or looser than what RTMPose expects. |
| `scripts/download_rtmpose_hand5.sh` | Two routes with a fallback; the URL and config name are verified but the script has not been executed. |
| `mim download` config self-containment | Route 1 tests whether the config loads and falls back to a sparse clone if not. Which route fires is unknown until run. |
| Overlay encoders | ffmpeg h264 pipe with an OpenCV `mp4v` fallback; neither exercised. |

---

## 5. Issues in `hand_labelling_21kp` that affect this comparison

Found by reading that code; **none are fixed here** (different repo, out of scope). They matter
because they can make a good model look bad.

| Issue | Location | Effect on the comparison |
| --- | --- | --- |
| `mp_support` excludes `wilor_pnpfail` even though those rows *were* 2D-matched | `postprocess.py:138` — note `MEASURED_SOURCES` at `postprocess.py:34` *does* include it | A track that is mostly refit-rejected reads as uncorroborated and can be dropped as `wilor_only` under misleading "single-model" evidence. Since our fusion emits `wilor_pnpfail` for rejected refits, **this directly penalises `T2b`.** Check the `source_mix` before concluding anything from a coverage drop. |
| Tracks shorter than `--min-len` (default 10) are discarded | `postprocess.py:63`, `run_clip.py:177` | Fragmentation becomes missing output. `compare_runs.py` reports `tracks_dropped_short` and `dets_lost_short` so you can see how much is lost this way before blaming the model. |
| The CLIP foot filter is **off by default** | `run_clip.py:186` | If the failing run did not pass `--foot-filter`, no hand-vs-foot classification ran at all. Worth confirming what the failing run actually used. |
| Foot rejection also requires the track to sit low in the frame | `postprocess.py:276`, threshold `run_clip.py:188` | It is not a general hand-vs-foot gate. A foot high in frame passes. |
| Rigid PnP cannot change articulation | `fuse_2d_3d.py:102` | This is the motivation for `T2b`, and `T2a` is what proves whether it matters. |

---

## 6. Decisions taken, so they are not re-litigated

- **Emit the mono-pipeline's exact npz schemas** rather than a new format. Cost: our stages inherit
  its vocabulary quirks (see §5). Benefit: `fuse_2d_3d.py`, `postprocess.py`, `render_overlay.py` and
  `run_clip.py --from-npz` all work unchanged, `T2a` is free, and the rigid baseline stays
  bit-identical to the historical one.
- **Reuse the mono-pipeline's `source` labels** (`fused` / `wilor` / `wilor_pnpfail` / `lifted_2d`)
  instead of honest new names, because `postprocess.py` branches on those exact strings. An additive
  `producer` array carries the real model names — `wilor` rows contain no WiLoR.
- **RTMPose boxes default to the EgoForce 3D npz** (`--boxes npz`). Gives exact 1:1 pairing per
  physical hand, so the fusion cannot mis-associate, and guarantees the `width`/`height`/`step`
  agreement that `fuse_2d_3d.py` checks. `T2d` covers the independent-detector question.
- **Did not reimplement rigid PnP.** The existing one already accepts our files.
- **Did not fix the `mp_support` bug** or anything else in the other repo. It is a separate change
  with its own review; recorded here instead.
- **Joint limits are a coarse magnitude cap**, not per-DOF anatomical ranges. MANO's per-joint
  axis-angle frames are not clean flexion axes, so a real limit set has to be derived and validated
  separately. The load-bearing plausibility constraint is the MANO parameterisation plus `w_pose`
  pulling toward EgoForce's already-plausible prediction. Do not read `w_limit` as anatomy.
- **`--hand-conf` reaches the YOLO hand detector only.** The RTMDet forearm score floor is hardcoded
  at 0.3 in `demo/inference.py::detect_bounding_boxes` and is not parameterised. If forearm recall
  turns out to matter, that is a small change to plumb through.

---

## 7. Quick reference

```bash
python fusion/selftest.py                                          # 34 checks, no GPU
python fusion/run_testcases.py --config fusion/testcases.yaml --dry-run
python fusion/compare_runs.py --runs-root _DATA/runs --out _DATA/runs
python demo/render_landmarks.py --video <uncalibrated.mp4> --duration-seconds 10
```
