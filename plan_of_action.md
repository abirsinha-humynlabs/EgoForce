# Plan of action — EgoForce evaluation for the egocentric hand pipeline

**Last updated:** 2026-09-17
**Branch:** `feat/egoforce-rtmpose-fusion`
**Status:** GPU environment stood up and **both run types have executed on real footage**
(episode_047, episode_002). EgoForce's geometry checks out; its *detection coverage* is the blocking
defect. `episode_048` additionally has HaWoR and MINT/ADAPT runs, compared in §3.3. The head-to-head
type 1 vs type 2 comparison the branches exist for has **not** been read yet — §4 item 4.

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

Everything is on `feat/egoforce-rtmpose-fusion`. [`fusion/README.md`](fusion/README.md) is the quick
orientation; [`PIPELINE.md`](PIPELINE.md) is the design and engineering document — what was built,
why it is shaped that way, the data contracts, and what was verified versus assumed.

### Code

| Component | File | Notes |
| --- | --- | --- |
| Landmark video/keypoint viewer | `demo/render_landmarks.py` | Standalone; AnyCalib intrinsics; uncalibrated footage |
| EgoForce 3D producer | `fusion/run_egoforce_3d.py` | Drop-in for `run_wilor_3d.py`, identical npz schema |
| RTMPose 2D producer | `fusion/run_rtmpose_2d.py` | Drop-in for `run_mediapipe_2d.py`, identical npz schema |
| Articulated fusion | `fusion/fuse_egoforce_rtmpose.py`, `fusion/mano_refit.py` | Writes the mono-pipeline fused schema |
| Three delivery metrics | `fusion/evaluate_run.py` | M1 coverage/recovery, M2 geometric validity, M3 fingertip agreement (proxy) |
| Diagnostics | `fusion/metrics.py` | Coverage, gaps, fragmentation, identity flips, depth/reprojection QC |
| Comparison table | `fusion/compare_runs.py` | Recomputes from the fused npz so rigid and articulated land on the same axes |
| Test matrix | `fusion/run_testcases.py`, `fusion/testcases.yaml` | `--dry-run` prints every command |
| Review overlays | `fusion/render_comparison.py`, `viz_delivery/render_v4_panels.py` | |
| Self-test | `fusion/selftest.py` | **34/34 passing**, no GPU needed |
| Run definitions | `run/` on `run/type1-egoforce-only` and `run/type2-fusion` | Each branch pins one run: script, manifest entry, README |

### Environment (done, 2026-09-11)

AWS `g5.xlarge`, Amazon Linux 2023, NVIDIA A10G 24 GB, driver **615.71.09** via `nvidia-open` from
NVIDIA's `cuda-amzn2023` repo. Three things cost most of that day and are worth not rediscovering:

- The EBS root volume ships at 20 GB and the conda env needs 30–45 GB. Grown to 80 GB.
- **AL2023 uses versioned kernel package names.** The running 6.12 kernel needs `kernel6.12-devel`;
  the generic `kernel-devel` tracks the 6.1 stream and conflicts with hundreds of lines of dnf noise.
  Do not pass `--allowerasing` — that installs mismatched headers and the dkms build produces a
  broken module.
- The instance role could not read AWS's S3 driver bucket, so the NVIDIA dnf repo was used instead.

### Runs completed

Delivered under `s3://…/labelling_results/hand_pose_EgoForce/` (the only prefix these scripts write):

| Clip | Runs |
| --- | --- |
| `episode_047` | type1 v1/v2/v3, type2 v1/v2 |
| `episode_002` | type1 v1/v2/v3, type2 v1/v2 |

A **stabilise stage** was added outside this repo between v1 and v2 (`_stabilise_stats.json`:
depth gate at z ≥ 0.05, temporal smooth, rigidify). It is not in `fusion/`; its outputs carry
`kp2d_raw` / `kp3d_cam_raw_pre` so the pre-stabilisation values remain recoverable.

Separately, `episode_048` has a **HaWoR** run and a **MINT/ADAPT** run produced by other work, which
§3 compares.

---

## 3. What the runs showed

### 3.1 EgoForce on episode_047 — the geometry is sound, the detection is not

From `type1_egoforce_only_v2`:

- **`reproj_median_px` = 2.6e-05.** The 3D and the stored `K` describe the same camera. The QC gate
  passes and nothing downstream is invalidated by a calibration mismatch.
- **`head_vs_lift_median_px` = 25.6 px.** EgoForce's own 2D keypoint head disagrees with its own 3D
  lift by 26 px at the median. The lift is the less trustworthy half.
- **Depth reached −0.419 m** — joints behind the camera. 11.5 % of joints sat below 5 cm. The v2
  depth gate dropped 355 rows, of which 174 were right-hand rows whose shallowest joint had a median
  of **−0.184 m**.

**Coverage is the headline defect.** Right hand: 764 of 1823 frames lost across 124 gaps — **42 % of
a 60.7 s clip**. Left hand: 343 frames (19 %). The right hand is the working hand (median wrist speed
28.8 px/frame vs 5.8 left) and it is the one that fails.

### 3.2 The drift at t ≈ 7 s is detection dropout, not filter lag

| Time | Frames | What happens |
| --- | --- | --- |
| 6.10–6.27 s | 183–188 | Right hand tracks cleanly, 57–67 px/frame, depth steady ~0.31 m |
| **6.27–6.63 s** | **189–198** | **333 ms blackout.** Reappears **336 px** away |
| 6.67–6.87 s | 199–206 | Tracks again; depth climbs 0.29 → 0.41 m |
| **6.87–7.23 s** | **207–216** | **Second 333 ms blackout.** Reappears **405 px** away |
| 7.23 s on | 217+ | Hand slows; tracking settles to 3–13 px/frame |

Confirmed as genuine **detector misses**, not the depth gate: the detector fired on **0 of 10** frames
in each blackout, and the gate removed nothing in that window.

Two mechanisms, both measured:

1. **The hand is leaving the frame.** At frame 188 the wrist sat at u = 1783 on a 1920-wide image,
   moving right at 57 px/frame — it exits in ~2.4 frames. Clip-wide, **47 % of the right hand's
   ≥5-frame gaps begin within 250 px of a side edge** (left hand: 8 %).
2. **Depth collapses with speed.** Median |ΔZ| per frame: 0.3 cm (<10 px/frame) → 0.6 → 1.0 →
   **9.8 cm (>100 px/frame)**. A 30× degradation; 10 cm in 33 ms is solver failure, not hand motion.

### 3.3 MINT/ADAPT vs HaWoR on episode_048

Full analysis and a rendered side-by-side video were produced locally — see §8.

| | MINT/ADAPT | HaWoR |
| --- | --- | --- |
| rows backed by image evidence | 3277 (**89.9 %**) | 3628 (**99.5 %**) |
| right hand, frames with evidence | 85.0 % | **99.3 %** |
| bone-length CV (median) | **0.0051** | 0.0274 |
| 2D acceleration median | **1.06 px** | 5.37 px |
| negative Z / below 5 cm | 0 % / 0 % | 0 % / 0 % |

**HaWoR wins evidence-backed coverage**, which is the one criterion post-processing cannot
manufacture. **MINT wins stability — but the comparison is not apples-to-apples**: MINT/ADAPT is
smoothed and rigidified, while HaWoR's metadata says `"postprocessing": "none - raw HaWoR output"`.
Those two rows measure the ADAPT stage, not the model. No claim about underlying keypoint accuracy is
supported by this data.

**Trap worth knowing:** the two npz files **do not share a row order** — only **44.8 %** of rows carry
the same handedness at the same index. Joining them by row position mismatches hands on more than half
the rows and inflates apparent disagreement ~4× (117 px vs the true 26 px). Key on `(frame, hand)`.

---

## 4. Next steps, in order

1. **Settle whether the t ≈ 7 s dropouts are the hand leaving frame or the detector failing on a
   visible hand.** Extract frames 188–200 from `left_eye.mp4` and look. Edge proximity is strong
   circumstantial evidence for the former but not proof, and it decides whether detector tuning is
   worth anything here. Cheap, and it gates item 2.
2. **Run `T1c`** (`--hand-conf 0.15 --max-misses 6`) on episode_047 and compare coverage against the
   existing type1. `max_misses = 2` cannot bridge a 10-frame gap by construction. If coverage jumps,
   the missing-hand work belongs in detection and association, not in the pose model.
3. **Chase the depth-at-speed failure.** Negative Z and 9.8 cm/frame depth jumps originate in the ray
   space solve, not the detector, and the stabilise stage only masks them by dropping rows.
4. **Compare type 1 against type 2 head-to-head.** Both have run, but `fusion/evaluate_run.py` has
   never been pointed at both with a shared `--rtmpose` reference, so M3 has not been read. This is
   the comparison the whole branch structure exists for, and it is one command.
5. **Re-do MINT vs HaWoR at the same processing level** — either put HaWoR through the stabilise
   stage, or compare raw MINT (`kp3d_cam_raw_pre`, already in its npz) against raw HaWoR.
6. **Tune the refit weights** (`fusion/mano_refit.py::RefitWeights`). Still untuned on real footage,
   and it needs no GPU once the npz files exist.
7. **Annotate ground truth** on a few difficult clips. `fusion/metrics.py::fingertip_error` is written
   and tested and has nothing to consume; until then every accuracy statement is a proxy.
8. **Foot false positives** — detector fine-tuning with feet and shoes as negatives. Unchanged from
   the original plan, and no keypoint model substitutes for it.


---

## 5. Untested surfaces — what could still be wrong

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

## 6. Issues in `hand_labelling_21kp` that affect this comparison

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

## 7. Decisions taken, so they are not re-litigated

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

## 8. Quick reference

### Local analysis artefacts (deliberately not committed)

`hand21kp/local_dump/` holds the episode_048 MINT-vs-HaWoR work: `comparison_report.md`,
`comparison.json`, `per_keypoint.csv`, the rendered `episode_048_MINT_vs_HaWoR_sidebyside.mp4`
(2560x774, 60.8 s), and re-runnable `compare_mint_vs_hawor.py` / `render_side_by_side.py`. It also
carries `inputs/` with both npz files and the 264 MB source video, which is why it is gitignored
rather than committed. Both scripts regenerate everything from `inputs/`.

```bash
python fusion/selftest.py                                          # 34 checks, no GPU
python fusion/run_testcases.py --config fusion/testcases.yaml --dry-run
python fusion/compare_runs.py --runs-root _DATA/runs --out _DATA/runs
python demo/render_landmarks.py --video <uncalibrated.mp4> --duration-seconds 10
```
