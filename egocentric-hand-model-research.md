# Hand Detection and Tracking for Rectified Egocentric Video

[Certain] A low aligned hand-pose error does not establish accurate wrist position, stable finger motion, or low drift in a moving-camera recording. These are different outputs and need different evidence. The research below supports a practical shortlist and an evaluation strategy; it does not establish a winner on a particular camera without testing its footage.

[Likely] **For calibrated stereo, start with POEM-v2 and a separate hand detector, then add temporal constraints. For monocular video, start with EgoForce for camera-space position and compare HaPTIC and HaWoR for sequence consistency.** Keep WiLoR as the common detection/reconstruction baseline. HandFlow is a particularly relevant recent experiment, but its initial release needs more integration work for two hands. A complementary 2D landmark model can improve the system when its observations are demonstrably better; averaging two 3D predictions is not a sound default.

[Certain] This report distinguishes four requirements:

| Output | Meaning | What it does not establish |
|---|---|---|
| 21 image keypoints | Pixel locations of wrist and finger landmarks | Metric depth |
| Root-relative 3D pose | Finger articulation and orientation relative to the wrist | Wrist trajectory or distance to the camera |
| Camera-space metric 3D | Joint coordinates including translation relative to the camera | Motion relative to the room when the camera moves |
| World-space metric 3D | Hand motion after applying a camera trajectory in a common metric frame | Immunity to camera-pose error |

[Likely] Until hardware and latency requirements are specified, the appropriate selection criterion is offline GPU accuracy. The order below prioritizes relevance, usable public implementations, geometric compatibility, and temporal behavior over headline speed or publication date.

[Certain] **The principal candidates have the following public implementation evidence.** “Weights documented” means that the official repository gives a checkpoint or download procedure; it does not mean a checkpoint was downloaded, loaded, or benchmarked here. A GitHub repository by itself is weaker evidence than an inference entry point and documented assets.

| Model and paper | Public GitHub | Input and output | Temporal behavior | Release evidence and principal integration issue |
|---|---|---|---|---|
| **POEM-v2**, TPAMI 2025; preprint 2024 | [JubSteven/POEM-v2](https://github.com/JubSteven/POEM-v2) | Calibrated multiview RGB; 21 3D joints and mesh in the reference-camera frame | Multiview reconstruction; add sequence tracking/smoothing | Training, evaluation, real-data inference, checkpoint links. Demo expects crops/masks and handedness; requires a camera adapter. [^1][^2] |
| **EgoForce**, SIGGRAPH 2026 | [dfki-av/EgoForce](https://github.com/dfki-av/EgoForce) | Monocular hand/forearm context plus camera geometry; 21 hand joints and camera-space mesh | Frame estimator plus translation Kalman filter | Video demo and evaluation; main model and detector assets listed on Hugging Face. Keep calibrated crop geometry and forearm context. [^3][^4] |
| **HaPTIC**, 2025 preprint | [JudyYe/haptic](https://github.com/JudyYe/haptic) | Monocular video, hand crops and full frames; MANO pose and trajectory | Learned temporal attention | Video inference, checkpoint download script, training code. Depth trajectory needs an initial offset; “global” is not automatically a SLAM-fixed room frame. [^5][^6] |
| **HaWoR**, CVPR 2025 | [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) | Monocular video; hand motion plus camera/world reconstruction | Learned temporal pose model and motion infiller | Video demo, checkpoints, evaluation. Training remains described as forthcoming; world mode adds SLAM and metric-depth dependencies. [^7][^8] |
| **HandFlow**, July 2026 preprint | [mxxu00/HandFlow](https://github.com/mxxu00/HandFlow) | Monocular video; temporal MANO parameters, optional ViPE world conversion | Generative sequence model with overlapping windows | V1 inference/visualization code; denoiser and normalization assets listed. No training release; left-hand mirroring and two-hand orchestration need attention. [^9][^10] |
| **WiLoR**, CVPR 2025 | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | Full RGB image; hand detections, MANO mesh and 21 joints | Per-frame; no temporal module | Detector and reconstruction checkpoints, image/demo inference. Metric translation and stable IDs require further treatment. [^11][^12] |
| **HaMeR**, CVPR 2024 | [geopavlakos/hamer](https://github.com/geopavlakos/hamer) | RGB hand crops; MANO reconstruction and joints | Per-frame | Training, evaluation, checkpoints and demos. Detector/crop quality and camera translation remain separate concerns. [^13] |
| **UmeTrack**, SIGGRAPH Asia 2022 | [facebookresearch/UmeTrack](https://github.com/facebookresearch/UmeTrack) | One/multiple calibrated headset views; articulated hand model and 21 landmarks | Native multiframe/multiview tracker | Pretrained inference and examples. Different landmark anatomy; supplied evaluation uses ground-truth crop generation. [^14][^15][^16] |
| **RTMPose-m Hand5**, 2023 | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | Hand crop; 21 2D keypoints | Per-frame; video processing is not a learned hand-motion model | Dedicated hand detector/landmark demo and checkpoint. Useful independent 2D branch; needs egocentric validation. [^17][^18] |
| **MediaPipe Hand Landmarker** | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | RGB; 21 image landmarks and hand-centered 3D landmarks | Tracking-assisted crop reuse and redetection | Packaged model and video API. Its “world landmarks” have origin at the hand center, not the room or camera. [^19] |

[Likely] **POEM-v2 is my first stereo reconstruction experiment, not a claim of proven superiority on a head-mounted two-camera rig.** It explicitly handles camera geometry and avoids discarding the second view. Prefer v2 to the original POEM: the paper documents poor camera-configuration transfer in v1 and introduces v2 to address it. The inspected demo returns `pred_joints_3d`, transforms coordinates, and processes known hand-side information; it is a reconstruction component, not a complete detector and identity tracker. [^1][^2][^20]

[Likely] Build the stereo adapter before judging the model. Supply synchronized views of the same hand, each crop’s updated intrinsics, and consistent extrinsics. Run a hand detector on both full images, associate left/right-camera detections, and maintain separate anatomical left/right tracks. Evaluate the `medium`, `large`, and MANO-parametric variants where practical; the largest model is not automatically best after domain shift. Temporal fitting is still required if frame-level outputs wobble.

[Certain] **EgoForce is unusually well matched to monocular egocentric camera-space estimation.** It combines hand/forearm evidence, calibrated crop information, 21 hand heatmaps, and a ray-based placement solver. Its translation filter is part of the reported method. The repository includes pinhole evaluation modes and a video demo, so rectified inputs are within its documented scope. [^3][^21]

[Likely] Benchmark its raw and filtered outputs separately. Preserve enough of the image to supply the forearm branch, but allow missing-arm handling when the arm is genuinely outside view. On already rectified footage, supply the rectified pinhole model rather than the source fisheye calibration. Treat the predicted metric scale as an estimate learned from visual/anatomical evidence, not as a stereo measurement.

[Certain] **HaPTIC is a strong temporal comparison with a subtle coordinate limitation.** It predicts camera-space trajectories and depth changes relative to an initial depth. Its “cross-view” attention aggregates video frames, rather than proving native stereo inference. Its trajectory metrics include sequence or first-frame alignment, so they do not directly measure unaligned absolute camera-space placement. Public code includes both inference and training. [^5][^6]

[Likely] Use HaPTIC when coherent motion and reproducible fine-tuning matter. For a room-fixed trajectory, pair camera-space output with independently validated camera motion. Keep the initial-depth error visible in evaluation instead of removing it with an alignment and describing the result as absolute accuracy.

[Certain] **HaWoR directly addresses moving-camera egocentric sequences.** It combines a temporal hand estimator, camera trajectory estimation, and motion completion. The release provides camera/world video demos, detector and model weights, and HOT3D evaluation. Its documented world pipeline depends on modified DROID-SLAM and Metric3D. [^7][^8]

[Likely] Compare its camera-space hand estimates before comparing its complete world pipeline. Otherwise, a better camera trajectory could be mistaken for a better finger estimator. If reliable stereo/VIO camera poses already exist, using those poses is a reasonable adaptation, but it changes the published pipeline and must be evaluated as such. Completed out-of-view poses should be stored as inferred, not observed.

[Certain] **HandFlow has moved beyond a paper-only announcement.** Its current official model listing includes `handflow_denoiser.pt` and `normalization_stats.npz`. The GitHub release describes inference and visualization only, a right-hand-trained model, explicit intrinsics, and optional ViPE camera motion. The supplied left-hand instruction requires mirroring. [^9][^10]

[Likely] It deserves a serious experiment for offline temporal quality, after the initial baselines work. Verify left-hand unmirroring, intrinsics transformation, independent tracking of both hands, missing-detection behavior, reproducibility across random seeds, and export of numerical joints. A smooth generative completion during occlusion is a hypothesis about motion. It should not silently become a ground-truth label or a measured trajectory.

[Certain] **WiLoR is a useful common baseline and detector source.** Its paper reports better video coherence than the compared image estimators despite having no temporal module. The authors’ dynamic evaluation includes displacement between consecutive predictions as well as jerk and translation measures. Low displacement alone cannot distinguish accurate stability from suppressed real motion. The official MANO wrapper explicitly appends fingertip landmarks and reorders the joint output. [^11][^12][^22]

[Likely] Use the full official WiLoR model first for an accuracy study. Treat its detector and reconstruction network as separate ablation choices. Compare whether failures come from missing hands, unstable crop boxes, incorrect articulation, or translation. HaMeR remains a valuable independent baseline, especially when testing whether a more recent model actually improves the relevant failure cases.

[Certain] **UmeTrack needs two adaptations before it satisfies the usual 21-keypoint contract.** Its source defines 21 landmarks including a palm center, fingertip-first ordering, and only two non-tip thumb landmarks. This differs anatomically from the common wrist plus four landmarks per finger. A permutation alone is insufficient. Its unknown-skeleton evaluation also uses ground-truth hand poses to generate crops in both calibration and tracking passes. [^15][^16]

[Likely] Use UmeTrack if investing in a native egocentric tracker, including crop initialization/recovery and landmark conversion, is acceptable. Derive any missing target landmark from its articulated model or fit a target hand representation, then quantify the conversion error. The original checkpoint’s camera/data domain matters; the HOT3D paper’s retrained model should not be assumed to be the same model distributed by the original repository.

[Likely] **RTMPose Hand5 is the most practical independent 2D branch to add to these experiments.** It is hand-specific, exposes landmark predictions, and has an official detector-plus-video example. Keep MediaPipe as a lightweight baseline. Neither should be promoted to the accuracy winner without a representative egocentric test, and whole-body pose models should not be the only way to locate hands when the wearer’s body is absent from the frame. [^17][^18][^19]

[Certain] **The following quantitative results are useful only within their own protocols.** The tables deliberately avoid a single cross-paper leaderboard.

| HOT3D paper, UmeTrack experiment | UmeTrack test MKPE, mm | HOT3D-Quest3 test MKPE, mm |
|---|---:|---:|
| UmeTrack-only training, one view | 13.6 | 24.2 |
| UmeTrack-only training, two views | 9.7 | 25.6 |
| Mixed UmeTrack + HOT3D training, one view | 13.4 | 15.4 |
| Mixed UmeTrack + HOT3D training, two views | 9.5 | 10.9 |

[Certain] This experiment supplies ground-truth hand shape and visible-hand boxes. Mixed-data stereo reduces HOT3D error by **29.2%**, calculated as `(15.4 − 10.9) / 15.4`; the paper’s “41% improvement” uses a different presentation of the ratio. More importantly, adding a second view to the UmeTrack-only checkpoint worsens its HOT3D result. That is direct evidence that stereo geometry does not remove training-domain mismatch. [^23]

| EgoForce paper, monocular evaluation | HOT3D camera-space error, mm | HOT3D aligned error, mm | H2O camera-space error, mm |
|---|---:|---:|---:|
| HandDGP baseline | 61.3 | 8.6 | 29.9 |
| EgoForce | 43.9 | 6.6 | 25.0 |

[Certain] These are the paper’s CS-MJE and PS-MJE results. The HandDGP comparison uses the authors’ reimplementation/retraining protocol, and HOT3D uses a custom split. EgoForce also reports about **14 FPS for the complete two-hand pipeline on an RTX 3090**. None of these results predicts accuracy or speed on another rig. [^21]

| HandFlow paper, HOT3D comparison | W-MPJPE, mm | WA-MPJPE, mm | Acceleration error, m/s² |
|---|---:|---:|---:|
| Dyn-HaMR | 69.11 | 31.01 | 4.77 |
| HaWoR | 73.88 | 27.37 | 11.57 |
| HandFlow | 43.00 | 16.17 | 4.18 |

[Certain] These are author-reported values on the paper’s validation protocol, with differing SLAM pipelines and some baseline values drawn from earlier work. Its runtime table reports **3.19 seconds reconstruction plus 16.21 seconds SLAM for 150 frames on an H100 80 GB**: about 47 reconstruction FPS, but **7.7 FPS including those two stages**. A quoted reconstruction rate is not an end-to-end deployment rate. [^24]

[Likely] These results justify experiments; they do not establish HandFlow as the universal low-drift winner or prove stereo UmeTrack beats monocular EgoForce on the same data. Compare identical frames, camera models, detectors, visibility rules, and alignment conventions. A 6 mm aligned result and a 44 mm absolute result can both correctly describe the same estimator.

[Certain] **Other public models are relevant for narrower experiments.** They should not all be added to the production pipeline.

| Model | Repository | When it is useful | Why it is secondary here |
|---|---|---|---|
| WildHands, ECCV 2024 | [ap229997/hands](https://github.com/ap229997/hands) | Egocentric training and crop/camera-aware baselines | Demo branch warns that its ViTPose-derived crops can fail without error handling. [^25] |
| Hamba, NeurIPS 2024 | [humansensinglab/Hamba](https://github.com/humansensinglab/Hamba) | Alternative single-image hand reconstruction | Released weights; Mamba/custom-kernel installation; single-image model, not a temporal tracker. [^26] |
| HandDGP, ECCV 2024 | [nianticlabs/HandDGP](https://github.com/nianticlabs/HandDGP) | Camera-space placement baseline with known intrinsics | Official weights are FreiHAND-trained; egocentric transfer must be measured. [^27] |
| Dyn-HaMR, CVPR 2025 | [ZhengdiYu/Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) | Offline interaction-aware optimization, including supplied camera/pose initialization | Heavier multi-stage pipeline; motion priors may fill unobserved movement. [^28] |
| HandOccNet, CVPR 2022 | [namepllet/HandOccNet](https://github.com/namepllet/HandOccNet) | Occlusion-focused single-image comparison | Older reconstruction baseline; does not by itself solve identity or camera drift. [^29] |
| HOPformer, ECCV 2026 according to release | [Sid2697/HOPformer](https://github.com/Sid2697/HOPformer) | Joint hand-object/contact estimation | Broader task and dataset-specific setup; checkpoint access is gated. Requires a 21-joint MANO configuration change. [^30] |

[Certain] UniHand’s ICLR 2026 motion-modeling paper and UST-Hand’s CVPR 2026 paper are relevant research leads, but a matching usable official GitHub implementation was not established in this review. HandOS’s inspected project page also did not expose a usable implementation. They are excluded from the runnable shortlist under the GitHub requirement. The repository `IRMVLab/UniHand` is a different work on forecasting end-effector trajectories, not the Sun et al. model. [^31][^32][^33][^34]

[Likely] **For stereo, the strongest engineering direction is geometry plus a hand prior plus temporal estimation.** I would compare two concrete systems:

| Component | System A: multiview model | System B: complementary 2D/3D fusion |
|---|---|---|
| Full-frame hand localization | WiLoR detector or an egocentric-finetuned hand detector | Same detector, to make the first comparison fair |
| Hand reconstruction | POEM-v2 from both calibrated views | WiLoR or EgoForce as a 3D articulation/shape prior |
| Image measurements | Native predictions; optionally add validated independent 2D landmarks | RTMPose Hand5 landmarks from each view |
| Metric placement | Multiview reconstruction, checked against reprojection | Weighted geometric fitting to both views, with triangulated anchors where reliable |
| Identity | Track association across time and cameras | Same association module |
| Temporal stability | Robust kinematic fitting or root/articulation filtering | Same temporal module |
| World motion if needed | Validated stereo/VIO camera trajectory | Same camera trajectory |

[Likely] **System B is the justified two-model fusion experiment:** one model supplies image-localized keypoints; the other supplies a plausible connected hand. Their roles differ. Use view visibility, calibrated landmark uncertainty, reprojection residuals, and temporal consistency to control their weights. If the 2D model is wrong on an occluded fingertip, it must not drag a reasonable 3D estimate toward that error. Conversely, a plausible mesh should not override clearly observed fingertip positions merely to remain near the learned mean pose.

[Certain] With exactly two views, a wrong correspondence can still produce a plausible 3D point; there is no third-view vote. Rectification narrows correspondence search but does not establish anatomical identity. [Likely] Reject implausible disparity, negative depth, large vertical mismatch, inconsistent handedness, and poor reprojection. Use neighboring joint structure and track history for additional evidence. When one view loses a joint, preserve a lower-confidence prior-based estimate or mark it missing rather than treating both views as valid observations.

[Likely] A useful offline objective is to optimize each track’s root translation, orientation, finger pose, and approximately constant shape over a short overlapping window:

```text
objective = robust reprojection error in all valid views
          + weighted distance from the learned hand-pose prior
          + anatomical constraints
          + motion-adaptive temporal regularization
          + optional verified surface-depth/contact constraints
```

[Likely] This is a proposed architecture, not a released model combination with demonstrated performance. Initialize it from POEM-v2 or the monocular estimator. Start with high-confidence visible measurements, introduce outlier rejection, and tune on held-out motions. Keeping shape stable across a track can reduce scale breathing; forcing the wrist or every finger to remain stationary can erase actual movement. MANO is the parametric hand representation used by many estimators, not another detector to ensemble with them. [^22]

[Likely] For monocular video, first compare EgoForce, WiLoR, HaPTIC, and HaWoR independently. Add RTMPose observations only after measuring where they improve visible-joint localization. A custom “EgoForce into HaPTIC” or “WiLoR into HandFlow” feature substitution generally requires architectural adaptation or training; matching a 21-joint output shape does not make neural features interchangeable. Where scale is unconstrained, temporal smoothness cannot make absolute depth observable.

[Certain] **Dense stereo and point tracking have supporting roles.** [FoundationStereo](https://github.com/NVlabs/FoundationStereo) estimates stereo correspondence; [CoTracker](https://github.com/facebookresearch/co-tracker) follows queried image points. Neither directly defines the required anatomical 21-joint hand skeleton. [^35][^36]

[Likely] Use dense stereo to check visible hand-surface depth or constrain a mesh. Sampling its depth at a joint’s 2D projection can return skin, the occluding object, or background; an internal skeletal joint is not necessarily on that visible surface. Use CoTracker only as a bounded propagation/consistency experiment with fresh hand observations and recovery. A tracked skin pixel is not permanently the same projected anatomical joint as fingers rotate or occlude.

[Certain] **Rectified video requires rectified geometry.** OpenCV’s stereo calibration/rectification machinery produces projection matrices and rectification rotations; resizing images changes their intrinsics. Triangulated coordinates belong to the coordinate frame encoded by those matrices. [^37]

[Likely] Apply the following input contract to every model adapter:

1. Keep original capture timestamps and verify actual left/right synchronization. Equal FPS and frame counts are insufficient.
2. Use the virtual pinhole intrinsics for the rectified images. Do not use original fisheye intrinsics or undistort already rectified pixels again with the original distortion coefficients.
3. Retain `P_left`, `P_right`, rectification rotations, baseline units, image dimensions, and valid-image masks. Record whether extrinsics map camera to world or world to camera.
4. Map all crop predictions back to full-image pixels before stereo matching, or transform each projection matrix consistently into its crop coordinates.
5. When translating outputs to the original camera/body/world frame, include the inverse rectification rotation and the calibrated camera-to-body transform.
6. Exclude black borders and unsupported image regions from detection confidence and geometry. Test peripheral hands explicitly, where rectification can stretch and blur fingers.
7. Transform handedness, keypoints, camera geometry, and output orientation together when mirroring left-hand crops. Mirroring the image alone is insufficient.

[Certain] For an ideal horizontal stereo pair with equal principal points, depth is `Z = f B / d`. First-order uncertainty is `sigma_Z ≈ Z² sigma_d / (f B)`. If principal points differ, use corrected disparity or the supplied projection matrices. The following is a calculation, not a camera benchmark:

| Assumed focal length | Baseline | Disparity uncertainty | Hand depth | Approximate depth uncertainty |
|---:|---:|---:|---:|---:|
| 700 px | 60 mm | 1 px | 0.35 m | 2.9 mm |
| 700 px | 60 mm | 1 px | 0.50 m | 6.0 mm |
| 700 px | 60 mm | 1 px | 0.80 m | 15.2 mm |

[Certain] Here 1 px is the uncertainty in disparity, not independently in each camera’s landmark. Calibration, synchronization, model bias, and occlusion add further error. [Likely] This is why sharper hand crops and better per-view landmark localization can matter as much as adding another reconstruction network.

[Likely] **Treat drift as several failure modes, with a specific response for each.** Calling every visible fluctuation “tracking noise” would hide the actual problem.

| Failure | Diagnostic evidence | Appropriate response |
|---|---|---|
| Crop drift | Hand drifts toward box edge; landmark errors follow crop changes | Fresh full-frame detection, crop margin, recovery logic |
| Landmark jitter | Error fluctuates around labeled/externally verified motion | Robust observations and adaptive temporal filtering |
| Depth/scale breathing | Hand size or camera-space depth varies inconsistently with measurements | Stereo geometry, stable hand shape, accurate intrinsics |
| Identity switch | Left/right identity changes at crossing or reappearance | Anatomical handedness plus temporal/cross-view association |
| Camera-pose drift | Multiple static references and hand world trajectories move together | Improve camera tracking and scale consistency |
| Occlusion hallucination | Smooth joints continue with no image support | Mark inferred spans; measure reappearance error separately |
| Excessive smoothing | Delayed contact, attenuated flexion, flattened velocity peaks | Reduce regularization; preserve fast articulation |

[Likely] Maintain track states such as observed, temporarily predicted, lost, and reacquired. Use predicted wrist position, box overlap, handedness, pose similarity, and cross-view agreement for association; wrist proximity alone is fragile when hands cross. Reset or reinitialize after long gaps and camera cuts. Bounded prediction can bridge short gaps, but prediction duration and uncertainty should be explicit.

[Likely] Filter root motion and local articulation separately. An adaptive Kalman filter with an offline Rauch–Tung–Striebel pass is a useful baseline for continuous valid intervals; [FilterPy](https://github.com/rlabbe/filterpy) provides filtering/smoothing components. [One Euro Filter](https://github.com/casiez/OneEuroFilter) is a useful low-latency comparison. Neither implementation understands hand anatomy automatically, and neither is established as optimal for this footage. [^38][^39]

[Likely] Reject large outliers before smoothing. Treat occluded and depth-untrusted values as missing or high-uncertainty measurements. Handle rotations on their rotation manifold. For offline processing, avoid smoothing across cuts, hand identity changes, or long tracking losses. Short-gap interpolation can be delivered as a separate inferred stream. Verify that numerical joint exports, wrist trajectories, and downstream velocities use the intended output, not merely a smoothed overlay.

[Certain] In a head-mounted recording, camera-frame hand movement includes the effect of camera movement. With a consistent metric camera-to-world transform, the geometric conversion is `X_world(t) = T_world_from_camera(t) X_camera(t)`. [Likely] Assess stationary-hand behavior in a frame appropriate to the task, with evidence that the hand is stationary. Otherwise, the filter may suppress genuine hand or camera-relative motion. A world transform from drifting SLAM can contaminate an otherwise accurate hand reconstruction.

[Likely] **The smallest convincing evaluation should include difficult continuous sequences, not only selected clean frames.** Start with roughly 20–30 clips spanning different hands, distances, objects, lighting, peripheral positions, and head movement. Label several hundred carefully selected visible frames for initial ranking, plus short contiguous windows for temporal analysis; expand if rankings vary substantially by condition. Keep a held-out subject/session split, and keep all frames from a stereo pair and nearby sequence windows in the same split.

[Likely] Include stationary hand with moving head, moving fingers with stationary wrist, fast reaches, grasp/contact transitions, two hands crossing, partial out-of-frame hands, prolonged object occlusion, blur, and dark/low-texture scenes. If only 2D labels are available, conclude only about 2D accuracy and geometric consistency. For absolute 3D claims, acquire independent reference measurements, such as calibrated additional views or motion capture; the model’s own stereo estimates are not ground truth.

[Likely] Report these outputs together:

| Requirement | Measurements |
|---|---|
| Detection | Recall, false positives, hand-presence accuracy, missed-hand duration |
| 2D joints | Pixel error, PCK curves, visible/occluded split, fingertip errors, p50/p95 |
| Metric 3D | Unaligned camera-space MPJPE, wrist translation error, root-relative MPJPE |
| Articulation | Per-finger error, joint-angle error, bone-length variation |
| Temporal stability | Position error on verified stationary intervals; acceleration error against reference; lag and peak attenuation |
| Stereo | Reprojection and epipolar residuals, disagreement by distance, invalid-depth rate |
| Tracking | ID switches, longest missing interval, recovery delay and reappearance discontinuity |
| World trajectory | Camera error and wrist trajectory error separately, with alignment and scale conventions disclosed |
| Runtime | Both hands, both views, detector, preprocessing, reconstruction, camera tracking, export; peak memory |

[Likely] Use this ablation order: raw single-model output; geometry correction; temporal estimator; independent 2D observations; learned temporal model. First hold the detector and camera poses fixed to isolate the hand estimator. Then evaluate complete native pipelines to measure deployable behavior. Compare raw, cleaned, and smoothed streams; lower acceleration alone is not success if contact timing or fingertip accuracy deteriorates. Add occlusion inference only as an explicitly scored capability.

[Likely] Store at least timestamps, anatomical hand ID, explicit joint names/order, 2D coordinates per view, camera-frame 3D coordinates, optional world coordinates, per-joint validity/visibility, source of depth, camera calibration version, and smoothing metadata. Preserve detector confidence separately from landmark uncertainty. A MediaPipe handedness score is not a per-keypoint localization confidence. [^19]

[Likely] Define the target skeleton before implementing adapters: wrist, then thumb CMC/MCP/IP/tip, then MCP/PIP/DIP/tip for index, middle, ring, and little fingers. Verify each model's anatomical definitions as well as index order; a tensor with 21 entries is insufficient. For MANO-based sequence releases, explicitly export joints through a documented regressor/fingertip mapping instead of assuming the demo already writes the required keypoint file.

[Likely] Calibrate fusion weights on held-out labels. Two networks trained on overlapping datasets can make correlated mistakes, so agreement does not by itself establish correctness. Compare error distributions at each confidence level, especially for fingertips, occlusions, and peripheral crops; keep a fusion only if it improves those measurements without increasing lag or missed-hand duration.

[Certain] Public availability and use terms vary: POEM-v2 specifies non-commercial scientific research; EgoForce and UmeTrack state non-commercial terms; WiLoR and HaWoR state CC-BY-NC-ND terms for models. Their MANO/dependency terms also apply. HaPTIC’s inspected repository did not present a root license, so its use terms were not established here. HandFlow’s MIT label does not replace the terms of its required components. These are repository statements, not a legal interpretation. [^1][^3][^9][^11][^14][^7][^6]

[Likely] **The recommended implementation order is WiLoR + RTMPose as a measurable baseline, POEM-v2 for calibrated stereo, and EgoForce versus HaPTIC/HaWoR for monocular footage.** Add HandFlow as the recent temporal challenger. Invest in UmeTrack if its native tracking design is worth the camera-domain, crop-recovery, and joint-convention work. Choose the final system by visible fingertip error, absolute wrist accuracy, missed-hand recovery, and motion-preserving stability on held-out clips.

[Certain] Research scope: primary papers, official project pages, public GitHub documentation/source excerpts, and selected checkpoint listings were reviewed through **10 September 2026**. No candidate was installed or benchmarked on target footage. GPU, frame rate, baseline, synchronization quality, post-rectification calibration, and accuracy tolerance remain unspecified. Consequently, deployment rankings and fusion benefits above are explicitly engineering recommendations rather than verified target-camera results.

**Sources**

[^1]: Yang et al. [POEM-v2 official repository](https://github.com/JubSteven/POEM-v2), release branch; TPAMI 2025 implementation and usage documentation.
[^2]: POEM-v2 authors. [`tool/infer_hand.py`](https://github.com/JubSteven/POEM-v2/blob/release/tool/infer_hand.py), public inference adapter, crop geometry, handedness and joint export.
[^3]: Millerdurai et al. [EgoForce official repository](https://github.com/dfki-av/EgoForce), SIGGRAPH 2026; inference, camera variants, Kalman option and license.
[^4]: EgoForce authors. [Official checkpoint inventory](https://huggingface.co/chris10/EgoForce/tree/main/_DATA), main model and detector assets.
[^5]: Ye et al. [Predicting 4D Hand Trajectory from Monocular Videos](https://arxiv.org/html/2501.08329v1), 2025 preprint; trajectory parameterization, temporal attention and evaluation alignment.
[^6]: Ye et al. [HaPTIC official repository](https://github.com/JudyYe/haptic) and [model download script](https://github.com/JudyYe/haptic/blob/main/scripts/dl_model.sh), current public release.
[^7]: Zhang et al. [HaWoR official repository](https://github.com/ThunderVVV/HaWoR), CVPR 2025; inference/evaluation instructions, assets and training release status.
[^8]: Zhang et al. [HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.pdf), CVPR 2025.
[^9]: Xu et al. [HandFlow official repository](https://github.com/mxxu00/HandFlow), V1 release, 2026; inference scope, intrinsics, handedness and ViPE dependencies.
[^10]: HandFlow authors. [Official checkpoint inventory](https://huggingface.co/mxxu00/HandFlow/tree/main), denoiser and normalization statistics.
[^11]: Potamias et al. [WiLoR official repository](https://github.com/rolpotamias/WiLoR), CVPR 2025; detector/reconstruction assets and release/use terms.
[^12]: Potamias et al. [WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](https://arxiv.org/html/2409.12259v2), Sections 5.3 and 11, 2025 version.
[^13]: Pavlakos et al. [HaMeR: Reconstructing Hands in 3D with Transformers](https://github.com/geopavlakos/hamer), CVPR 2024 official implementation.
[^14]: Han et al. [UmeTrack official repository](https://github.com/facebookresearch/UmeTrack), SIGGRAPH Asia 2022; pretrained inference and known/unknown-skeleton evaluation.
[^15]: UmeTrack authors. [`lib/common/hand.py`](https://github.com/facebookresearch/UmeTrack/blob/main/lib/common/hand.py), landmark names and counts.
[^16]: UmeTrack authors. [`run_eval_unknown_skeleton.py`](https://github.com/facebookresearch/UmeTrack/blob/main/run_eval_unknown_skeleton.py), crop generation and calibration/evaluation logic.
[^17]: OpenMMLab. [MMPose hand video and inference demos](https://mmpose.readthedocs.io/en/latest/demos.html), dedicated RTMDet/RTMPose hand pipeline.
[^18]: OpenMMLab. [RTMPose Hand5 model documentation](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose_hand5.md), training datasets and checkpoint.
[^19]: Google. [MediaPipe Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), updated August 2026; tracking configuration and coordinate/confidence semantics.
[^20]: Yang et al. [Multi-view Hand Reconstruction with a Point-Embedded Transformer](https://arxiv.org/html/2408.10581v2), POEM-v2 paper; 21-joint output and camera-configuration generalization.
[^21]: Millerdurai et al. [EgoForce paper](https://arxiv.org/html/2605.12498v1), SIGGRAPH 2026; Tables 1–2, runtime, evaluation splits, 21-joint decoder and translation filtering.
[^22]: WiLoR authors. [`wilor/models/mano_wrapper.py`](https://github.com/rolpotamias/WiLoR/blob/main/wilor/models/mano_wrapper.py), fingertip extension and joint mapping.
[^23]: Banerjee et al. [HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](https://arxiv.org/html/2411.19167v2), CVPR 2025, Section 4.1 and Table 2.
[^24]: Xu et al. [HandFlow: Fully Generative 4D Hand Recovery with Flow Matching](https://arxiv.org/html/2607.11221v1), July 2026, Tables 2–3 and evaluation protocol.
[^25]: Prakash et al. [WildHands official repository](https://github.com/ap229997/hands), ECCV 2024, and [demo branch](https://github.com/ap229997/hands/tree/demo).
[^26]: Dong et al. [Hamba official repository](https://github.com/humansensinglab/Hamba), NeurIPS 2024; inference, checkpoints and dependencies.
[^27]: Valassakis and Garcia-Hernando. [HandDGP official repository](https://github.com/nianticlabs/HandDGP), ECCV 2024; FreiHAND checkpoint and evaluation.
[^28]: Yu et al. [Dyn-HaMR official repository](https://github.com/ZhengdiYu/Dyn-HaMR), CVPR 2025; custom camera/pose initialization and optimization.
[^29]: Park et al. [HandOccNet official repository](https://github.com/namepllet/HandOccNet), CVPR 2022.
[^30]: Bansal et al. [HOPformer official repository](https://github.com/Sid2697/HOPformer), ECCV 2026 according to the authors; checkpoint access, hand-object task and 21-joint setup.
[^31]: Sun et al. [UniHand: A Unified Model for Diverse Controlled 4D Hand Motion Modeling](https://arxiv.org/abs/2602.21631), 2026; relevant motion-modeling paper, implementation not established here.
[^32]: Han et al. [UST-Hand: An Uncertainty-aware Spatiotemporal Point Cloud Interaction Network for 3D Self-supervised Hand Pose Estimation](https://arxiv.org/abs/2605.17742), 2026; implementation not established here.
[^33]: Chen et al. [HandOS official project page](https://idea-research.github.io/HandOSweb/), CVPR 2025; implementation not established here.
[^34]: Ma et al. [Uni-Hand forecasting repository](https://github.com/IRMVLab/UniHand), TPAMI 2026; distinct from the Sun et al. work.
[^35]: NVIDIA. [FoundationStereo official repository](https://github.com/NVlabs/FoundationStereo), CVPR 2025 stereo matching implementation.
[^36]: Meta. [CoTracker official repository](https://github.com/facebookresearch/co-tracker), image-point tracking implementation.
[^37]: OpenCV. [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), rectification, projection and coordinate transforms.
[^38]: Labbe. [FilterPy](https://github.com/rlabbe/filterpy), Kalman filtering and smoothing library.
[^39]: Casiez et al. [One Euro Filter](https://github.com/casiez/OneEuroFilter), reference filtering implementations.
