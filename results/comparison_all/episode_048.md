# Run evaluation — three metrics

**M1** and **M2** are exact properties of the output. **M3 is a proxy**: it measures disagreement with an independent 2D model, which ranks variants but is not accuracy — two models that agree may both be wrong.

## M1 — Hand coverage & recovery  *(exact)*

Targets the "hands missing from the output" defect.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| left coverage within span | 0.8925 | 0.8179 | 0.8925 | 0.8179 |
| left longest gap (s) | 1.467 | 3.167 | 1.467 | 3.167 |
| left median recovery (s) | 0.117 | 0.100 | 0.117 | 0.100 |
| right coverage within span | 0.9962 | 0.8697 | 0.9962 | 0.8697 |
| right longest gap (s) | 0.100 | 1.933 | 0.100 | 1.933 |
| right median recovery (s) | 0.033 | 0.067 | 0.033 | 0.067 |
| detections | 3442 | 3073 | 3442 | 3073 |

## M2 — Geometric validity  *(exact)*

Targets the "wrong finger / wrist geometry" defect. `reproj_median_px` must be < 1 — above that the 3D and the intrinsics describe different cameras and nothing else here means anything. `bone_cv_median` catches a hand whose bone lengths drift frame to frame, which reads as a pulsing hand in the overlay.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| reproj median (px) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| reproj p95 (px) | 57594467.5720 | 0.0000 | 57594467.5720 | 0.0000 |
| depth implausible frac | 0.10721 | 0.00000 | 0.10721 | 0.00000 |
| bone length CV (median) | 0.04327 | 0.00511 | 0.04293 | 0.00519 |
| bone length CV (max) | 0.13430 | 0.01487 | 0.12322 | 0.01480 |
| degenerate span count | 57 | 0 | 57 | 0 |
| wrist Z median (m) | 0.389 | 0.401 | 0.395 | 0.403 |

## M3 — Fingertip agreement  *(PROXY, not accuracy)*

Both runs are scored against the **same** RTMPose 2D observation, so the comparison is apples-to-apples. Lower is better only in the sense of "closer to the independent observation".

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| rows compared | 3007 | 3006 | 3007 | 3006 |
| fingertips median (px) | 27.115 | 28.337 | 1.455 | 9.008 |
| fingertips p95 (px) | 66.102 | 73.914 | 26.866 | 44.496 |
| all joints median (px) | 25.398 | 26.620 | 3.990 | 11.184 |
| non-fingertips median (px) | 24.960 | 26.253 | 4.990 | 11.769 |

## Source mix

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| `fused` | 0 | 0 | 2637 | 2637 |
| `wilor` | 3442 | 3073 | 0 | 0 |
| `wilor_pnpfail` | 0 | 0 | 805 | 436 |
