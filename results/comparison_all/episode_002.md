# Run evaluation — three metrics

**M1** and **M2** are exact properties of the output. **M3 is a proxy**: it measures disagreement with an independent 2D model, which ranks variants but is not accuracy — two models that agree may both be wrong.

## M1 — Hand coverage & recovery  *(exact)*

Targets the "hands missing from the output" defect.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| left coverage within span | 0.9197 | 0.7869 | 0.9192 | 0.7863 |
| left longest gap (s) | 0.567 | 0.733 | 0.567 | 0.733 |
| left median recovery (s) | 0.100 | 0.133 | 0.100 | 0.133 |
| right coverage within span | 0.8837 | 0.8241 | 0.8837 | 0.8241 |
| right longest gap (s) | 0.800 | 1.367 | 0.800 | 1.367 |
| right median recovery (s) | 0.100 | 0.100 | 0.100 | 0.100 |
| detections | 3276 | 2924 | 3275 | 2923 |

## M2 — Geometric validity  *(exact)*

Targets the "wrong finger / wrist geometry" defect. `reproj_median_px` must be < 1 — above that the 3D and the intrinsics describe different cameras and nothing else here means anything. `bone_cv_median` catches a hand whose bone lengths drift frame to frame, which reads as a pulsing hand in the overlay.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| reproj median (px) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| reproj p95 (px) | 55573642.2980 | 0.0000 | 55634164.7330 | 0.0000 |
| depth implausible frac | 0.10745 | 0.00000 | 0.10748 | 0.00000 |
| bone length CV (median) | 0.04427 | 0.01068 | 0.04706 | 0.01102 |
| bone length CV (max) | 0.08395 | 0.01810 | 0.08675 | 0.02321 |
| degenerate span count | 67 | 1 | 67 | 1 |
| wrist Z median (m) | 0.413 | 0.430 | 0.410 | 0.432 |

## M3 — Fingertip agreement  *(PROXY, not accuracy)*

Both runs are scored against the **same** RTMPose 2D observation, so the comparison is apples-to-apples. Lower is better only in the sense of "closer to the independent observation".

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| rows compared | 2581 | 2560 | 2649 | 2630 |
| fingertips median (px) | 36.724 | 37.477 | 3.233 | 13.087 |
| fingertips p95 (px) | 162.514 | 125.689 | 108.136 | 64.411 |
| all joints median (px) | 30.756 | 32.218 | 6.781 | 14.838 |
| non-fingertips median (px) | 29.741 | 31.193 | 7.899 | 15.220 |

## Source mix

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| `fused` | 0 | 0 | 1951 | 1951 |
| `wilor` | 3276 | 2924 | 0 | 0 |
| `wilor_pnpfail` | 0 | 0 | 1324 | 972 |
