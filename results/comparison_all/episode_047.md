# Run evaluation — three metrics

**M1** and **M2** are exact properties of the output. **M3 is a proxy**: it measures disagreement with an independent 2D model, which ranks variants but is not accuracy — two models that agree may both be wrong.

## M1 — Hand coverage & recovery  *(exact)*

Targets the "hands missing from the output" defect.

| Metric | T1_egoforce | T1_stabilised | T2_fusion | T2_stabilised |
| --- | --- | --- | --- | --- |
| left coverage within span | 0.9111 | 0.8118 | 0.9111 | 0.8118 |
| left longest gap (s) | 1.333 | 2.967 | 1.333 | 2.967 |
| left median recovery (s) | 0.133 | 0.133 | 0.133 | 0.133 |
| right coverage within span | 0.6737 | 0.5774 | 0.6759 | 0.5774 |
| right longest gap (s) | 1.633 | 2.033 | 1.633 | 2.033 |
| right median recovery (s) | 0.100 | 0.100 | 0.083 | 0.100 |
| detections | 2879 | 2524 | 2883 | 2524 |

## M2 — Geometric validity  *(exact)*

Targets the "wrong finger / wrist geometry" defect. `reproj_median_px` must be < 1 — above that the 3D and the intrinsics describe different cameras and nothing else here means anything. `bone_cv_median` catches a hand whose bone lengths drift frame to frame, which reads as a pulsing hand in the overlay.

| Metric | T1_egoforce | T1_stabilised | T2_fusion | T2_stabilised |
| --- | --- | --- | --- | --- |
| reproj median (px) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| reproj p95 (px) | 75241171.6520 | 0.0000 | 76640834.6340 | 0.0000 |
| depth implausible frac | 0.12331 | 0.00000 | 0.12452 | 0.00000 |
| bone length CV (median) | 0.05338 | 0.01352 | 0.05314 | 0.01803 |
| bone length CV (max) | 0.11144 | 0.02329 | 0.10994 | 0.02682 |
| degenerate span count | 29 | 1 | 29 | 1 |
| wrist Z median (m) | 0.323 | 0.330 | 0.314 | 0.319 |

## M3 — Fingertip agreement  *(PROXY, not accuracy)*

Both runs are scored against the **same** RTMPose 2D observation, so the comparison is apples-to-apples. Lower is better only in the sense of "closer to the independent observation".

| Metric | T1_egoforce | T1_stabilised | T2_fusion | T2_stabilised |
| --- | --- | --- | --- | --- |
| rows compared | 2276 | 2235 | 2298 | 2283 |
| fingertips median (px) | 42.522 | 46.312 | 4.564 | 25.058 |
| fingertips p95 (px) | 152.574 | 149.299 | 83.266 | 112.066 |
| all joints median (px) | 38.539 | 40.682 | 8.068 | 23.549 |
| non-fingertips median (px) | 38.081 | 39.954 | 9.160 | 23.371 |

## Source mix

| Metric | T1_egoforce | T1_stabilised | T2_fusion | T2_stabilised |
| --- | --- | --- | --- | --- |
| `fused` | 0 | 0 | 1745 | 1745 |
| `wilor` | 2879 | 2524 | 0 | 0 |
| `wilor_pnpfail` | 0 | 0 | 1138 | 779 |
