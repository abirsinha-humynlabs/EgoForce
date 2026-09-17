# Run evaluation — three metrics

**M1** and **M2** are exact properties of the output. **M3 is a proxy**: it measures disagreement with an independent 2D model, which ranks variants but is not accuracy — two models that agree may both be wrong.

## M1 — Hand coverage & recovery  *(exact)*

Targets the "hands missing from the output" defect.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| left coverage within span | 0.6665 | 0.5875 | 0.6654 | 0.5864 |
| left longest gap (s) | 2.933 | 3.233 | 2.933 | 3.233 |
| left median recovery (s) | 0.200 | 0.133 | 0.200 | 0.133 |
| right coverage within span | 0.6401 | 0.5786 | 0.6412 | 0.5797 |
| right longest gap (s) | 2.467 | 2.600 | 2.467 | 2.600 |
| right median recovery (s) | 0.233 | 0.200 | 0.233 | 0.200 |
| detections | 2380 | 2124 | 2380 | 2124 |

## M2 — Geometric validity  *(exact)*

Targets the "wrong finger / wrist geometry" defect. `reproj_median_px` must be < 1 — above that the 3D and the intrinsics describe different cameras and nothing else here means anything. `bone_cv_median` catches a hand whose bone lengths drift frame to frame, which reads as a pulsing hand in the overlay.

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| reproj median (px) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| reproj p95 (px) | 63941831.1220 | 0.0000 | 63941831.1220 | 0.0000 |
| depth implausible frac | 0.10798 | 0.00000 | 0.10756 | 0.00000 |
| bone length CV (median) | 0.05568 | 0.01357 | 0.05539 | 0.01411 |
| bone length CV (max) | 0.08881 | 0.02073 | 0.09335 | 0.02093 |
| degenerate span count | 46 | 0 | 46 | 0 |
| wrist Z median (m) | 0.478 | 0.492 | 0.483 | 0.503 |

## M3 — Fingertip agreement  *(PROXY, not accuracy)*

Both runs are scored against the **same** RTMPose 2D observation, so the comparison is apples-to-apples. Lower is better only in the sense of "closer to the independent observation".

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| rows compared | 1913 | 1902 | 1931 | 1929 |
| fingertips median (px) | 28.734 | 32.878 | 1.401 | 18.672 |
| fingertips p95 (px) | 119.106 | 110.498 | 85.797 | 79.805 |
| all joints median (px) | 25.996 | 29.724 | 4.785 | 17.641 |
| non-fingertips median (px) | 25.449 | 29.099 | 5.488 | 17.423 |

## Source mix

| Metric | T1 | T1_stab | T2 | T2_stab |
| --- | --- | --- | --- | --- |
| `fused` | 0 | 0 | 1315 | 1315 |
| `wilor` | 2380 | 2124 | 0 | 0 |
| `wilor_pnpfail` | 0 | 0 | 1065 | 809 |
