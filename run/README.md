# Run type 2 — EgoForce + RTMPose-m Hand5, articulated fusion

This branch pins one run definition. The code is shared with
`feat/egoforce-rtmpose-fusion`; what this branch fixes is **which pipeline is run, with which
parameters, writing to which key** — so a delivered npz is traceable to a commit.

The companion is `run/type1-egoforce-only`. That one is the baseline that isolates the *model swap*;
this one adds the *fusion* on top. Both must be run on the same clip for either number to mean
anything — if type 2 does not beat type 1, the refit is not earning its complexity.

## What runs

| Stage | Script | Output |
| --- | --- | --- |
| 1b | `fusion/run_egoforce_3d.py` | `<clip>_3d_keypoints.npz` (+ MANO params), `<clip>_3d_meta.json` |
| 1a | `fusion/run_rtmpose_2d.py --boxes npz` | `<clip>_2d_keypoints.npz`, `<clip>_2d_meta.json` |
| 2 | `fusion/fuse_egoforce_rtmpose.py --mode articulated` | `<clip>_hand21_keypoints.npz`, `<clip>_fuse_stats.json` |

`--boxes npz` reuses EgoForce's hand boxes, so every 2D row corresponds 1:1 to a 3D row on the same
frame and the same physical hand. The fusion then needs no wrist matching and cannot mis-pair, and
the `width`/`height`/`step` agreement is true by construction.

## What the refit actually does

Instead of rigidly re-posing a fixed skeleton onto the 2D — which is what the existing
`fuse_2d_3d.py::pnp_refit` does, and why a wrongly bent finger stays wrongly bent there — this
optimises the MANO **parameters** (global orientation, 15 joint rotations, shape, translation)
against RTMPose's confidence-weighted landmarks. Full objective in
[`../PIPELINE.md`](../PIPELINE.md) §5.3. Two parts are load-bearing:

- **The depth prior.** 2D reprojection is scale–depth degenerate — a hand twice as large at twice the
  depth projects identically. RTMPose contributes nothing to depth, so translation stays anchored to
  EgoForce, the model that actually resolved it.
- **The reject gates.** With articulation free, a refit can "win" on reprojection by stretching bones
  or sliding in depth. `accept_refit` rejects on bone-length change, depth change, a reprojection
  ceiling, and failure to improve. Rejected rows fall back to raw EgoForce 3D, labelled
  `wilor_pnpfail`.

## Write boundary

`run/run_clip.sh` writes to exactly one S3 prefix:

```
s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/
```

Every other S3 access is a read. **Do not add writes elsewhere.**

Note this means `viz_delivery_batch.py` cannot be run unmodified — it writes the rendered mp4
(`:95`), worker locks (`:49`, `:55`) and a host report (`:164`), all under
`labelling_results/viz_delivery/`, none of them ours. Either point its `VIZ` constant at
`hand_pose_EgoForce`, or get that prefix authorised separately.

## Run it

```bash
conda activate egoforce
bash scripts/download_rtmpose_hand5.sh      # once, if not already fetched
./run/run_clip.sh                           # defaults to episode_047
CLIP=episode_048 S3_INPUT=s3://.../home/episode_048/ ./run/run_clip.sh
```

Override `RTM_CFG` / `RTM_CKPT` if the download script took its sparse-clone fallback route, where
the config lands under `_DATA/rtmpose_hand5/mmpose_src/configs/…` instead.

## QC gates — the script refuses to upload if either fails

1. **`reproj_median_px < 1.0`** from `_3d_meta.json`. The one check proving `kp3d_cam` and the stored
   `K` describe the same camera. If it fails, everything downstream is meaningless.
2. **At least one refit accepted.** If `refit.reasons.accepted == 0`, every row fell back to raw
   EgoForce and this is type 1 output wearing a type 2 label — delivering it would misrepresent the
   run. Inspect `refit.reasons` (it breaks down rejections by cause) before overriding.

Also worth reading in `_fuse_stats.json` before you trust the result: `bone_change_median` and
`refit.depth_change_m_median`. If those grew, the refit is buying reprojection accuracy by deforming
the hand, and the gates need tightening rather than loosening.

## Manifest entry

`run/manifest_entry.json`:

```json
{
  "clip": "episode_047_egoforce_fusion",
  "run_type": "egoforce + rtmpose articulated fusion",
  "with_filter": false,
  "head_key": "labelling_results/6dof_head_pose_v2/home_episode_047/head_pose_6dof.npz",
  "npz_key": "labelling_results/hand_pose_EgoForce/episode_047/type2_egoforce_rtmpose/episode_047_hand21_keypoints.npz",
  "input_dir": "validation-result/ZED/home/episode_047/"
}
```

Four things about it that are not obvious:

- **`clip` must be unique across the whole manifest.** `viz_delivery_batch.out_key()` keys the output
  mp4 off `clip` alone, and `claim_next` skips any clip whose output already exists. A duplicate
  silently collides — `episode_047` is taken by the pre-existing filtered entry, and
  `episode_047_egoforce` by run type 1.
- **`input_dir` is a bucket-relative key**, not an `s3://` URI. `viz_delivery_batch.py:75-78` does
  `download_file(B, ind + "left_eye.mp4")` with `B` already the bucket.
- **`run_type` is documentary.** Nothing reads it; the renderer is chosen from `with_filter`.
- **`with_filter: false`** on purpose, and it must match type 1 — the filtered renderer applies CLIP
  foot and bad-shape drops, which would confound a comparison of the two run types.

`head_key` follows `6dof_head_pose_v2/<input_dir rel path with / and space → _>/head_pose_6dof.npz`,
but that rule only holds for 528 of the 617 manifest entries — 54 use `v3`, 34 have a region prefix
that differs from `input_dir`, and 2 are truncated. **Verify against S3 before trusting a derived
`head_key`**; this one was confirmed to exist (131,570 bytes).

## Clip facts (verified)

`validation-result/ZED/home/episode_047/` — 1920×1080, 30 fps CFR, 1824 frames (60.8 s).
Rectified intrinsics fx = fy = 1065.0692, cx = 959.691, cy = 541.633, no distortion block, and
cx ≈ W/2, so the calibration matches this video. `imu_accel.csv` is present, so the renderer's IMU
panel will populate.
