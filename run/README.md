# Run type 1 — EgoForce alone

This branch pins one run definition. The code is shared with
`feat/egoforce-rtmpose-fusion`; what this branch fixes is **which pipeline is run, with which
parameters, writing to which key** — so a delivered npz is traceable to a commit.

The companion is `run/type2-fusion`, which runs EgoForce + RTMPose-m Hand5 with the articulated MANO
refit. Comparing the two is the whole point: type 1 is the baseline that isolates the model swap, and
type 2 adds the fusion on top. See [`../PIPELINE.md`](../PIPELINE.md) §5.3 for what the refit does and
why it is the only variant that can correct a wrongly bent finger.

## What runs

| Stage | Script | Output |
| --- | --- | --- |
| 1b | `fusion/run_egoforce_3d.py` | `<clip>_3d_keypoints.npz`, `<clip>_3d_meta.json`, overlay mp4 |
| 2 | `fusion/fuse_egoforce_rtmpose.py --mode egoforce-only` | `<clip>_hand21_keypoints.npz`, `<clip>_fuse_stats.json` |

No 2D model and nothing to match, so every row is `source='wilor'` with `depth_measured=True`.
(The label reuses the mono-pipeline's vocabulary so `postprocess.py` works untouched; the additive
`producer` array records that it is actually EgoForce. See `PIPELINE.md` §5.4.)

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
./run/run_clip.sh                                   # defaults to episode_047
CLIP=episode_048 S3_INPUT=s3://.../home/episode_048/ ./run/run_clip.sh
```

The script stages the input, runs both stages, **gates on QC, and refuses to upload if it fails**.
The gate is `reproj_median_px < 1.0` from `_3d_meta.json` — the one check proving `kp3d_cam` and the
stored `K` describe the same camera. If it fails, everything downstream is meaningless.

## Manifest entry

`run/manifest_entry.json` holds the entry for this run type:

```json
{
  "clip": "episode_047_egoforce",
  "run_type": "egoforce only",
  "with_filter": false,
  "head_key": "labelling_results/6dof_head_pose_v2/home_episode_047/head_pose_6dof.npz",
  "npz_key": "labelling_results/hand_pose_EgoForce/episode_047/type1_egoforce_only/episode_047_hand21_keypoints.npz",
  "input_dir": "validation-result/ZED/home/episode_047/"
}
```

Four things about it that are not obvious:

- **`clip` must be unique across the whole manifest.** `viz_delivery_batch.out_key()` keys the output
  mp4 off `clip` alone, and `claim_next` skips any clip whose output already exists. A duplicate
  silently collides — `episode_047` is already taken by the pre-existing filtered entry, and
  `episode_047_egoforce_fusion` is taken by run type 2.
- **`input_dir` is a bucket-relative key**, not an `s3://` URI. `viz_delivery_batch.py:75-78` does
  `download_file(B, ind + "left_eye.mp4")` with `B` already the bucket.
- **`run_type` is documentary.** Nothing reads it; the renderer is chosen from `with_filter`.
- **`with_filter: false`** on purpose. The filtered renderer applies CLIP foot and bad-shape drops,
  which would confound a comparison of the pose models.

`head_key` follows `6dof_head_pose_v2/<input_dir rel path with / and space → _>/head_pose_6dof.npz`,
but that rule only holds for 528 of the 617 manifest entries — 54 use `v3`, 34 have a region prefix
that differs from `input_dir`, and 2 are truncated. **Verify against S3 before trusting a derived
`head_key`**; this one was confirmed to exist (131,570 bytes).

## Clip facts (verified)

`validation-result/ZED/home/episode_047/` — 1920×1080, 30 fps CFR, 1824 frames (60.8 s).
Rectified intrinsics fx = fy = 1065.0692, cx = 959.691, cy = 541.633, no distortion block, and
cx ≈ W/2, so the calibration matches this video. `imu_accel.csv` is present, so the renderer's IMU
panel will populate.
