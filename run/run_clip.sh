#!/usr/bin/env bash
# Type 1 - EgoForce alone.
#
# One clip, end to end: stage the input from S3, run EgoForce, write the fused-schema npz, gate on
# QC, upload. This branch pins the run definition so the delivered npz is traceable to a commit.
#
# WRITE BOUNDARY: this script writes to exactly one S3 prefix,
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/
# Every other S3 access is a read. Do not add writes elsewhere.
#
# Usage:
#   ./run/run_clip.sh
#   CLIP=episode_048 S3_INPUT=s3://.../validation-result/ZED/home/episode_048/ ./run/run_clip.sh
#
# Requires the `egoforce` conda env active and AWS credentials with read on the input prefix and
# write on hand_pose_EgoForce/.

set -euo pipefail

CLIP="${CLIP:-episode_047}"
S3_INPUT="${S3_INPUT:-s3://stage-humyn-egocentric-stereo-data/validation-result/ZED/home/episode_047/}"
RUN_TYPE="type1_egoforce_only"
S3_OUT="s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce"

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
WORK="${WORK:-$HOME/egoforce_runs/${CLIP}/${RUN_TYPE}}"

echo "clip=${CLIP}  run_type=${RUN_TYPE}"
echo "input=${S3_INPUT}"
echo "work=${WORK}"
echo "repo=${REPO}  commit=$(git -C "$REPO" rev-parse --short HEAD)"

mkdir -p "$WORK/input" "$WORK/out"

# ---------------------------------------------------------------- stage input (READ ONLY)
aws s3 cp "${S3_INPUT}left_eye.mp4"     "$WORK/input/left_eye.mp4"
aws s3 cp "${S3_INPUT}calibration.json" "$WORK/input/calibration.json"

# ---------------------------------------------------------------- stage 1b: EgoForce 3D
# --stem matters: the output filename must match the manifest's npz_key. Without it the stem comes
# from the video basename ("left_eye") and the uploaded key would not match.
python "$REPO/fusion/run_egoforce_3d.py" \
    --video "$WORK/input/left_eye.mp4" \
    --calib "$WORK/input/calibration.json" \
    --calib-block rectified --eye left \
    --stem "$CLIP" \
    --out  "$WORK/out" \
    --overlay

# ---------------------------------------------------------------- stage 2: into the fused schema
# egoforce-only: no 2D model, nothing to match. Every row is source='wilor' with measured depth.
python "$REPO/fusion/fuse_egoforce_rtmpose.py" \
    --egoforce "$WORK/out/${CLIP}_3d_keypoints.npz" \
    --mode egoforce-only \
    --stem "$CLIP" \
    --out  "$WORK/out"

# ---------------------------------------------------------------- QC gate, before anything leaves
# reproj_median_px is the one check that proves kp3d_cam and K describe the same camera. If it fails
# every downstream number is meaningless, so refuse to upload rather than deliver a bad npz.
python - "$WORK/out/${CLIP}_3d_meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
r = meta.get('reproj_median_px')
d = (meta.get('depth_Z_m') or {}).get('frac_below_5cm')
print(f"  reproj_median_px   = {r}")
print(f"  frac_below_5cm     = {d}")
print(f"  head_vs_lift_px    = {meta.get('head_vs_lift_median_px')}")
print(f"  detections         = {meta.get('total_detections')} over {meta.get('frames_processed')} frames")
if r is None or r >= 1.0:
    sys.exit(f"ABORT: reproj_median_px={r} (need < 1.0) - the 3D and K describe different cameras")
print("  QC OK")
PY

# ---------------------------------------------------------------- upload (ONLY hand_pose_EgoForce/)
DEST="${S3_OUT}/${CLIP}/${RUN_TYPE}"
aws s3 cp "$WORK/out/${CLIP}_hand21_keypoints.npz" "${DEST}/${CLIP}_hand21_keypoints.npz"
aws s3 cp "$WORK/out/${CLIP}_fuse_stats.json"      "${DEST}/${CLIP}_fuse_stats.json"
aws s3 cp "$WORK/out/${CLIP}_3d_meta.json"         "${DEST}/${CLIP}_3d_meta.json"

echo
echo "uploaded to ${DEST}/"
echo "manifest npz_key should read:"
echo "  labelling_results/hand_pose_EgoForce/${CLIP}/${RUN_TYPE}/${CLIP}_hand21_keypoints.npz"
