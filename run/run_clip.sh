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

# ---------------------------------------------------------------- render the delivery video
# Same renderer and same flags the fleet driver uses (viz_delivery_batch.py:82-88), so the output is
# comparable with the existing deliveries. render_v4_panels.py is the no-filter renderer, which is
# what with_filter=false in the manifest entry selects.
mkdir -p "$WORK/render"
BUCKET="stage-humyn-egocentric-stereo-data"
HEAD_KEY="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[0]['head_key'])" "$REPO/run/manifest_entry.json")"
aws s3 cp "s3://${BUCKET}/${HEAD_KEY}" "$WORK/input/head_pose_6dof.npz"

IMU_ARG=()
if aws s3 cp "${S3_INPUT}imu_accel.csv" "$WORK/input/imu_accel.csv" >/dev/null 2>&1; then
    IMU_ARG=(--imu "$WORK/input/imu_accel.csv")
    echo "  imu_accel.csv present - gravity-aligned panels enabled"
fi

# CLIP hand-vs-foot rejection. On by default for parity with the fleet driver. Set CLIP_FOOT=0 when
# comparing run types: dropping tracks in the renderer changes what you see independently of the
# pose model, which confounds the comparison.
CLIP_FOOT_ARG=()
[[ "${CLIP_FOOT:-1}" == "1" ]] && CLIP_FOOT_ARG=(--clip-foot)

python "$REPO/viz_delivery/render_v4_panels.py" \
    --video "$WORK/input/left_eye.mp4" \
    --npz   "$WORK/out/${CLIP}_hand21_keypoints.npz" \
    --head  "$WORK/input/head_pose_6dof.npz" \
    --out   "$WORK/render" \
    --future-sec 3 --past-sec 0 \
    "${CLIP_FOOT_ARG[@]}" "${IMU_ARG[@]}"

# Web encode, same ladder as viz_delivery_batch.py:97-100. The renderer names its output from the
# VIDEO stem (left_eye), so rename to the clip + run type here or the two run types collide.
ffmpeg -y -loglevel error -i "$WORK/render/left_eye_wrist_traj_panels.mp4" \
    -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
    -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
    "$WORK/render/${CLIP}_${RUN_TYPE}_wrist_traj_panels.mp4"

# ---------------------------------------------------------------- the three delivery metrics
# M3 needs an independent 2D reference. Type 2 produces one; for type 1 point EVAL_RTMPOSE at the
# type 2 run's *_2d_keypoints.npz so both are scored against the SAME observation.
EVAL_ARG=()
[[ -n "${EVAL_RTMPOSE:-}" ]] && EVAL_ARG=(--rtmpose "$EVAL_RTMPOSE")
python "$REPO/fusion/evaluate_run.py" \
    --run "${RUN_TYPE}=$WORK/out" \
    --out "$WORK/eval" "${EVAL_ARG[@]}"

# ---------------------------------------------------------------- upload (ONLY hand_pose_EgoForce/)
DEST="${S3_OUT}/${CLIP}/${RUN_TYPE}"
aws s3 cp "$WORK/out/${CLIP}_hand21_keypoints.npz" "${DEST}/${CLIP}_hand21_keypoints.npz"
aws s3 cp "$WORK/out/${CLIP}_fuse_stats.json"      "${DEST}/${CLIP}_fuse_stats.json"
aws s3 cp "$WORK/out/${CLIP}_3d_meta.json"         "${DEST}/${CLIP}_3d_meta.json"
aws s3 cp "$WORK/render/${CLIP}_${RUN_TYPE}_wrist_traj_panels.mp4" "${DEST}/${CLIP}_${RUN_TYPE}_wrist_traj_panels.mp4"
aws s3 cp "$WORK/eval/report.json"                 "${DEST}/evaluation_report.json"
aws s3 cp "$WORK/eval/report.md"                   "${DEST}/evaluation_report.md"

echo
echo "uploaded to ${DEST}/  (npz, stats, meta, rendered mp4, evaluation report)"
echo "manifest npz_key should read:"
echo "  labelling_results/hand_pose_EgoForce/${CLIP}/${RUN_TYPE}/${CLIP}_hand21_keypoints.npz"
