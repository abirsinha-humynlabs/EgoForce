#!/usr/bin/env bash
# Rebuild a delivery from an existing run's output, stabilised. CPU only - no GPU, no re-detection.
#
# Reads a completed run directory (produced by run/run_clip.sh), applies fusion/stabilise_run.py,
# re-renders the delivery video and re-scores the three metrics, then uploads under a distinct
# run-type suffix so the original delivery is NOT overwritten and the two stay comparable.
#
# WRITE BOUNDARY: unchanged from run_clip.sh - exactly one S3 prefix,
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/
#
# Usage:
#   ./run/run_stabilised.sh
#   CLIP=episode_048 SRC_RUN=type1_egoforce_only ./run/run_stabilised.sh
#   NO_UPLOAD=1 ./run/run_stabilised.sh          # build locally, upload nothing

set -euo pipefail

CLIP="${CLIP:-episode_047}"
SRC_RUN="${SRC_RUN:-type1_egoforce_only}"
DST_RUN="${DST_RUN:-${SRC_RUN}_v2}"
S3_OUT="s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce"

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SRC="${SRC:-$HOME/egoforce_runs/${CLIP}/${SRC_RUN}}"
WORK="${WORK:-$HOME/egoforce_runs/${CLIP}/${DST_RUN}}"

echo "clip=${CLIP}  ${SRC_RUN} -> ${DST_RUN}"
echo "src=${SRC}"
echo "work=${WORK}"
echo "repo=${REPO}  commit=$(git -C "$REPO" rev-parse --short HEAD)"

[[ -f "$SRC/out/${CLIP}_hand21_keypoints.npz" ]] || { echo "no source run at $SRC/out"; exit 1; }
[[ -f "$SRC/input/left_eye.mp4" ]] || { echo "source run has no staged video"; exit 1; }

mkdir -p "$WORK/out" "$WORK/render" "$WORK/eval"

# ---------------------------------------------------------------- stabilise (gate/smooth/rigidify)
python "$REPO/fusion/stabilise_run.py" \
    --npz "$SRC/out/${CLIP}_hand21_keypoints.npz" \
    --out "$WORK/out" --stem "$CLIP"

# The 3D npz carries `processed_frames`, which evaluate_run.py uses as the coverage DENOMINATOR.
# Without it coverage falls back to an inferred range and stops being comparable with the source run.
cp "$SRC/out/${CLIP}_3d_keypoints.npz" "$WORK/out/"
cp "$SRC/out/${CLIP}_3d_meta.json" "$WORK/out/"

# ---------------------------------------------------------------- render (same flags as run_clip.sh)
IMU_ARG=()
[[ -f "$SRC/input/imu_accel.csv" ]] && IMU_ARG=(--imu "$SRC/input/imu_accel.csv")
CLIP_FOOT_ARG=()
[[ "${CLIP_FOOT:-0}" == "1" ]] && CLIP_FOOT_ARG=(--clip-foot)

python "$REPO/viz_delivery/render_v4_panels.py" \
    --video "$SRC/input/left_eye.mp4" \
    --npz   "$WORK/out/${CLIP}_hand21_keypoints.npz" \
    --head  "$SRC/input/head_pose_6dof.npz" \
    --out   "$WORK/render" \
    --future-sec 3 --past-sec 0 \
    "${CLIP_FOOT_ARG[@]}" "${IMU_ARG[@]}"

ffmpeg -y -loglevel error -i "$WORK/render/left_eye_wrist_traj_panels.mp4" \
    -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
    -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
    "$WORK/render/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4"

# ---------------------------------------------------------------- the three metrics
python "$REPO/fusion/evaluate_run.py" --run "${DST_RUN}=$WORK/out" --out "$WORK/eval"

# ---------------------------------------------------------------- upload
if [[ "${NO_UPLOAD:-0}" == "1" ]]; then
    echo "NO_UPLOAD=1 - built locally, nothing uploaded"; exit 0
fi
DEST="${S3_OUT}/${CLIP}/${DST_RUN}"
aws s3 cp "$WORK/out/${CLIP}_hand21_keypoints.npz"  "${DEST}/${CLIP}_hand21_keypoints.npz"
aws s3 cp "$WORK/out/${CLIP}_stabilise_stats.json"  "${DEST}/${CLIP}_stabilise_stats.json"
aws s3 cp "$WORK/out/${CLIP}_3d_meta.json"          "${DEST}/${CLIP}_3d_meta.json"
aws s3 cp "$WORK/render/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4" "${DEST}/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4"
aws s3 cp "$WORK/eval/report.json"                  "${DEST}/evaluation_report.json"
aws s3 cp "$WORK/eval/report.md"                    "${DEST}/evaluation_report.md"

echo
echo "uploaded to ${DEST}/"
