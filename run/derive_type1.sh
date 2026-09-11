#!/usr/bin/env bash
# Derive the type 1 (EgoForce-only) delivery from a completed type 2 run. No GPU.
#
# The stage-1b EgoForce pass is bit-identical between the two run types: same script, same flags,
# same checkpoint, and it is deterministic - episode_047's type 1 and type 2 runs produced
# reproj_median_px = 2.643257162677913e-05 from independent invocations. Type 1 is then just the
# `egoforce-only` fusion over that same npz, so re-running inference to obtain it buys nothing and
# costs ~5 minutes of GPU.
#
# WRITE BOUNDARY: unchanged - exactly one prefix,
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/
#
# Usage:
#   CLIP=episode_048 ./run/derive_type1.sh
#   NO_UPLOAD=1 CLIP=episode_048 ./run/derive_type1.sh

set -euo pipefail

CLIP="${CLIP:-episode_047}"
SRC_RUN="${SRC_RUN:-type2_egoforce_rtmpose}"
DST_RUN="${DST_RUN:-type1_egoforce_only}"
S3_OUT="s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce"

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SRC="${SRC:-$HOME/egoforce_runs/${CLIP}/${SRC_RUN}}"
WORK="${WORK:-$HOME/egoforce_runs/${CLIP}/${DST_RUN}}"

echo "clip=${CLIP}  ${SRC_RUN} -> ${DST_RUN}  (no GPU)"
echo "repo=${REPO}  commit=$(git -C "$REPO" rev-parse --short HEAD)"

[[ -f "$SRC/out/${CLIP}_3d_keypoints.npz" ]] || { echo "no 3D npz at $SRC/out"; exit 1; }
mkdir -p "$WORK/out" "$WORK/render" "$WORK/eval" "$WORK/input"

cp "$SRC/out/${CLIP}_3d_keypoints.npz" "$SRC/out/${CLIP}_3d_meta.json" "$WORK/out/"

# Link the source run's staged inputs into this one. Without them the derived run directory is not
# self-contained and anything downstream that expects a complete run - run/run_stabilised.sh, which
# needs the video to re-render - refuses it with "source run has no staged video". Symlinks, not
# copies: left_eye.mp4 alone is 200-265 MB and there is no reason to hold two.
for f in left_eye.mp4 calibration.json head_pose_6dof.npz imu_accel.csv; do
    [[ -f "$SRC/input/$f" ]] && ln -sf "$SRC/input/$f" "$WORK/input/$f"
done

# egoforce-only: no 2D model, nothing to match. Every row is source='wilor' with measured depth.
python "$REPO/fusion/fuse_egoforce_rtmpose.py" \
    --egoforce "$WORK/out/${CLIP}_3d_keypoints.npz" \
    --mode egoforce-only --stem "$CLIP" --out "$WORK/out"

python - "$WORK/out/${CLIP}_3d_meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
r = meta.get('reproj_median_px')
print(f"  reproj_median_px   = {r}")
print(f"  frac_below_5cm     = {(meta.get('depth_Z_m') or {}).get('frac_below_5cm')}")
print(f"  head_vs_lift_px    = {meta.get('head_vs_lift_median_px')}")
if r is None or r >= 1.0:
    sys.exit(f"ABORT: reproj_median_px={r} (need < 1.0)")
print("  QC OK")
PY

CLIP_FOOT_ARG=(); [[ "${CLIP_FOOT:-0}" == "1" ]] && CLIP_FOOT_ARG=(--clip-foot)
IMU_ARG=();       [[ -f "$SRC/input/imu_accel.csv" ]] && IMU_ARG=(--imu "$SRC/input/imu_accel.csv")

python "$REPO/viz_delivery/render_v4_panels.py" \
    --video "$SRC/input/left_eye.mp4" \
    --npz   "$WORK/out/${CLIP}_hand21_keypoints.npz" \
    --head  "$SRC/input/head_pose_6dof.npz" \
    --out   "$WORK/render" --future-sec 3 --past-sec 0 \
    "${CLIP_FOOT_ARG[@]}" "${IMU_ARG[@]}"

ffmpeg -y -loglevel error -i "$WORK/render/left_eye_wrist_traj_panels.mp4" \
    -vf "scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
    -c:v libx264 -profile:v high -crf 18 -preset veryfast -movflags +faststart \
    "$WORK/render/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4"

EVAL_ARG=(); [[ -n "${EVAL_RTMPOSE:-}" ]] && EVAL_ARG=(--rtmpose "$EVAL_RTMPOSE")
python "$REPO/fusion/evaluate_run.py" --run "${DST_RUN}=$WORK/out" --out "$WORK/eval" "${EVAL_ARG[@]}"

if [[ "${NO_UPLOAD:-0}" == "1" ]]; then echo "NO_UPLOAD=1 - nothing uploaded"; exit 0; fi
DEST="${S3_OUT}/${CLIP}/${DST_RUN}"
aws s3 cp "$WORK/out/${CLIP}_hand21_keypoints.npz" "${DEST}/${CLIP}_hand21_keypoints.npz"
aws s3 cp "$WORK/out/${CLIP}_fuse_stats.json"      "${DEST}/${CLIP}_fuse_stats.json"
aws s3 cp "$WORK/out/${CLIP}_3d_meta.json"         "${DEST}/${CLIP}_3d_meta.json"
aws s3 cp "$WORK/render/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4" "${DEST}/${CLIP}_${DST_RUN}_wrist_traj_panels.mp4"
aws s3 cp "$WORK/eval/report.json"                 "${DEST}/evaluation_report.json"
aws s3 cp "$WORK/eval/report.md"                   "${DEST}/evaluation_report.md"
echo; echo "uploaded to ${DEST}/"
