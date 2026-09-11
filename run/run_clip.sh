#!/usr/bin/env bash
# Type 2 - EgoForce + RTMPose-m Hand5, articulated MANO refit.
#
# One clip, end to end: stage the input from S3, run EgoForce, run RTMPose on EgoForce's boxes, refit
# MANO articulation against RTMPose's confidence-weighted landmarks, gate on QC, upload. This branch
# pins the run definition so the delivered npz is traceable to a commit.
#
# WRITE BOUNDARY: this script writes to exactly one S3 prefix,
#   s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce/
# Every other S3 access is a read. Do not add writes elsewhere.
#
# Usage:
#   ./run/run_clip.sh
#   CLIP=episode_048 S3_INPUT=s3://.../validation-result/ZED/home/episode_048/ ./run/run_clip.sh
#
# Requires the `egoforce` conda env active, mmpose installed, the RTMPose Hand5 checkpoint fetched
# (scripts/download_rtmpose_hand5.sh), and AWS credentials with read on the input prefix and write on
# hand_pose_EgoForce/.

set -euo pipefail

CLIP="${CLIP:-episode_047}"
S3_INPUT="${S3_INPUT:-s3://stage-humyn-egocentric-stereo-data/validation-result/ZED/home/episode_047/}"
RUN_TYPE="type2_egoforce_rtmpose"
S3_OUT="s3://stage-humyn-egocentric-stereo-data/labelling_results/hand_pose_EgoForce"

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
WORK="${WORK:-$HOME/egoforce_runs/${CLIP}/${RUN_TYPE}}"

# RTMPose config/checkpoint. Defaults match scripts/download_rtmpose_hand5.sh's mim route; on its
# sparse-clone fallback route the config lives under _DATA/rtmpose_hand5/mmpose_src/configs/...
RTM_CFG="${RTM_CFG:-$REPO/_DATA/rtmpose_hand5/rtmpose-m_8xb256-210e_hand5-256x256.py}"
RTM_CKPT="${RTM_CKPT:-$REPO/_DATA/rtmpose_hand5/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth}"

echo "clip=${CLIP}  run_type=${RUN_TYPE}"
echo "input=${S3_INPUT}"
echo "work=${WORK}"
echo "repo=${REPO}  commit=$(git -C "$REPO" rev-parse --short HEAD)"

for f in "$RTM_CFG" "$RTM_CKPT"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; echo "run: bash scripts/download_rtmpose_hand5.sh" >&2; exit 1; }
done

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

# ---------------------------------------------------------------- stage 1a: RTMPose 2D
# --boxes npz reuses EgoForce's hand boxes, so every 2D row corresponds 1:1 to a 3D row on the same
# frame and the same physical hand. The fusion then needs no wrist matching and cannot mis-pair.
python "$REPO/fusion/run_rtmpose_2d.py" \
    --video "$WORK/input/left_eye.mp4" \
    --boxes npz --boxes-npz "$WORK/out/${CLIP}_3d_keypoints.npz" \
    --rtmpose-config     "$RTM_CFG" \
    --rtmpose-checkpoint "$RTM_CKPT" \
    --stem "$CLIP" \
    --out  "$WORK/out" \
    --overlay

# ---------------------------------------------------------------- stage 2: articulated refit
python "$REPO/fusion/fuse_egoforce_rtmpose.py" \
    --egoforce "$WORK/out/${CLIP}_3d_keypoints.npz" \
    --rtmpose  "$WORK/out/${CLIP}_2d_keypoints.npz" \
    --mode articulated \
    --stem "$CLIP" \
    --out  "$WORK/out" \
    --video "$WORK/input/left_eye.mp4" --overlay

# ---------------------------------------------------------------- QC gate, before anything leaves
python - "$WORK/out/${CLIP}_3d_meta.json" "$WORK/out/${CLIP}_fuse_stats.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
fuse = json.load(open(sys.argv[2]))
r = meta.get('reproj_median_px')
print(f"  reproj_median_px   = {r}")
print(f"  frac_below_5cm     = {(meta.get('depth_Z_m') or {}).get('frac_below_5cm')}")
print(f"  head_vs_lift_px    = {meta.get('head_vs_lift_median_px')}")
print(f"  source_mix         = {fuse.get('source_mix')}")
print(f"  refit              = {(fuse.get('refit') or {}).get('reasons')}")
print(f"  bone_change_median = {(fuse.get('refit') or {}).get('bone_change_median')}")
if r is None or r >= 1.0:
    sys.exit(f"ABORT: reproj_median_px={r} (need < 1.0) - the 3D and K describe different cameras")
# A refit that never gets accepted means the fusion contributed nothing; delivering it as a
# "fusion" run would be misleading. Surface it loudly rather than silently shipping type 1 twice.
reasons = (fuse.get('refit') or {}).get('reasons') or {}
if reasons.get('total') and not reasons.get('accepted'):
    sys.exit("ABORT: 0 refits accepted - this is type 1 output wearing a type 2 label. "
             "Inspect refit.reasons before delivering.")
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
# This run produced its own RTMPose observation, so it is the natural M3 reference. Point type 1's
# EVAL_RTMPOSE at this same file to score both runs against identical evidence.
EVAL_RTMPOSE="${EVAL_RTMPOSE:-$WORK/out/${CLIP}_2d_keypoints.npz}"
EVAL_ARG=()
[[ -f "$EVAL_RTMPOSE" ]] && EVAL_ARG=(--rtmpose "$EVAL_RTMPOSE")
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
aws s3 cp "$WORK/out/${CLIP}_2d_meta.json"         "${DEST}/${CLIP}_2d_meta.json"

echo
echo "uploaded to ${DEST}/  (npz, stats, meta, rendered mp4, evaluation report)"
echo "manifest npz_key should read:"
echo "  labelling_results/hand_pose_EgoForce/${CLIP}/${RUN_TYPE}/${CLIP}_hand21_keypoints.npz"
