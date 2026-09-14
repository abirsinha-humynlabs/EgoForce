#!/usr/bin/env bash
# One clip, both run types, delivering ONLY the repaired+stabilised output.
#
# run/run_clip.sh + run/derive_type1.sh each render and upload their RAW result. When only the
# stabilised delivery is wanted that is two 13-minute single-threaded renders producing artefacts
# destined for deletion. This does the same work without them:
#
#   1. one EgoForce 3D pass (GPU) and one RTMPose 2D pass (GPU)
#   2. BOTH fusions off that single inference - articulated for type 2 (GPU refit),
#      egoforce-only for type 1 (CPU, seconds). The 3D stage is identical between run types and
#      deterministic, so running inference twice buys nothing.
#   3. repair + stabilise both, rendering the two CONCURRENTLY - they are independent, and running
#      them in series left 70.8% of the 4 vCPU idle.
#   4. upload only <run>_v3.
#
# Roughly 30 minutes instead of 75, with no change to what is delivered.
#
# WRITE BOUNDARY unchanged: one prefix, labelling_results/hand_pose_EgoForce/.
#
# Usage:
#   CLIP=episode_088 S3_INPUT=s3://.../validation-result/ZED/home/episode_088/ ./run/run_clip_v3.sh
#   NO_UPLOAD=1 CLIP=... ./run/run_clip_v3.sh

set -euo pipefail

CLIP="${CLIP:?set CLIP}"
S3_INPUT="${S3_INPUT:?set S3_INPUT}"
BUCKET="stage-humyn-egocentric-stereo-data"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
WORK="${WORK:-$HOME/egoforce_runs/${CLIP}}"
T2="$WORK/type2_egoforce_rtmpose"
T1="$WORK/type1_egoforce_only"

RTM_CFG="${RTM_CFG:-$REPO/_DATA/rtmpose_hand5/rtmpose-m_8xb256-210e_hand5-256x256.py}"
RTM_CKPT="${RTM_CKPT:-$REPO/_DATA/rtmpose_hand5/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth}"
for f in "$RTM_CFG" "$RTM_CKPT"; do
    [[ -f "$f" ]] || { echo "missing: $f (run scripts/download_rtmpose_hand5.sh)" >&2; exit 1; }
done

echo "clip=${CLIP}  commit=$(git -C "$REPO" rev-parse --short HEAD)"
mkdir -p "$T2/input" "$T2/out" "$T1/input" "$T1/out"

# ---------------------------------------------------------------- stage input (READ ONLY)
aws s3 cp "${S3_INPUT}left_eye.mp4"     "$T2/input/left_eye.mp4"
aws s3 cp "${S3_INPUT}calibration.json" "$T2/input/calibration.json"
aws s3 cp "${S3_INPUT}imu_accel.csv"    "$T2/input/imu_accel.csv" || echo "  no imu_accel.csv"

# Head pose by discovery - the layout under 6dof_head_pose_v2/ does not mirror the input path.
if [[ -z "${HEAD_KEY:-}" ]]; then
    mapfile -t HC < <(aws s3 ls "s3://${BUCKET}/labelling_results/6dof_head_pose_v2/" \
        | awk '{print $2}' | grep -E "_${CLIP}/$" || true)
    [[ ${#HC[@]} -eq 1 ]] || { echo "expected 1 head-pose dir ending in _${CLIP}, got ${#HC[@]}: ${HC[*]:-none}" >&2; exit 1; }
    HEAD_KEY="labelling_results/6dof_head_pose_v2/${HC[0]}head_pose_6dof.npz"
    echo "  head pose: ${HC[0]%/} (discovered)"
fi
aws s3 cp "s3://${BUCKET}/${HEAD_KEY}" "$T2/input/head_pose_6dof.npz"
for f in left_eye.mp4 calibration.json head_pose_6dof.npz imu_accel.csv; do
    [[ -f "$T2/input/$f" ]] && ln -sf "$T2/input/$f" "$T1/input/$f"
done

# ---------------------------------------------------------------- inference, once (GPU)
# No --overlay: the raw overlays are diagnostics we are not delivering, and each costs a full
# re-encode of the clip.
python "$REPO/fusion/run_egoforce_3d.py" --video "$T2/input/left_eye.mp4" \
    --calib "$T2/input/calibration.json" --calib-block rectified --eye left \
    --stem "$CLIP" --out "$T2/out"

python "$REPO/fusion/run_rtmpose_2d.py" --video "$T2/input/left_eye.mp4" \
    --boxes npz --boxes-npz "$T2/out/${CLIP}_3d_keypoints.npz" \
    --rtmpose-config "$RTM_CFG" --rtmpose-checkpoint "$RTM_CKPT" \
    --stem "$CLIP" --out "$T2/out"

# ---------------------------------------------------------------- both fusions off that one pass
python "$REPO/fusion/fuse_egoforce_rtmpose.py" \
    --egoforce "$T2/out/${CLIP}_3d_keypoints.npz" --rtmpose "$T2/out/${CLIP}_2d_keypoints.npz" \
    --mode articulated --min-kpt-conf "${MIN_KPT_CONF:-0.20}" --stem "$CLIP" --out "$T2/out"

cp "$T2/out/${CLIP}_3d_keypoints.npz" "$T2/out/${CLIP}_3d_meta.json" "$T1/out/"
python "$REPO/fusion/fuse_egoforce_rtmpose.py" \
    --egoforce "$T1/out/${CLIP}_3d_keypoints.npz" \
    --mode egoforce-only --stem "$CLIP" --out "$T1/out"

# ---------------------------------------------------------------- QC gate, before anything leaves
python - "$T2/out/${CLIP}_3d_meta.json" "$T2/out/${CLIP}_fuse_stats.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1])); fuse = json.load(open(sys.argv[2]))
r = meta.get('reproj_median_px')
print(f"  reproj_median_px   = {r}")
print(f"  frac_below_5cm     = {(meta.get('depth_Z_m') or {}).get('frac_below_5cm')}")
print(f"  head_vs_lift_px    = {meta.get('head_vs_lift_median_px')}")
print(f"  refit              = {(fuse.get('refit') or {}).get('reasons')}")
if r is None or r >= 1.0:
    sys.exit(f"ABORT: reproj_median_px={r} (need < 1.0)")
reasons = (fuse.get('refit') or {}).get('reasons') or {}
if reasons.get('total') and not reasons.get('accepted'):
    sys.exit("ABORT: 0 refits accepted - this is type 1 wearing a type 2 label")
print("  QC OK")
PY

# ---------------------------------------------------------------- stabilise both, CONCURRENTLY
export CLIP CLIP_FOOT="${CLIP_FOOT:-0}"
export EVAL_RTMPOSE="${EVAL_RTMPOSE:-$T2/out/${CLIP}_2d_keypoints.npz}"
pids=()
for SRC in type1_egoforce_only type2_egoforce_rtmpose; do
    ( SRC_RUN="$SRC" DST_RUN="${SRC}_v3" bash "$REPO/run/run_stabilised.sh" ) &
    pids+=($!)
    echo "  stabilise ${SRC} -> ${SRC}_v3  (pid ${pids[-1]})"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[[ $rc -eq 0 ]] || { echo "a stabilise stage failed (rc=$rc)" >&2; exit $rc; }

echo
echo "delivered ${CLIP}: type1_egoforce_only_v3 and type2_egoforce_rtmpose_v3"
echo "raw inference kept locally at $WORK (not uploaded)"
