#!/usr/bin/env bash
# Fetch the RTMPose-m Hand5 config + checkpoint for fusion/run_rtmpose_2d.py.
#
# Two routes are tried, in order:
#
#   1. `mim download mmpose` - the documented mmpose route. Small and fast. Whether the config it
#      writes is self-contained (its `_base_` and `from_file=` references resolved) depends on the
#      installed mim/mmpose versions, so we TEST that the config actually loads before trusting it.
#   2. A sparse shallow clone of the mmpose config tree, so `_base_ = ['../../../_base_/...']` and
#      `from_file='configs/_base_/datasets/coco_wholebody_hand.py'` resolve by being on disk, plus a
#      direct download of the published checkpoint.
#
# Verified against the mmpose repo (September 2026):
#   config     configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py
#   checkpoint rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth  (~55 MB)
#   reported   96.4 PCK@0.2 / 83.9 AUC / 5.06 EPE on Hand5, 256x256 input, 21 keypoints
#
# mmpose itself is NOT installed by scripts/install.sh. Install it first:
#   mim install "mmpose>=1.3.2"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DEST="${1:-${REPO_ROOT}/_DATA/rtmpose_hand5}"

CONFIG_NAME="rtmpose-m_8xb256-210e_hand5-256x256"
CONFIG_REL="configs/hand_2d_keypoint/rtmpose/hand5/${CONFIG_NAME}.py"
CKPT_NAME="rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
CKPT_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/${CKPT_NAME}"

mkdir -p "${DEST}"
echo "Destination: ${DEST}"

config_loads() {
    python - "$1" <<'PY' >/dev/null 2>&1
import sys
from mmengine.config import Config
Config.fromfile(sys.argv[1])
PY
}

# ---------------------------------------------------------------- route 1: mim
if command -v mim >/dev/null 2>&1; then
    echo "[1/2] trying: mim download mmpose --config ${CONFIG_NAME}"
    if mim download mmpose --config "${CONFIG_NAME}" --dest "${DEST}"; then
        CANDIDATE="${DEST}/${CONFIG_NAME}.py"
        if [[ -f "${CANDIDATE}" ]] && config_loads "${CANDIDATE}"; then
            echo
            echo "OK (mim). Use:"
            echo "  --rtmpose-config     ${CANDIDATE}"
            echo "  --rtmpose-checkpoint ${DEST}/${CKPT_NAME}"
            exit 0
        fi
        echo "    mim's config did not load standalone; falling back to the config tree."
    else
        echo "    mim download failed; falling back."
    fi
else
    echo "[1/2] mim not on PATH (it ships with openmim, already in scripts/requirements.txt)."
fi

# ---------------------------------------------------------------- route 2: sparse clone + curl
SRC="${DEST}/mmpose_src"
echo "[2/2] sparse-cloning the mmpose config tree into ${SRC}"
if [[ ! -d "${SRC}/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/open-mmlab/mmpose.git "${SRC}"
    git -C "${SRC}" sparse-checkout set configs
else
    echo "    already cloned; pulling"
    git -C "${SRC}" pull --ff-only || true
fi

if [[ ! -f "${SRC}/${CONFIG_REL}" ]]; then
    echo "ERROR: ${CONFIG_REL} not found in the clone. The upstream layout may have changed; run" >&2
    echo "       mim search mmpose --model rtmpose --valid-field task,config" >&2
    exit 1
fi

if [[ ! -f "${DEST}/${CKPT_NAME}" ]]; then
    echo "    downloading ${CKPT_NAME}"
    curl -fL --retry 3 -o "${DEST}/${CKPT_NAME}.part" "${CKPT_URL}"
    mv "${DEST}/${CKPT_NAME}.part" "${DEST}/${CKPT_NAME}"
else
    echo "    checkpoint already present"
fi

echo
echo "OK (config tree). Use:"
echo "  --rtmpose-config     ${SRC}/${CONFIG_REL}"
echo "  --rtmpose-checkpoint ${DEST}/${CKPT_NAME}"
echo
echo "Note: fusion/run_rtmpose_2d.py defaults to ${DEST}/${CONFIG_NAME}.py, which only exists on the"
echo "mim route. On this route pass --rtmpose-config explicitly (or symlink it)."
