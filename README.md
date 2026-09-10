<h1 align="center">
  <img src="assets/images/ego_glasses.png" alt="EgoForce glasses" height="58">
  EgoForce
  <img src="assets/images/force_hand.png" alt="EgoForce hand" height="58">
</h1>

<h2 align="center">
  <strong>Forearm-Guided Camera-Space 3D Hand Pose from a Monocular Egocentric Camera</strong>
</h2>

<p align="center">
  Christen Millerdurai<sup>1</sup>,
  Shaoxiang Wang<sup>1,2</sup>,
  Yaxu Xie<sup>1</sup>,
  Vladislav Golyanik<sup>3</sup>,
  Didier Stricker<sup>1,2</sup>,
  Alain Pagani<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>German Research Center for Artificial Intelligence (DFKI)
  &nbsp;|&nbsp;
  <sup>2</sup>Rhineland-Palatinate Technical University of Kaiserslautern-Landau (RPTU)
  &nbsp;|&nbsp;
  <sup>3</sup>Max Planck Institute for Informatics (MPII)
</p>

<p align="center">
  <strong><a href="https://www.siggraph.org/">ACM SIGGRAPH Conference Proceedings, 2026</a></strong>
</p>

<p align="center">
  <a href="https://dfki-av.github.io/EgoForce">Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2605.12498">arXiv</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/dfki-av/EgoForce/tree/main">Code</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/datasets/chris10/EgoForce">Data</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/chris10/EgoForce">Demo</a>
</p>

## Official PyTorch implementation

<p align="center">
<img src="assets/images/teaser.gif" alt="EgoForce" height="720"  /></br>
</p>


### Abstract

Reconstructing the absolute 3D pose and shape of the hands from the user’s viewpoint using a single head-mounted camera is crucial for practical egocen- tric interaction in AR/VR, telepresence, and hand-centric manipulation tasks, where sensing must remain compact and unobtrusive. While monocular RGB methods have made progress, they remain constrained by depth–scale am- biguity and struggle to generalize across the diverse optical configurations of head-mounted devices. As a result, models typically require extensive training on device-specific datasets, which are costly and laborious to ac- quire. This paper addresses these challenges by introducing EgoForce , a monocular 3D hand reconstruction framework that recovers robust, absolute 3D hand pose and its position from the user’s (camera-space) viewpoint. EgoForce operates across fisheye, perspective, and distorted wide-FOV camera models using a single unified network. Our approach combines a differentiable forearm representation that stabilizes hand pose, a unified arm–hand transformer that predicts both hand and forearm geometry from a single egocentric view, mitigating depth–scale ambiguity, and a ray space closed-form solver that enables absolute 3D pose recovery across diverse head-mounted camera models. Experiments on three egocentric benchmarks show that EgoForce achieves state-of-the-art 3D accuracy, reducing camera- space MPJPE by up to 28% on the HOT3D dataset compared to prior methods and maintaining consistent performance across camera configurations.


### EgoForce

<p align="center">
  <img src="assets/images/pipeline.png" alt="EgoForce pipeline" width="980">
</p>


<p align="center">
EgoForce processes a monocular egocentric RGB frame by extracting hand and forearm crops, tokenizing them, and conditioning the features on crop intrinsics (CIT). A transformer jointly infers hand–arm features to predict 2D keypoints (with confidences) and root-relative 3D hand and arm poses, which are lifted to camera-space meshes via the ray space solver. When the forearm is out of view, arm tokens are replaced with missing-arm tokens, and a hand-conditioned variational prior infers a plausible arm representation. We apply this workflow independently to the left and right hand-forearm crops.
</p>


## Usage


### Installation

#### 1. Create the environment

The install script targets a Conda environment named `egoforce` and installs the CUDA 12.6, PyTorch 2.8, TensorRT, MMCV, AnyCalib, PyTorch3D, and Project Aria dependencies used by the repo.

```bash
conda create -n egoforce python=3.10 -y
conda activate egoforce
bash scripts/install.sh
```

#### 2. Download model weights

The model weights, detector checkpoints, MANO files, and demo assets expected by `settings.py` live under the repo-local `_DATA/` directory.

```bash
bash scripts/download_model_weights.sh
```

By default, the main checkpoint path is [settings.py](settings.py#L28):

```python
config.POSE_3D.CHECKPOINT_PATH = os.path.join(_DATA_DIR, 'model_weights.pth')
```

#### 3. Download datasets

The dataset downloader clones the Hugging Face dataset repo with git-lfs and writes it to:

```text
<data-root>/EgoForce
```

You must pass the destination explicitly:

```bash
bash scripts/download_datasets.sh --data-root /path/to/datasets
```

After download, update [settings.py](settings.py#L12) so `config.DATASET.DIR` points to your dataset root with a trailing slash, for example:

```python
config.DATASET.DIR = "/path/to/datasets/"
```

The repo then resolves the dataset folders as:

- `EgoForce/HOT3D`
- `EgoForce/ARCTIC`
- `EgoForce/H2O`

#### 4. Verify the key paths

Before running experiments, make sure these paths exist:

- Data root: `_DATA/`
- datasets root: `config.DATASET.DIR + "EgoForce/..."`

### Evaluation

#### 1. Save predictions

The main entrypoint is [experiments/save_predictions.py](experiments/save_predictions.py). It runs EgoForce on a dataset split and saves a pickle file under `_DATA/predictions/`.

Supported datasets are:

- `ARCTIC`
- `H2O`
- `HO3D`
- `HOT3D`
- `HOT3D_PINHOLE`
- `HOT3D_EQUISOLID`
- `HOT3D_EQUIRECTANGULAR`
- `HOT3D_STEREOGRAPHIC`

Example:

```bash
python experiments/save_predictions.py \
  --test-dataset-name ARCTIC \
  --checkpoint-path _DATA/model_weights.pth
```

Common ablation and variant flags:

- `--no-undistort-inp`
- `--no-cit`
- `--no-arm-prior`
- `--no-arm-input`
- `--anycalib-624`
- `--anycalib-pin`
- `--depth-model`
- `--dgp-model`

Prediction files are written as:

```text
_DATA/predictions/<DATASET>_<suffix>_predictions.pkl
```

#### 2. Evaluate saved predictions

[experiments/evaluate_predictions.py](experiments/evaluate_predictions.py) reads the saved prediction PKLs, applies the matching suffix logic, and writes evaluation summaries under `results/OURS/`.

Example:

```bash
python experiments/evaluate_predictions.py \
  --test-dataset-name ARCTIC
```

If you evaluated a specific variant, pass the same flags used during prediction generation so the script resolves the correct suffix:

```bash
python experiments/evaluate_predictions.py \
  --test-dataset-name HOT3D \
  --no-cit
```

Useful options:

- `--disable-kalman-filter` disables translation smoothing. Kalman filtering is enabled by default.
- `--results-root <dir>` changes the output root from `results/`.

#### 3. Intrinsics robustness on HOT3D

[experiments/save_noisy_intrinsic_predictions.py](experiments/save_noisy_intrinsic_predictions.py) runs a HOT3D-only camera-noise sweep. It first estimates first-frame AnyCalib intrinsics, then evaluates multiple intrinsic noise levels and stores both prediction caches and camera-noise analysis artifacts.

```bash
python experiments/save_noisy_intrinsic_predictions.py
```

Optional controls:

- `--no-cit`
- `--ray-grid-size`
- `--radial-bins`
- `--force-recompute`
- `--noisy-predictions-dir <dir>`

This script writes noisy prediction PKLs, camera-noise analysis PKLs, AnyCalib intrinsics JSON files, and plots under `_DATA/noisy_predictions/`.

To aggregate the robustness results, run [experiments/evaluate_noisy_intrinsic_predictions.py](experiments/evaluate_noisy_intrinsic_predictions.py):

```bash
python experiments/evaluate_noisy_intrinsic_predictions.py
```

The default output directory is:

```text
results/intrinsics_robustness
```

#### 4. Hand-scale analysis

[experiments/evaluate_hand_scale.py](experiments/evaluate_hand_scale.py) evaluates hand-scale consistency and calibration behavior from prediction PKLs. It can auto-discover predictions under `_DATA/predictions/` by suffix, or you can pass files explicitly.

Auto-discovery example:

```bash
python experiments/evaluate_hand_scale.py --suffix undistort_inp_true
```

Explicit-file example:

```bash
python experiments/evaluate_hand_scale.py \
  --hot3d-predictions _DATA/predictions/HOT3D_undistort_inp_true_predictions.pkl \
  --arctic-predictions _DATA/predictions/ARCTIC_undistort_inp_true_predictions.pkl
```

By default, the script writes CSV summaries, plots, and a text report to:

```text
results/hand_scale_eval/<suffix>/
```

#### 5. Visibility-bin forearm ablation

[experiments/hand_joint_occlusion_graph.py](experiments/hand_joint_occlusion_graph.py) compares ARCTIC predictions with and without forearm input, grouped by hand-joint visibility.

It expects these two prediction files to exist in `_DATA/predictions/`:

- `ARCTIC_undistort_inp_true_predictions.pkl`
- `ARCTIC_undistort_inp_true_no_arm_input_predictions.pkl`

Run:

```bash
python experiments/hand_joint_occlusion_graph.py
```

Artifacts are written under:

```text
results/hand_joint_occlusion_graph/
```

## Demo 

### Gradio video demo

The Gradio app in [demo/run_app.py](demo/run_app.py) runs EgoForce on uploaded videos and shows the output video with the input view, ego-view render, and third-person render.

Start it with:

```bash
python demo/run_app.py
```

Useful launch options:

```bash
python demo/run_app.py --server-name 0.0.0.0 --server-port 7860
python demo/run_app.py --share
```

### Hand landmark video export

[demo/render_landmarks.py](demo/render_landmarks.py) is a headless CLI that runs EgoForce over a
video file and writes two things:

- an overlay mp4 with the 21-joint hand skeleton drawn on the input frames, and
- an `.npz` holding the camera-space 3D joints, their 2D projections and per-hand
  visibility flags for every processed frame.

It needs neither Gradio nor the pytorch3d rasteriser (the overlay is drawn with OpenCV
from `pred_j2d`), so it is the lighter path when you want keypoints rather than meshes.

Minimal run on an uncalibrated video — intrinsics are estimated from the first processed
frame with AnyCalib, exactly as the Gradio demo does:

```bash
python demo/render_landmarks.py \
  --video /path/to/input.mp4 \
  --duration-seconds 10
```

Outputs default to `_DATA/outputs/<video stem>_landmarks.{mp4,npz}`.

If the rig is already calibrated, pass the intrinsics and skip AnyCalib entirely:

```bash
python demo/render_landmarks.py \
  --video /path/to/input.mp4 \
  --camera-model pinhole \
  --focal 736.6 736.6 \
  --principal 960.0 540.0
```

`--camera-model` accepts `pinhole`, `rational8` (8 distortion coefficients) and
`fisheye624` (12 coefficients); pass the coefficients with `--distortion`. Without
`--camera-model`, `--lens` selects the AnyCalib lens model (`fisheye624`,
`pinhole_distortion` or `pinhole`).

Other useful flags:

- `--start-seconds` / `--duration-seconds` / `--max-frames` / `--stride` select the clip.
  Output fps is divided by `--stride`.
- `--draw-forearm` also draws the predicted 3-joint forearm chain.
- `--skip-video` dumps only the keypoint npz.
- `--no-kalman` disables the translation Kalman filter (enabled by default, and its
  frequency is set from the source fps).
- `--no-undistort-inp` matches the `--no-undistort-inp` evaluation ablation.
- `--verbose` lets the per-frame timing prints through.

#### Keypoint npz layout

`N` is the number of processed frames; the hand axis is `['left', 'right']`.

| Key | Shape | Meaning |
| --- | --- | --- |
| `j3d` | `(N, 2, 21, 3)` | Camera-space hand joints, metres |
| `j2d` | `(N, 2, 21, 2)` | Hand joints projected to source-image pixels |
| `arm_j3d` | `(N, 2, 3, 3)` | Camera-space forearm joints, metres |
| `arm_j2d` | `(N, 2, 3, 2)` | Forearm joints in source-image pixels |
| `visible` | `(N, 2)` | Per-hand detection flag |
| `failed` | `(N,)` | True where inference raised on that frame |
| `frame_index` | `(N,)` | Index in the source video |
| `timestamp_s` | `(N,)` | Source timestamp in seconds |
| `joint_names` | `(21,)` | Joint names in output order |
| `skeleton_edges` | `(23, 2)` | Joint-index pairs for drawing |
| `calibration` | scalar | JSON string with the intrinsics actually used |

Frames where a hand is not detected are filled with `NaN` and flagged in `visible`.

The joint order follows `mano_joint_mapping` in [models/mano_layer.py](models/mano_layer.py):
wrist first, then thumb, index, middle, ring and pinky, each running MCP → PIP → DIP → tip.

#### Requirements

This script has the same hard requirements as the rest of the demo — an NVIDIA GPU with
CUDA. `demo/inference.py` imports `torch_tensorrt` and compiles both the detector and
HALO for TensorRT, and [scripts/install.sh](scripts/install.sh) pins CUDA 12.6, so there
is no CPU or Apple-silicon path.

- NVIDIA GPU with a CUDA 12.6 capable driver. The repo does not document a VRAM
  figure. The work per frame is small and fixed — RTMDet-tiny plus a YOLO pose
  detector on the full frame, then HALO on four `224x224` crops (hand and forearm
  for each hand) — so VRAM is dominated by the TensorRT and inductor
  `max-autotune` compilation rather than by inference itself. Anything from
  roughly 12 GB up should be comfortable; treat that as an estimate, not a
  measured requirement.
- The environment from `bash scripts/install.sh` (Python 3.10, PyTorch 2.8 + cu126,
  torch-TensorRT 2.8, mmcv 2.1.0, mmdetection, pytorch3d, AnyCalib).
- Weights from `bash scripts/download_model_weights.sh`, which populates `_DATA/` with
  `model_weights.pth`, the detector checkpoints and the MANO files.
- `ffmpeg` on `PATH` for h264 output. Without it the script falls back to OpenCV's
  `mp4v` encoder.

The first invocation pays a one-off TensorRT/inductor compile before the first frame is
produced.

### Evaluation and 2D/3D fusion (`fusion/`)

The [`fusion/`](fusion) package runs EgoForce as a drop-in replacement for the WiLoR + MediaPipe
stages of an existing egocentric hand-labelling pipeline, and adds a fusion with RTMPose-m Hand5 that
refits MANO articulation against confidence-weighted 2D landmarks instead of rigidly re-posing a
fixed skeleton.

| Document | Read it for |
| --- | --- |
| [`fusion/README.md`](fusion/README.md) | How to run it — stage map and commands |
| [`PIPELINE.md`](PIPELINE.md) | Design and engineering: why it is shaped this way, the refit objective, the data contracts, what was verified how |
| [`plan_of_action.md`](plan_of_action.md) | What is still pending, the GPU requirement, and the known issues that affect the comparison |

`python fusion/selftest.py` runs 34 checks with no GPU, no checkpoints and no MANO files. The
producer stages themselves have the same CUDA requirement as the rest of the demo, and additionally
need `mim install "mmpose>=1.3.2"` plus `bash scripts/download_rtmpose_hand5.sh`.

### Project Aria live demo

The live Aria demo in [demo/run_aria.py](demo/run_aria.py) streams RGB frames from a Project Aria device and runs inference frame by frame. The same entrypoint supports both USB and Wi-Fi streaming.

The Unity visualization project lives in:

```text
unity_rendering/unity_scene
```

The current Unity scene was tested with:

```text
Unity 6000.3.17f1
```

#### Terminal 1: start EgoForce streaming

Activate the environment and move into the demo directory:

```bash
conda activate egoforce
cd /path/to/EgoForce/demo
```

For USB streaming, run:

```bash
python3 run_aria.py
```

For Wi-Fi streaming, pass the interface and the device IP explicitly:

```bash
python3 run_aria.py --interface WifiStation --ip 192.168.88.19
```

For Aria Gen 1, `WifiStation` is the Python SDK streaming interface for routing traffic through a Wi-Fi router. See the Gen 1 docs:

- API reference: `aria.sdk.StreamingInterface.WifiStation` in the [Aria Client SDK API reference](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/api_reference)
- Sample workflow: “Using Wi-Fi” in the [streaming_subscribe Gen 1 sample](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples/streaming_subscribe)

Notes:

- `Usb` is the default interface, so no IP is required for USB mode.
- For `WifiStation`, `--ip` is required.
- The runner uses ephemeral streaming certificates.
- Unity mesh streaming is enabled by default. If you want to run the Aria demo without Unity, set `UNITY_ENABLE=0`.

#### Terminal 2: start Unity

Open the Unity project in a second terminal:

```bash
"/path/to/Unity/Hub/Editor/6000.3.17f1/Editor/Unity" \
  -projectPath "/path/to/EgoForce/unity_rendering/unity_scene" \
  -force-vulkan \
  -logFile /path/to/logs/egoforce_unity_vulkan.log
```

#### Expected runtime flow

1. Start the Aria stream from Terminal 1.
2. Start the Unity project from Terminal 2.
3. The Python process publishes mesh buffers and camera frames to Unity over `tcp://*:5555`.
4. Unity receives the live hand and arm meshes and displays them in the scene.

#### Troubleshooting

- If you run over Wi-Fi, make sure the Aria device is reachable on the selected network and that no VPN or firewall rule is intercepting the traffic.
- If Unity opens for the first time, let it finish package import before expecting live updates.
- Check [Project Aria streaming documentation](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples/streaming_subscribe) for device-side setup details.

## Citation

If you find this code useful for your research, please cite our paper:
```
@inproceedings{millerdurai2026egoforce,
      title={EgoForce: Forearm-Guided Camera-Space 3D Hand Pose from a Monocular Egocentric Camera},
      author={Millerdurai, Christen and Wang, Shaoxiang and Xie, Yaxu and Golyanik, Vladislav and Stricker, Didier and Pagani, Alain},
      booktitle={Proceedings of the SIGGRAPH 2026 Conference Papers},
      year={2026}
}
```
## License

EgoForce is under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. The license also applies to the pre-trained models.
