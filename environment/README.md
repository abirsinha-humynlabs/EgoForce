# Resolved environment for EgoForce

The `egoforce` conda env as it actually existed on the run machine, captured with
`conda env export` / `pip freeze` before teardown. `scripts/requirements.txt` declares
dependencies; this pins the working resolution.

Python 3.10.21, torch 2.8.0+cu126, on an NVIDIA A10G with driver 615.71.09.

    conda env create -f env/egoforce.conda.yml

This env runs the fusion, the renderer (`viz_delivery/render_v4_panels.py`) and the scorer
(`fusion/evaluate_run.py`). HaWoR inference runs in a *separate* env on torch 1.13.0+cu117;
the delivery scripts activate each in turn. Do not try to merge the two.

`egoforce.pip.txt` is kept for the dependencies installed from git, whose exact commits appear
nowhere else:

- `pytorch3d @ git+.../pytorch3d.git@0a7d4c1a`
- `anycalib @ git+.../AnyCalib.git@027a8497`
- `chumpy @ git+.../chumpy.git@580566ea`

Two entries point at vendored source inside this repo and are editable installs, so they
rebuild from the tree rather than from an index: `datapipes` (`thirdparty/datapipes`) and
`mmdet` (`thirdparty/mmdetection`). Entries of the form
`pkg @ file:///home/conda/feedstock_root/...` are conda packages that `pip freeze` renders as
local paths — install those from the conda yml, not from the pip file.

One quirk the delivery scripts work around: this env's
`activate.d/activate-gcc_linux-64.sh` dereferences an unset `SYS_SYSROOT`, so any script that
activates it must not run under `set -u`. See `delivery/v1v2/` in the HaWoR repo.
