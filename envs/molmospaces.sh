#!/bin/bash
# Install MolmoSpaces as its README says, in the benchmark-molmospaces env. Run from the repository root:
#     bash envs/molmospaces.sh
set -e
[ -d benchmarks/molmospaces ] || git clone --branch v0.2.9 https://github.com/allenai/molmospaces.git benchmarks/molmospaces
conda create -y -n benchmark-molmospaces python=3.11
pip="conda run --no-capture-output -n benchmark-molmospaces pip install"
# MolmoSpaces leaves mujoco-warp unpinned; 4.x needs MuJoCo 4 (mujoco-core), which clashes with its mujoco 3.5
$pip -e "benchmarks/molmospaces[mujoco]" mujoco-warp==3.6.0
$pip -r envs/client.txt
conda env config vars set -n benchmark-molmospaces \
MLSPACES_ASSETS_DIR="$PWD/benchmarks/molmospaces_assets" MLSPACES_CACHE_DIR="$PWD/benchmarks/molmospaces_cache"
[ "$(uname)" = Linux ] && conda env config vars set -n benchmark-molmospaces MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# download the assets and scenes (~13 GB), once
mkdir -p benchmarks/molmospaces_assets benchmarks/molmospaces_cache
conda run -n benchmark-molmospaces python -m molmo_spaces.molmo_spaces_constants
