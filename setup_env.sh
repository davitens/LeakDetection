#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create the env
cd /workspace
uv venv --python 3.11 --clear
source .venv/bin/activate
apt-get update && apt-get install -y python3-dev
uv pip install autogluon
git clone https://github.com/autogluon/tabarena.git
cd tabarena/
uv pip install --prerelease=allow -e ./tabarena[benchmark]
cd ..
cd LeakDetection
uv pip install -r requirements.txt
cd /workspace
mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
rm -rf tabarena

# Credentials for private Hugging Face / GitHub access.
# Never paste a token into this file: it is tracked by git. Export it instead:
#   export HF_TOKEN=...   (and use `huggingface-cli login`)
#   export GH_TOKEN=...   (or `gh auth login`)
git config --global include.path /workspace/.gitconfig
git config --file /workspace/.gitconfig credential.helper 'store --file /workspace/.git-credentials'


# LFS stuff
# 1. Ensure LFS is initialized in the environment
git lfs install

# 2. Pull the actual 400MB+ files from GitHub's LFS storage
git lfs pull