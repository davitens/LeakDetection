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

# To store credentials
git config --global include.path /workspace/.gitconfig
git config --file /workspace/.gitconfig credential.helper 'store --file /workspace/.git-credentials'



# Paste your HF token manually when prompted
# Example:
# git config --global credential.helper store

# ***REMOVED-LEAKED-TOKEN***
# ***REMOVED-LEAKED-TOKEN***