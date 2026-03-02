#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "======================================"
echo "🚀 AutoGluon + TabArena Environment Setup"
echo "======================================"

# ---------------------------------------
# 1️⃣  Create a fresh virtual environment
# ---------------------------------------
echo "Creating fresh virtual environment with uv..."
uv venv --clear

# ---------------------------------------
# 2️⃣  Activate virtual environment
# ---------------------------------------
echo "Activating virtual environment..."
source .venv/bin/activate

# ---------------------------------------
# 3️⃣  Install AutoGluon
# ---------------------------------------
echo "Installing AutoGluon..."
uv pip install autogluon

# ---------------------------------------
# 4️⃣  Clone TabArena repository
# ---------------------------------------
echo "Cloning TabArena repository..."
git clone https://github.com/autogluon/tabarena.git
cd tabarena/

# ---------------------------------------
# 5️⃣  Install TabArena (editable + benchmark deps)
# ---------------------------------------
echo "Installing TabArena with benchmark dependencies..."
uv pip install --prerelease=allow -e ./tabarena[benchmark]

# ---------------------------------------
# 6️⃣  Install additional requirements
# ---------------------------------------
echo "Installing additional requirements..."
cd ..
uv pip install -r requirements.txt

# ---------------------------------------
# 7️⃣  Cleanup
# ---------------------------------------
echo "Removing unused TabArena clone..."
rm -rf tabarena

echo "======================================"
echo "✅ Main setup complete"
echo "======================================"

# =====================================================
# 🔁 RunPod Reset Section (Only if environment breaks)
# =====================================================
echo "If resetting RunPod environment..."

apt update && apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.12 python3.12-venv python3.12-dev

echo "Reactivating workspace virtual environment..."
source /workspace/venv/bin/activate

echo "Configuring Git credential storage..."
git config --global credential.helper store

echo "Logging into Hugging Face..."
hf auth login

# Paste your HF token manually when prompted
# Example:
# ***REMOVED-LEAKED-TOKEN***