# 1. Create a fresh virtual environment using uv
# The --clear flag ensures you start from scratch
uv venv --clear

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install the main AutoGluon library
uv pip install autogluon

# 4. Clone the TabArena repository
git clone https://github.com/autogluon/tabarena.git
cd tabarena/

# 5. Install TabArena in editable mode with benchmark dependencies
# Note: This uses the --prerelease=allow flag as seen in your logs
uv pip install --prerelease=allow -e ./tabarena[benchmark]

6. # Install requirements
cd ..
uv pip install -r requirements.txt

# 7. Delete unused folders
rm -rf tabarena



# When resetting runpod:
apt update && apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.12 python3.12-venv python3.12-dev
# Now your venv at /workspace/venv will work again!
source /workspace/venv/bin/activate

git config --global credential.helper store

hf auth login
***REMOVED-LEAKED-TOKEN***