pip install alfworld==0.3.3
pip uninstall opencv-python -y
pip install -e .
export ALFWORLD_DATA=/data/code/exp/alfworld
alfworld-download
