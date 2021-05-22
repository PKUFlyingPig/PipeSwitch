echo "========================================="
echo "step 1 : build pytorch v1.3.0 from source"
echo "========================================="
cd ..
REPO_DIR=$PWD
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda101 
cd ~
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.3.0
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
cd $REPO_DIR

echo "=================================================="
echo "step 2 : Copy modified files to the PyTorch folder"
echo "=================================================="
cd pytorch_plugin
bash overwrite.sh ~/pytorch
cd $REPO_DIR

echo "====================================="
echo "step 3 : Compile the modified PyTorch"
echo "====================================="
cd ~/pytorch
python setup.py install
cd $REPO_DIR

echo "==============================================="
echo "step 4 : add the path to the repo to PYTHONPATH"
echo "==============================================="
export PYTHONPATH=$REPO_DIR:$PYTHONPATH

echo "==================================="
echo "step 5 : install other dependencies"
echo "==================================="
