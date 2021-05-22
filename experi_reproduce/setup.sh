############################################
# step 1 : build pytorch v1.3.0 from source
############################################
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda101 # or [magma-cuda92 | magma-cuda100 ] depending on your cuda version
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.3.0
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install


