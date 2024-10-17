# Install SR methods other than PySR and uDSR
### Install libraries
```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential make curl libgsl-dev zlib1g-dev libatlas-base-dev \
    libffi-dev libffi8ubuntu1 libgmp-dev libgmp10 libncurses-dev libncurses5 \
    libtinfo5 autotools-dev libicu-dev libbz2-dev ccache \
    python3-dev python3-distutils libssl-dev libopenblas-dev \
    libblas-dev liblapack-dev zip unzip tar gfortran openjdk-17-jdk
sudo apt clean
```

### Install ghcup
```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
# ghcup tui
source ~/.bashrc
ghcup install cabal 3.10.2.0
ghcup install stack 2.13.1
ghcup set cabal 3.10.2.0
```

### Create Conda Env
```bash
conda env create -f environment.yml
conda activate pansr
```

### ITEA
Need to comment out ghcup install code in `install_ghcup.sh`
```bash
cd ../experiment/methods/src/
git clone https://github.com/folivetti/ITEA.git
cd ITEA

# Using ghcup
sed -i '/cabal install/ s/^/#/' install_ghcup.sh
./install_ghcup.sh
cd ..
```

For RedHat, need to install these libraries first 
```
conda install openblas-devel -c anaconda
```
then add the following to `cabal.project`
```
package hmatrix-morpheus
    extra-lib-dirs: /home/xh29/miniconda3/envs/srbench/lib
    extra-include-dirs: /home/xh29/miniconda3/envs/srbench/include
```


### DSR
```bash
git clone https://github.com/lacava/deep-symbolic-regression
cd deep-symbolic-regression
pip install ./dsr # Install DSR package
cd ..
```

### ellyn
```bash
git clone https://github.com/cavalab/ellyn
cd ellyn
git checkout cdff25b2851d942db1cdb2a6796ea61c41396c7c
python setup.py install
cd ..
```

### feat
```bash
git clone https://github.com/cavalab/feat.git 
cd feat
git checkout tags/0.5.1
python setup.py install
cd ..
```


### GP-GOMEA
```bash
conda install -c conda-forge armadillo=9.900.4 -y

git clone https://github.com/marcovirgolin/GP-GOMEA 
cd GP-GOMEA
# fix version
git checkout 6a92cb671c2772002b60df621a513d8b4df57887

# use correct python version
LIBNAME=$(echo $CONDA_PREFIX/lib/libboost_python*.so)
PYVERSION=${LIBNAME##$CONDA_PREFIX/lib/libboost_python}
PYVERSION=${PYVERSION%.so}
LBOOST="lboost_python$PYVERSION"
LNP="lboost_numpy$PYVERSION"

echo "PYVERSION: ${PYVERSION}"
echo "LBOOST: $LBOOST"
echo "LNUMPY: $LNP"

sed -i "s/lboost_python37/$LBOOST/" Makefile-variables.mk
sed -i "s/lboost_numpy37/$LNP/" Makefile-variables.mk

# add extra flags to varables
echo "EXTRA_FLAGS=-I ${CONDA_PREFIX}/include" >> Makefile-variables.mk
echo "EXTRA_LIB=-L ${CONDA_PREFIX}/lib" >> Makefile-variables.mk
# check
tail Makefile-variables.mk

# append EXTRA_FLAGS and CSSFLAGS
sed -i 's/^CXXFLAGS.*/& \$(EXTRA_FLAGS)/' Makefile-Python_Release.mk
sed -i 's/LIB_BOOST_NUMPY.*/& \$(EXTRA_LIB)/' Makefile-Python_Release.mk
#check
cat Makefile-Python_Release.mk | grep EXTRA_FLAGS
cat Makefile-Python_Release.mk | grep EXTRA_LIB

# todo: add CONDA_PREFIX to Make-file
make 
# copy the .so library into the python package
cp dist/Python_Release/GNU-Linux/gpgomea pyGPGOMEA/gpgomea.so

pip install .
cd ..
```


### Operon
**Add & Update Packages**
```bash
conda install -c conda-forge "cmake>=3.20" "eigen>=3.4.0" "ceres-solver=2.0.0" "cxxopts" "gcc=11.2.0" "gcc_impl_linux-64=11.2.0" "gcc_impl_linux-64=11.2.0" "gxx=11.2.0" "optuna" "pkg-config" -y
```

**Install Operon**
```bash
PYTHON_SITE=${CONDA_PREFIX}/lib/python`pkg-config --modversion python3`/site-packages
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++

## aria-csv
git clone https://github.com/AriaFallah/csv-parser
mkdir -p ${CONDA_PREFIX}/include/aria-csv
cd csv-parser
git checkout 544c764d0585c61d4c3bd3a023a825f3d7de1f31
cp parser.hpp ${CONDA_PREFIX}/include/aria-csv/parser.hpp
cd ..
# rm -rf csv-parser

## doctest
git clone https://github.com/doctest/doctest.git
mkdir -p ${CONDA_PREFIX}/include/doctest
cd doctest
git checkout b7c21ec5ceeadb4951b00396fc1e4642dd347e5f
cp doctest/doctest.h ${CONDA_PREFIX}/include/doctest/doctest.h
cd ..

## eve
git clone https://github.com/jfalcou/eve
mkdir -p ${CONDA_PREFIX}/include/eve
cd eve
git checkout cfcf03be08f99320f39c74d1205d0514e62c3c8e
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEVE_BUILD_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
cd ..
cat > ${CONDA_PREFIX}/lib/pkgconfig/eve.pc << EOF
prefix=${CONDA_PREFIX}/include/eve
includedir=${CONDA_PREFIX}/include/eve
libdir=${CONDA_PREFIX}/lib

Name: Eve
Description: Eve - the Expression Vector Engine for C++20
Version: 2022.03.0
Cflags: -I${CONDA_PREFIX}/include/eve
EOF

## vstat
git clone https://github.com/heal-research/vstat.git
cd vstat
git switch cpp20-eve
git checkout 736fa4802730ac14c2dfaf989a2b9016349c58d3
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
cd ..
# rm -rf vstat

## pratt-parser
git clone https://github.com/foolnotion/pratt-parser-calculator.git
cd pratt-parser-calculator
git checkout a15528b1a9acfe6adefeb41334bce43bdb8d578c
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
cd ..
# rm -rf pratt-parser-calculator

## fast-float
git clone https://github.com/fastfloat/fast_float.git
cd fast_float
git checkout 32d21dcecb404514f94fb58660b8029a4673c2c1
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
cd ..
# rm -rf fast_float

## robin_hood
git clone https://github.com/martinus/robin-hood-hashing.git
cd robin-hood-hashing
git checkout 9145f963d80d6a02f0f96a47758050a89184a3ed
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRH_STANDALONE_PROJECT=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
cd ..
# rm -rf robin-hood-hashing

# operon
git clone https://github.com/heal-research/operon.git
cd operon
# git switch cpp20
git checkout 9d7d410e43d18020df25d6311822be8c3680ac56
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon -- VERBOSE=1
cmake --install build
cd ..
# rm -rf operon

## pyoperon
git clone https://github.com/heal-research/pyoperon.git
cd pyoperon
# git switch cpp20
git checkout e4ef1047240b2df66555ddd54463ab707863aae6
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=x86-64 -mavx2 -mfma" \
    -DCMAKE_INSTALL_PREFIX=${PYTHON_SITE} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t pyoperon_pyoperon -- VERBOSE=1
cmake --install build
cd ..
# rm -rf pyoperon

```

# Install PySR
### Create Conda Env
```bash
conda env create -f pysr_environment.yml
conda activate pansr_pysr
```

### Install julia:
```bash
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc
conda activate pansr_pysr
juliaup add 1.10.0     # for PySR v0.16.9
juliaup default 1.10.0 # for PySR v0.16.9
```

### Install PySR:
```bash
python -c 'import pysr; pysr.install()' 
```

# Install uDSR
### Create Conda Env
```bash
conda env create -f udsr_environment.yml
conda activate pansr_udsr
```

### Install uDSR:
```bash
git clone https://github.com/cavalab/srbench.git
cd srbench
git checkout separate-envs
mv algorithms/uDSR/udsr-competition/ ../../experiment/methods/src/
cd ..
rm -rf srbench
cd ../experiment/methods/src/udsr-competition
pip install -e ./dso # Install DSO package and core dependencies
```