# Setup PULP-RISC-V-GCC-TOOLCHAIN
wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
tar -xvf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2 -C ./
mv v1.0.16-pulp-riscv-gcc-ubuntu-18 pulp-riscv-gcc-toolchain
rm v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
rm -rf v1.0.16-pulp-riscv-gcc-ubuntu-18
git config --global --add safe.directory /workspaces/APAI-Docker/pulp-sdk

# Build PULP-SDK
cd pulp-sdk/ 
git submodule update --init --recursive
cd ..
CURR_PATH=$(pwd)
USR_PATH=$(pwd)/pulp-riscv-gcc-toolchain
export PULP_RISCV_GCC_TOOLCHAIN=$USR_PATH
export PATH=$USR_PATH/bin:$PATH
source pulp-riscv-gcc-toolchain/sourceme.sh
source pulp-sdk/configs/pulp-open.sh
cd pulp-sdk/
make build
cd ..
