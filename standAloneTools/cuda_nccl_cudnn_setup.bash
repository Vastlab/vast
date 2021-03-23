#!/bin/bash

RSYNCALIAS=$1
ALTCP=${2:-adhamija}

touninstall=(
    "*cublas*"
    "cuda*"
    "nsight*"
    "nvidia*"
    "*nvidia*"
    "nccl*"
    "cudnn*"
)

cuda_install_file="cuda_11.1.1_455.32.00_linux.run"
toinstall=(
    "libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb"
    "nccl-local-repo-ubuntu1804-2.8.4-cuda11.1_1.0-1_amd64.deb"
)

RED='\033[0;31m'
NC='\033[0m' # No Color
packages_path="/scratch/"$ALTCP"/cuda_packages/"
mkdir -p $packages_path
rsync -zarvh $RSYNCALIAS@quicksilver.vast.uccs.edu:$packages_path $packages_path
cd $packages_path

CUDA_UNINSTALLER=/usr/local/cuda/bin/cuda-uninstaller
if test -f "$CUDA_UNINSTALLER"; then
    echo -e "${RED}Uninstalling CUDA using $CUDA_UNINSTALLER${NC}"
    sh $CUDA_UNINSTALLER
else
    echo -e "${RED}Could not find uninstall script at $CUDA_UNINSTALLER${NC} press enter to continue"
    read varname
fi

echo -e "${RED}Uninstalling package ${touninstall[@]}${NC}"
apt-get -qq --yes --purge remove "${touninstall[@]}"
apt-get -qq --yes autoremove
apt-get -qq --yes autoclean
rm -rf /usr/local/cuda*

echo -e "${RED}Installing cuda using $cuda_install_file${NC}"
sh $cuda_install_file

echo -e "${RED}Installing packages ${toinstall[@]}${NC}"
dpkg -i "${toinstall[@]}"

# Enable persistence mode after installation has completed.
echo -e "${RED}Enabling persistence mode${NC}"

NVIDIAPATH=/usr/share/doc/NVIDIA_GLX-1.0/samples/
tar -xf $NVIDIAPATH/nvidia-persistenced-init.tar.bz2 -C /tmp/
mv /tmp/nvidia-persistenced-init $NVIDIAPATH
sh $NVIDIAPATH/nvidia-persistenced-init/install.sh

# Check if script installed succesfully
if [ $? -eq 0 ]; then
        echo "${RED}Persistence mode succesfully enabled.${NC}"
else
        echo "${RED}Persistence mode installation failed, reloading daemons${NC}"
        systemctl daemon-reload
        sh $NVIDIAPATH/nvidia-persistenced-init/install.sh
fi

