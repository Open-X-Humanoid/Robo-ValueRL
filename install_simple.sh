#!/bin/bash

# create conda environment
conda create -n robo_valuerl python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate robo_valuerl

cd lerobot_2_1
pip install -e .
cd ..

cd robo_valuerl
pip install -r requirements.txt
cd ..


cd openpi


pip install -r requirements.txt
pip install -e .

pip list | grep openpi


pip install datasets==3.6.0
pip install albumentations
pip install matplotlib
pip install torchcodec==0.4.0
# get site-packages path and copy transformers files
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TRANSFORMERS_TARGET="$SITE_PACKAGES/transformers"

if [ -d "$TRANSFORMERS_TARGET" ]; then
    echo "Copying transformers files to $TRANSFORMERS_TARGET"
    cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_TARGET/"
    echo "Installation completed successfully!"
else
    echo "Error: transformers not found. Please install it first."
    exit 1
fi
