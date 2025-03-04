# Create a new conda environment for voodoo3d
ml load CUDA/12.1.1
ml proxy

conda create -n voodoo3d python=3.10 pytorch=2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda activate voodoo3d
pip install -r requirements.txt

# pip install gdown
# pip install onedrive-d

# download pretrained models
# cd pretrained_models

# gdown https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz

pip install numpy==1.26.4
pip install -e ../expdata