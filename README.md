# FinalProject
Music Composer
Python 3.11.9
Py -3.11
pip install tensorflow==2.13.0
pip install keras==2.13.1
pip install numpy>=1.22.4
………………………………………………
Install NVIDIA Driver
Make sure your GPU driver is up to date (check using nvidia-smi). It should support CUDA 11.8.

2. Install CUDA Toolkit 11.8
Download from NVIDIA official site:
🔗 https://developer.nvidia.com/cuda-11-8-0-download-archive

Install it and ensure the installer adds the CUDA bin and libnvvp directories to your system's PATH.

3. Install cuDNN 8.6 for CUDA 11.x
Download from NVIDIA:
🔗 https://developer.nvidia.com/rdp/cudnn-archive

You’ll need an NVIDIA Developer account (free). Download the cuDNN 8.6 version for CUDA 11.x and copy the contents (bin, include, lib) into your CUDA installation directory, e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\.
…………………………………….
Number of Layers:
num_layers=2
model has 2 layers in both the encoder and decoder.

Embedding Dimension (d_model):
d_model=64
The embedding dimension for the model is 64.

Number of Attention Heads:
num_heads=2
Each multi-head attention mechanism has 2 heads.

Feedforward Network Dimension:
d_feedforward=128
The feedforward network in each layer has a hidden dimension of 128.
Summary:
Number of Layers: 2
Embedding Dimension: 64
Number of Attention Heads: 2
Feedforward Network Dimension: 128

