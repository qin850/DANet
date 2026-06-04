# DANet
A network is proposed for pest classification in zero-sample scenarios.
# Download Pre-trained Weights
# ImageNet-1k Pre-trained Backbone
# We use ImageNet-1k pre-trained weights for the ViT backbone at 224 × 224 resolution.
ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
If needed, please download the corresponding pre-trained backbone before training or evaluation.
# Datasets
DANet is evaluated on the IP102 pest dataset, which contains 102 pest categories commonly observed in agricultural environments.

## Repository Structure

├── README.md
├── models/
├── datasets/
├── data/
├── split.py
├── train.py
└── test.py

Usage
Training
python train.py
Testing
python test.py
