import streamlit as st
import subprocess
import sys

# Function to install required packages
def install_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        st.success("Required packages installed successfully!")
    except subprocess.CalledProcessError:
        st.error("Failed to install required packages. Please check your internet connection and try again.")
        st.stop()

# Check if required packages are installed
try:
    import torch
    import cv2
    import numpy as np
    import albumentations as A
    from PIL import Image
    import matplotlib.pyplot as plt
    import segmentation_models_pytorch as smp
except ImportError:
    st.warning("Some required packages are missing. Installing them now...")
    install_packages()
    st.experimental_rerun()

import torch.nn as nn
import time

# Rest of your Streamlit app code goes here
# (Include all the code from the previous version, starting from the title)

st.title('Road Segmentation Model')

# Model parameters
IMG_SIZE = 512
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

# Define the model architecture
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images):
        return self.backbone(images)

# ... (rest of your code)
