import streamlit as st
import torch
import cv2
import numpy as np
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
import time

# Streamlit app title
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

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = SegmentationModel()
    model.load_state_dict(torch.load('best-model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    augmented = aug(image=image)
    image = augmented['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    return image.unsqueeze(0)

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Add a button to generate road map
    if st.button('Generate Road Map'):
        # Show loading spinner
        with st.spinner('Generating road map...'):
            # Simulate some processing time
            time.sleep(2)

            # Preprocess the image
            image_np = np.array(image)
            preprocessed_image = preprocess_image(image_np)

            # Make prediction
            with torch.no_grad():
                logits = model(preprocessed_image)
                pred_mask = torch.sigmoid(logits)
                pred_mask = (pred_mask > 0.5).float()

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            # Display original image in the first column
            with col1:
                st.image(image, caption='Original Image', use_column_width=True)

            # Display predicted road map in the second column
            with col2:
                st.image(pred_mask.squeeze().cpu().numpy(), caption='Predicted Road Map', use_column_width=True, clamp=True)

        st.success('Road map generated successfully!')

else:
    st.write("Please upload an image to get started.")
