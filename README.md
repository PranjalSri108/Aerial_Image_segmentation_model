# Aerial_Image_segmentation_model

## Project Overview

This project implements a deep learning model for road segmentation using a U-Net architecture. Road segmentation is a crucial task in computer vision, particularly for applications like autonomous driving and urban planning. The goal is to identify and delineate road areas in images.

### Key Features:

1. **Dataset**: Uses a custom dataset of road images and their corresponding segmentation masks.
2. **Model Architecture**: Implements a U-Net model using the EfficientNet-B0 encoder.
3. **Data Augmentation**: Applies various augmentations to increase dataset diversity and model robustness.
4. **Training Pipeline**: Includes functions for training and evaluating the model.
5. **Visualization**: Provides functionality to visualize the original image, ground truth, and predicted segmentation masks.

### Technical Stack:

- PyTorch for deep learning
- Albumentations for image augmentations
- Segmentation Models PyTorch (SMP) for the U-Net architecture
- OpenCV and Matplotlib for image processing and visualization

This project demonstrates the entire workflow of a segmentation task, from data preparation to model training and result visualization. It's designed to be run in a Google Colab environment, making it accessible and easy to execute without extensive local setup.

![alt text](https://github.com/PranjalSri108/Aerial_Image_segmentation_model/blob/main/Streamlit_deployed_app.png?raw=true)
