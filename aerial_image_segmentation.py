import sys
sys.path.append('/home/robosobo/ML_code/Road_seg_dataset')
import torch
import cv2
import albumentations as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

import torch.nn as nn

CSV_PATH = '/home/robosobo/ML_code/Road_seg_dataset/train.csv'
DATA_DIR = '/home/robosobo/ML_code/Road_seg_dataset/'

EPOCHS = 25
LR = 0.001
BATCH_SIZE = 8
IMG_SIZE = 512

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

df = pd.read_csv(CSV_PATH)
df.head()

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

def get_valid_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ])

class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = DATA_DIR + row.images
        mask_path = DATA_DIR + row.masks

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

trainset = SegmentationDataset(train_df, augmentations=get_train_augs())
validset = SegmentationDataset(valid_df, augmentations=get_valid_augs())

print(f'Length of trainset: {len(trainset)}')
print(f'Length of validset: {len(validset)}')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

print(f'Total no. of batches in trainloader: {len(trainloader)}')
print(f'Total no. of batches in validloader: {len(validloader)}')

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

    def forward(self, images, masks=None):
        logits = self.backbone(images)

        if masks is not None:
            return logits, DiceLoss(mode='binary')(logits, masks) + nn.BCEWithLogitsLoss()(logits, masks)

        return logits

model = SegmentationModel()

def train_fn(dataloader, model, optimizer):
    model.train()

    total_loss = 0.0

    for images, masks in tqdm(dataloader):
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_fn(dataloader, model):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            logits, loss = model(images, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_loss = np.Inf

for i in range(EPOCHS):
    train_loss = train_fn(trainloader, model, optimizer)
    valid_loss = eval_fn(validloader, model)

    if valid_loss < best_loss:
        torch.save(model.state_dict(), 'best-model.pt')
        print("SAVED MODEL")
        best_loss = valid_loss

    print(f'Epoch: {i+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')


idx = 9

model.load_state_dict(torch.load('/home/robosobo/ML_code/best-model.pt'))
image, mask = validset[idx]

logits_mask = model(image.unsqueeze(0)) # (channel, height, width) --> (b, c, h, w)
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask > 0.5) * 1.0

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title('Ground Truth Mask')
plt.subplot(1, 3, 3)
plt.imshow(pred_mask.squeeze().detach().numpy(), cmap='gray')
plt.title('Predicted Mask')
plt.show()