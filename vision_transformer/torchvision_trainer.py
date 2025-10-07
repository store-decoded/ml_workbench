from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from vit_pytorch.efficient import ViT
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms as v2
from torchvision.transforms import Compose, ColorJitter, ToTensor,Grayscale,Normalize,Resize,Lambda
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights,ResNet34_Weights
import os
jitter = Compose(
        [

            ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.5),
            Grayscale(num_output_channels=1),
            Resize(size=(256, 256), antialias=True),
            ToTensor(),
            Lambda(lambda x: x.repeat(3,1,1)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
print("CUDA AVAILABEL",torch.cuda.is_available())

model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

# Hyperparameters:
batch_size = 100
epochs = 6
lr = 3e-5
gamma = 0.7
seed = 142
IMG_SIZE = 224
patch_size = 16
num_classes = 89



train_ds = torchvision.datasets.ImageFolder("./datasets/split_data/train/", transform=jitter)
valid_ds = torchvision.datasets.ImageFolder("./datasets/split_data/validation/", transform=jitter)
test_ds = torchvision.datasets.ImageFolder("./datasets/split_data/test/", transform=jitter)


# Data Loaders:
train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
valid_loader = data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

 

# Training device:
device = 'cuda'


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
# Linear Transformer:
efficient_transformer = Linformer(dim=128, seq_len=64+1, depth=12, heads=8, k=64)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Determine whether to use GPU (if available) or CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    best_val_acc = 0.0
    model.to(device)
    ckpt_dir = "models/resnet34"
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize running loss and correct predictions count for training
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data loader
        for inputs, labels in tqdm(train_loader):
            # Move inputs and labels to the device (GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset the gradients to zero before the backward pass
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)
            # Get the predicted class (with the highest score)
            _, preds = torch.max(outputs, 1)
            # Compute the loss between the predictions and actual labels
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()
            # Perform the optimization step to update model parameters
            optimizer.step()

            # Accumulate the running loss and the number of correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Compute average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        # Set the model to evaluation mode for validation
        model.eval()
        # Initialize running loss and correct predictions count for validation
        running_loss = 0.0
        running_corrects = 0

        # Disable gradient computation for validation (saves memory and computations)
        with torch.no_grad():
            # Iterate over the validation data loader
            for inputs, labels in val_loader:
                # Move inputs and labels to the device (GPU or CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass: compute the model output
                outputs = model(inputs)
                # Get the predicted class (with the highest score)
                _, preds = torch.max(outputs, 1)
                # Compute the loss between the predictions and actual labels
                loss = criterion(outputs, labels)

                # Accumulate the running loss and the number of correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Compute average validation loss and accuracy for this epoch
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.float() / len(val_loader.dataset)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, os.path.join(ckpt_dir, "last.pth"))
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, os.path.join(ckpt_dir, "best.pth"))

        # Print the results for the current epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

if __name__ == "__main__":
    model = model.to(device)
    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=epochs)

