import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# 1. Define standard PyTorch datasets and dataloaders
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_loader = DataLoader(datasets.ImageFolder('datasets/transfer_learning/train', transform=transform), batch_size=32, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder('datasets/transfer_learning/val', transform=transform), batch_size=32)

# 2. Transfer Learning setup: Load ViT, freeze parameters, replace classifier head
vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

for param in vit_model.parameters():
    param.requires_grad = False

in_features = vit_model.heads.head.in_features
vit_model.heads.head = nn.Linear(in_features, 3)

# 3. Wrap in LightningModule and train
class ViTLightning(pl.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit 
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.vit(x), y)
        return loss
        
    def configure_optimizers(self):
        # Only pass parameters that require gradients (our new head)
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.vit.parameters()), lr=1e-3)

trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(ViTLightning(vit_model), train_loader, val_loader)
