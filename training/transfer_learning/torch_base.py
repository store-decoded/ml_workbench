import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint



class BakeryTransferClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Pre-trained Model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 2. Freeze Base Layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Replace the Classification Head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only train the new fully connected layer
        return torch.optim.Adam(self.model.fc.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Batch can be just images, or (images, paths)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
            
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

if __name__ == '__main__':
    # 1. Data Preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='./dataset/transfer_learning', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # Save class names mapping for later use
    with open('class_names.txt', 'w') as f:
        f.write(','.join(train_dataset.classes))

    # 2. Initialize Model
    model = BakeryTransferClassifier(num_classes=len(train_dataset.classes))

    # 3. Checkpoint Setup
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints_tl',
        filename='resnet50-transfer-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',
        save_top_k=1
    )

    # 4. Train
    trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback], accelerator='auto')
    trainer.fit(model, train_dataloaders=train_loader)
