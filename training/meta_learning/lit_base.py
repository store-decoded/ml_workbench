import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models, transforms
from PIL import Image

# --- 1. Dataset for Episodic Training ---
class EpisodicDataset(Dataset):
    def __init__(self, root_dir, n_way, k_shot, q_size, episodes):
        self.root_dir = root_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_size = q_size
        self.episodes = episodes
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        sampled_classes = random.sample(self.classes, self.n_way)
        support_imgs, query_imgs, query_targets = [], [], []

        for c_idx, c_name in enumerate(sampled_classes):
            c_dir = os.path.join(self.root_dir, c_name)
            imgs = os.listdir(c_dir)
            sampled_imgs = random.sample(imgs, self.k_shot + self.q_size)
            
            for i, img_name in enumerate(sampled_imgs):
                img = Image.open(os.path.join(c_dir, img_name)).convert('RGB')
                img = self.transform(img)
                if i < self.k_shot:
                    support_imgs.append(img)
                else:
                    query_imgs.append(img)
                    query_targets.append(c_idx)

        return torch.stack(support_imgs), torch.stack(query_imgs), torch.tensor(query_targets)

# --- 2. Lightning Module (Completed) ---
class BakeryMetaClassifier(pl.LightningModule):
    def __init__(self, n_way=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.n_way = n_way
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        support_imgs, query_imgs, query_targets = batch
        
        # Remove DataLoader batch dim (assuming batch_size=1 episode)
        support_imgs = support_imgs.squeeze(0)
        query_imgs = query_imgs.squeeze(0)
        query_targets = query_targets.squeeze(0)

        support_emb = self(support_imgs) # [K*N, Emb]
        query_emb = self(query_imgs)     # [Q*N, Emb]

        # Calculate prototypes: reshape to [N, K, Emb] and mean across K
        k_shot = support_emb.shape[0] // self.n_way
        prototypes = support_emb.view(self.n_way, k_shot, -1).mean(dim=1) # [N, Emb]

        # Calculate distances: $d(Q, P)$
        distances = torch.cdist(query_emb, prototypes) # [Q*N, N]
        
        # Loss is sparse categorical crossentropy on negative distances
        loss = F.cross_entropy(-distances, query_targets)
        
        preds = torch.argmin(distances, dim=1)
        acc = (preds == query_targets).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        support_imgs, support_lbls, query_imgs = batch
        
        # Remove DataLoader batch dim
        support_imgs = support_imgs.squeeze(0)
        support_lbls = support_lbls.squeeze(0)
        query_imgs = query_imgs.squeeze(0)
        
        support_emb = self(support_imgs)
        query_emb = self(query_imgs)
        
        prototypes = torch.stack([
            support_emb[support_lbls == c].mean(dim=0) for c in range(self.n_way)
        ])
        
        distances = torch.cdist(query_emb, prototypes)
        return torch.argmin(distances, dim=1)

if __name__ == '__main__':
    # Initialize Data
    dataset = EpisodicDataset('./datasets/meta_learning/train', n_way=3, k_shot=5, q_size=5, episodes=100)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # Initialize Model
    model = BakeryMetaClassifier(n_way=3)

    # Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='./resources/checkpoints',
        filename='protonet-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',
        save_top_k=1
    )

    # Train
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], accelerator='auto')
    trainer.fit(model, dataloader)
