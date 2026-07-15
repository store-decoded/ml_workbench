import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import ModelCheckpoint

# --- 1. Lightning Module (Prototypical Network) ---
class ProtoNet(pl.LightningModule):
    def __init__(self, n_way: int = 5, k_shot: int = 5, query_size: int = 5, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Backbone feature extractor (ResNet18, stripped of classification head)
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        
    def forward(self, x):
        return self.encoder(x)
        
    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Remove the extra batch dimension added by the DataLoader
        images = images.squeeze(0)  # Shape becomes [50, 3, 84, 84]

        # In a real episodic sampler, you'd guarantee N classes, K+Q images per class.
        # For this functional strict codebase, we assume the dataloader provides 
        # a flattened batch of size (N * (K + Q), C, H, W).
        # We manually split them here based on the hyperparameters.
        n_way = self.hparams.n_way
        k_shot = self.hparams.k_shot
        q_size = self.hparams.query_size
        
        embeddings = self(images)
        
        # Reshape to [N, K+Q, EmbeddingDim]
        # Note: This assumes the batch is strictly ordered: N classes, K+Q items each.
        emb_dim = embeddings.size(-1)
        embeddings = embeddings.view(n_way, k_shot + q_size, emb_dim)
        
        # Split into support and query
        support = embeddings[:, :k_shot, :] # [N, K, Dim]
        query = embeddings[:, k_shot:, :]   # [N, Q, Dim]
        
        # Calculate Prototypes (mean of support set)
        prototypes = support.mean(dim=1)    # [N, Dim]
        
        # Flatten queries to compute distances
        query_flat = query.reshape(n_way * q_size, emb_dim) # [N*Q, Dim]
        
        # Calculate Euclidean distance: $d(x, y) = ||x - y||^2$
        distances = torch.cdist(query_flat, prototypes) # [N*Q, N]
        
        # Generate target labels for queries: [0,0.., 1,1.., N,N..]
        target_inds = torch.arange(n_way).repeat_interleave(q_size).to(self.device)
        
        # Loss is cross entropy on negative distances
        loss = F.cross_entropy(-distances, target_inds)
        
        # Accuracy
        preds = torch.argmin(distances, dim=1)
        acc = (preds == target_inds).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.squeeze(0)  # Shape becomes [50, 3, 84, 84]
        
        n_way, k_shot, q_size = self.hparams.n_way, self.hparams.k_shot, self.hparams.query_size
        
        embeddings = self(images)
        emb_dim = embeddings.size(-1)
        embeddings = embeddings.view(n_way, k_shot + q_size, emb_dim)
        
        support = embeddings[:, :k_shot, :]
        query = embeddings[:, k_shot:, :]
        
        prototypes = support.mean(dim=1)
        query_flat = query.reshape(n_way * q_size, emb_dim)
        
        distances = torch.cdist(query_flat, prototypes)
        target_inds = torch.arange(n_way).repeat_interleave(q_size).to(self.device)
        
        loss = F.cross_entropy(-distances, target_inds)
        preds = torch.argmin(distances, dim=1)
        acc = (preds == target_inds).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# --- 2. Dummy Episodic DataLoader (For demonstration) ---
# A production system requires a custom Sampler to guarantee N-way K-shot batches.
# We wrap standard ImageFolder to simulate this strict output.
class EpisodicWrapper(Dataset):
    def __init__(self, root_dir, n_way, k_shot, q_size, transform):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.n_way = n_way
        self.total_per_class = k_shot + q_size
        # Group indices by class
        self.class_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
            
        self.classes = list(self.class_indices.keys())

    def __len__(self):
        return 100 # Define arbitrary number of episodes per epoch

    def __getitem__(self, idx):
        # Sample N classes
        import random
        sampled_classes = random.sample(self.classes, self.n_way)
        
        episode_imgs = []
        episode_lbls = []
        
        for c in sampled_classes:
            # Sample K+Q images per class
            sampled_idx = random.sample(self.class_indices[c], self.total_per_class)
            for i in sampled_idx:
                img, _ = self.dataset[i]
                episode_imgs.append(img)
                episode_lbls.append(c)
                
        # Stack into single tensors
        return torch.stack(episode_imgs), torch.tensor(episode_lbls)

if __name__ == '__main__':
    # Configuration
    N_WAY = 5
    K_SHOT = 5
    Q_SIZE = 5
    
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Data
    train_dataset = EpisodicWrapper('./datasets/meta_learning/train', N_WAY, K_SHOT, Q_SIZE, transform)
    # Batch size 1 because the dataset returns a full episode (N * (K+Q) images)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    val_dataset = EpisodicWrapper('./datasets/meta_learning/eval', N_WAY, K_SHOT, Q_SIZE, transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize Model & Trainer
    model = ProtoNet(n_way=N_WAY, k_shot=K_SHOT, query_size=Q_SIZE)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./resources/checkpoints",
        filename="protonet-{epoch:02d}",
        save_top_k=1,
        monitor="train_loss"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./resources/checkpoints",
        filename="protonet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss", # Monitor validation loss instead
        mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=10, 
        callbacks=[checkpoint_callback],
        accelerator="auto"
    )
    
    # Train
    # Note: For production, add validation loader. Using train only for brevity.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
