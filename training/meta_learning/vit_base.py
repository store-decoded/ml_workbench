import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

class MetaViTLightning(pl.LightningModule):
    def __init__(self, n_way=3):
        super().__init__()
        self.n_way = n_way
        # Use ViT as the universal feature extractor
        self.encoder = models.vit_b_16(weights=None)
        self.encoder.heads = nn.Identity() # Output raw embeddings, no classification head
        
    def forward_episode(self, support_imgs, support_lbls, query_imgs):
        # Extract features
        support_emb = self.encoder(support_imgs)
        query_emb = self.encoder(query_imgs)
        
        # Calculate Prototypes
        prototypes = []
        for c in range(self.n_way):
            class_mask = (support_lbls == c)
            prototypes.append(support_emb[class_mask].mean(dim=0))
        prototypes = torch.stack(prototypes)
        
        # Calculate distances
        distances = torch.cdist(query_emb, prototypes)
        return -distances # Return negative distances as logits

    def training_step(self, batch, batch_idx):
        # In Meta-Learning, a dataloader provides an episode (support + query sets)
        support_imgs, support_lbls, query_imgs, query_lbls = batch 
        
        logits = self.forward_episode(support_imgs, support_lbls, query_imgs)
        loss = F.cross_entropy(logits, query_lbls)
        
        self.log('train_meta_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Usage: 
# model = MetaViTLightning(n_way=3)
# trainer = pl.Trainer(max_epochs=100)
# trainer.fit(model, episodic_train_dataloader)
