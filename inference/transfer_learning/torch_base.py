import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from training.transfer_learning.torch_base import BakeryTransferClassifier

# --- Custom Dataset for Inference ---
class InferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path

if __name__ == '__main__':
    # Configuration
    CHECKPOINT_PATH = "./checkpoints_tl/resnet50-transfer-epoch=04-train_loss=0.10.ckpt"
    TEST_IMAGES = [
        "./dataset/val/croissant/sample1.jpg", 
        "./dataset/val/baguette/sample2.jpg"
    ]

    # 1. Load the saved class names
    with open('class_names.txt', 'r') as f:
        class_names = f.read().split(',')

    # 2. Load Model & weights using LitAPI
    model = BakeryTransferClassifier.load_from_checkpoint(CHECKPOINT_PATH)
    
    # 3. Prepare DataLoader
    dataset = InferenceDataset(TEST_IMAGES)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    # 4. Predict using Trainer
    trainer = pl.Trainer(accelerator='auto')
    
    # predict() returns a list of batches, containing the outputs of predict_step
    predictions = trainer.predict(model, dataloaders=dataloader)
    
    # 5. Process and Display Results
    # Flatten the batched predictions list
    flat_preds = torch.cat(predictions).tolist()
    
    for path, pred_idx in zip(TEST_IMAGES, flat_preds):
        predicted_class = class_names[pred_idx]
        print(f"Image: {path.split('/')[-1]} -> Predicted: {predicted_class}")
