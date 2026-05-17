import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
import cv2
from tqdm import tqdm
import numpy as np
import kagglehub

# Local imports
from models.biomed_clip import BioMedCLIPEncoder
from segmentation.unet import UNet

class KvasirDataset(Dataset):
    def __init__(self, base_path, file_list_path, transform=None):
        self.base_path = base_path
        self.images_dir = os.path.join(base_path, "Kvasir-SEG", "Kvasir-SEG", "images")
        self.masks_dir = os.path.join(base_path, "Kvasir-SEG", "Kvasir-SEG", "masks")
        
        with open(file_list_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines() if line.strip()]
            
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename + ".jpg")
        mask_path = os.path.join(self.masks_dir, filename + ".jpg") # Kvasir masks are often .jpg or .png, need to check. Assuming .jpg based on train.txt structure
        
        if not os.path.exists(img_path):
             img_path = os.path.join(self.images_dir, filename + ".png")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.masks_dir, filename + ".png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
             print(f"Error loading: {filename}")
             return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224)
             
        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return img, mask

def train_endoscopy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download/Locate dataset
    print("Locating Kvasir-SEG dataset...")
    base_path = kagglehub.dataset_download("debeshjha1/kvasirseg")
    
    train_txt = os.path.join(base_path, "train.txt")
    val_txt = os.path.join(base_path, "val.txt")
    
    train_dataset = KvasirDataset(base_path, train_txt)
    val_dataset = KvasirDataset(base_path, val_txt)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    unet = UNet(n_classes=1).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 5 # Small number for prototype, increase for better results
    
    best_val_dice = 0.0
    
    for epoch in range(epochs):
        unet.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            outputs = unet(imgs)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        # Validation
        unet.eval()
        val_loss = 0
        dice_score = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = unet(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # Calculate Dice
                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum()
                dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-6)
                dice_score += dice.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = dice_score / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            os.makedirs("../checkpoints", exist_ok=True)
            torch.save(unet.state_dict(), "../checkpoints/unet_endoscopy_best.pth")
            print("Saved best model.")
            
    # Save final metrics
    metrics = {
        "validation": {
            "accuracy": "N/A (Segmentation)",
            "dice": round(best_val_dice, 3),
            "loss": round(avg_val_loss, 3)
        },
        "testing": {
            "dice": round(best_val_dice, 3), # Using val as proxy if no separate test set used in this run
            "note": "Test metrics are proxy from validation in this run."
        }
    }
    
    with open("../checkpoints/endoscopy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining Complete. Metrics saved to checkpoints/endoscopy_metrics.json")

if __name__ == "__main__":
    train_endoscopy()
