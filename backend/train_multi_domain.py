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

# Define separate adapters for better performance
class DomainAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super(DomainAdapter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        return self.layernorm(x + residual)

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super(DiseaseClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

# Datasets
class RadiologyDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.samples = []
        if not os.path.exists(self.root_dir):
            print(f"Path not found: {self.root_dir}")
            return
            
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(self.root_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(label_dir, file), 0 if label == 'NORMAL' else 1))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
             return torch.zeros(3, 224, 224), 0
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, label

class PathologyDataset(Dataset):
    def __init__(self, root_dir):
        # Assuming Breast Histopathology structure: patient_id/class_id/image.png
        self.samples = []
        # This dataset is huge, we might need to limit or sample it for prototype
        # Let's search for files
        count = 0
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    parts = root.split(os.sep)
                    # Label is usually the last folder name (0 or 1)
                    label = parts[-1]
                    if label in ['0', '1']:
                        self.samples.append((os.path.join(root, file), int(label)))
                        count += 1
                        if count > 5000: # Limit to 5000 for reasonable training time
                            break
            if count > 5000:
                break
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
             return torch.zeros(3, 224, 224), 0
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, label

class EndoscopyDataset(Dataset):
    def __init__(self, base_path, file_list_path):
        self.images_dir = os.path.join(base_path, "Kvasir-SEG", "Kvasir-SEG", "images")
        self.masks_dir = os.path.join(base_path, "Kvasir-SEG", "Kvasir-SEG", "masks")
        
        with open(file_list_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename + ".jpg")
        mask_path = os.path.join(self.masks_dir, filename + ".jpg")
        
        if not os.path.exists(img_path):
             img_path = os.path.join(self.images_dir, filename + ".png")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.masks_dir, filename + ".png")

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
             return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224)
             
        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask = (mask > 0.5).float()
        
        return img, mask

def train_domain(name, encoder, adapter, classifier, dataloader, val_loader, device, epochs=3):
    print(f"\n--- Training Domain: {name} ---")
    optimizer = optim.Adam(list(adapter.parameters()) + list(classifier.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        adapter.train()
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder.model.encode_image(imgs)
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)
            
            adapted = adapter(features.float())
            outputs = classifier(adapted)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": 100 * correct / total})
            
        # Validation
        adapter.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                features = encoder.model.encode_image(imgs)
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)
                adapted = adapter(features.float())
                outputs = classifier(adapted)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(adapter.state_dict(), f"../checkpoints/adapter_{name.lower()}_best.pth")
            torch.save(classifier.state_dict(), f"../checkpoints/classifier_{name.lower()}_best.pth")
            print("Saved best model.")
            
    return best_acc

def train_segmentation_domain(name, unet, dataloader, val_loader, device, epochs=3):
    print(f"\n--- Training Domain: {name} (Segmentation) ---")
    optimizer = optim.Adam(unet.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    best_dice = 0.0
    
    for epoch in range(epochs):
        unet.train()
        train_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
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
        dice_score = 0
        val_count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = unet(imgs)
                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum()
                dice = (2. * intersection) / (preds.sum() + masks.sum() + 1e-6)
                dice_score += dice.item()
                val_count += 1
                
        avg_dice = dice_score / val_count
        print(f"Val Dice: {avg_dice:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(unet.state_dict(), f"../checkpoints/unet_{name.lower()}_best.pth")
            print("Saved best model.")
            
    return best_dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = BioMedCLIPEncoder(device=device)
    
    # 1. Radiology (Commented out as it is already trained)
    # print("\nSetting up Radiology...")
    # rad_root = "d:/KaggleCache/chest_xray"
    # if not os.path.exists(rad_root):
    #     rad_root = "d:/KaggleCache/paultimothymooney_chest-xray-pneumonia/chest_xray"
    # rad_train = RadiologyDataset(rad_root, 'train')
    # rad_val = RadiologyDataset(rad_root, 'val')
    # rad_loader = DataLoader(rad_train, batch_size=4, shuffle=True)
    # rad_val_loader = DataLoader(rad_val, batch_size=4, shuffle=False)
    # adapter_rad = DomainAdapter().to(device)
    # classifier_rad = DiseaseClassifier(num_classes=2).to(device)
    
    # 2. Pathology (Uncommented now that download is complete)
    print("\nSetting up Pathology...")
    path_root = "d:/KaggleCache/datasets/paultimothymooney/breast-histopathology-images/versions/1"
    path_train = PathologyDataset(path_root)
    train_size = int(0.8 * len(path_train))
    val_size = len(path_train) - train_size
    path_train_set, path_val_set = torch.utils.data.random_split(path_train, [train_size, val_size])
    path_loader = DataLoader(path_train_set, batch_size=4, shuffle=True)
    path_val_loader = DataLoader(path_val_set, batch_size=4, shuffle=False)
    adapter_path = DomainAdapter().to(device)
    classifier_path = DiseaseClassifier(num_classes=2).to(device)
    
    # 3. Endoscopy (Commented out as it is already trained)
    # print("\nSetting up Endoscopy...")
    # endo_root = "d:/KaggleCache/debeshjha1_kvasirseg"
    # endo_train_txt = os.path.join(endo_root, "train.txt")
    # endo_val_txt = os.path.join(endo_root, "val.txt")
    # endo_train = EndoscopyDataset(endo_root, endo_train_txt)
    # endo_val = EndoscopyDataset(endo_root, endo_val_txt)
    # endo_loader = DataLoader(endo_train, batch_size=4, shuffle=True)
    # endo_val_loader = DataLoader(endo_val, batch_size=4, shuffle=False)
    # unet_endo = UNet(n_classes=1).to(device)
    
    # Train Pathology
    path_acc = train_domain("Pathology", encoder, adapter_path, classifier_path, path_loader, path_val_loader, device)
    
    # Save metrics (Load existing and update)
    metrics_path = "../checkpoints/all_domains_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    metrics["Pathology"] = {"val_accuracy": round(path_acc, 3)}
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nAll training complete. Metrics saved.")

if __name__ == "__main__":
    main()
