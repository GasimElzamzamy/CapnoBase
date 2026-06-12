# main.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

from src.data_prep import generate_dataset
from src.model import CapnoCNN

def setup_experiment_folder(base_dir="results"):
    """Creates a new incrementally numbered folder like results/1_experiment/"""
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(base_dir, '*_experiment'))
    
    if not existing_dirs:
        next_num = 1
    else:
        nums = []
        for d in existing_dirs:
            folder_name = os.path.basename(d)
            try:
                nums.append(int(folder_name.split('_')[0]))
            except ValueError:
                continue
        next_num = max(nums) + 1 if nums else 1
        
    exp_dir = os.path.join(base_dir, f"{next_num}_experiment")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

# 1. PyTorch Dataset Wrapper
class CapnoDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = torch.from_numpy(np.load(x_path))
        self.y = torch.from_numpy(np.load(y_path)).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # Trigger data generation if it doesn't exist yet
    if not os.path.exists('data/processed/X.npy'):
        generate_dataset()
        
    # Load dataset
    dataset = CapnoDataset('data/processed/X.npy', 'data/processed/y.npy')
    
    # Train / Validation Split (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CapnoCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🏋️ Training starting on device: {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            
        # Validation Phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Calculate Epoch Metrics
        t_loss = train_loss / len(train_loader.dataset)
        t_acc = train_correct / len(train_loader.dataset)
        v_loss = val_loss / len(val_loader.dataset)
        v_acc = val_correct / len(val_loader.dataset)
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} -> Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} || Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

    # --- SAVE THE MODEL WEIGHTS ---
    print("\n💾 Saving model weights to 'capno_cnn.pth'...")
    torch.save(model.state_dict(), 'capno_cnn.pth')

    print("📊 Generating and saving plots silently...")
    plt.ioff() # Turn off interactive pop-ups
    
    # Setup directory
    save_dir = setup_experiment_folder()
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Plotting 1: Training & Validation Metrics
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    
    metrics_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting 2: Confusion Matrix & ROC
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Clean', 'Artifact']).plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title('Confusion Matrix')
    
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC)')
    ax[1].legend(loc="lower right")
    
    plt.tight_layout()
    
    roc_path = os.path.join(save_dir, 'roc_confusion_matrix.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training complete! All graphs saved in {save_dir}")

if __name__ == "__main__":
    main()