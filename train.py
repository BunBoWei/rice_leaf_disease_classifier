import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import argparse
import random

from model import get_efficientnet_v2
from data import RiceDiseaseDataset
from aug import get_val_transforms

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cutmix(data, targets, alpha):
    """Applies CutMix augmentation to a batch of images."""
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    
    return data, (targets, shuffled_targets, lam)

def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(preds, targets):
    """Custom loss function for CutMix."""
    targets1, targets2, lam = targets
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def train_model(args):
    """Main function to train the model."""
    set_seed(args.run)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create Directories ---
    run_results_dir = os.path.join(args.data_root, 'results', f'run_{args.run}')
    os.makedirs(run_results_dir, exist_ok=True)
    best_model_path = os.path.join(run_results_dir, f'best_model.pt')

    # --- Data Loading ---
    train_split_file = os.path.join(args.data_root, f'splits/run_{args.run}/train.txt')
    val_split_file = os.path.join(args.data_root, f'splits/run_{args.run}/val.txt')
    
    # Initialize the training dataset with is_train=True
    train_dataset = RiceDiseaseDataset(data_root=args.data_root, split_file_path=train_split_file, is_train=True)
    # The validation dataset is initialized as before
    val_dataset = RiceDiseaseDataset(data_root=args.data_root, split_file_path=val_split_file, transform=get_val_transforms())
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Class Weights ---
    print("Calculating class weights...")
    all_labels = [sample[1] for sample in train_dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed class weights: {class_weights.cpu().numpy()}")

    # --- Model, Loss, and Optimizer Setup ---
    model = get_efficientnet_v2(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_loss = float('inf')
    early_stop_counter = 0

    # --- Training Loop ---
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        # ===== Phase A: Warm-up (First 5 epochs) =====
        if epoch == 0:
            print("--- Starting Phase A: Warm-up ---")
            for name, param in model.named_parameters():
                if 'layer1' in name or 'layer2' in name:
                    param.requires_grad = False
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_warmup, weight_decay=args.weight_decay)
        
        # ===== Phase B: Fine-tuning (Remaining epochs) =====
        if epoch == args.warmup_epochs:
            print("--- Starting Phase B: Fine-tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

        # --- Train one epoch ---
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            use_cutmix = args.cutmix_alpha > 0 and random.random() > 0.5
            if use_cutmix:
                inputs, mixed_labels = cutmix(inputs, labels, args.cutmix_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = cutmix_criterion(outputs, mixed_labels) if use_cutmix else criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Train Batch {batch_idx + 1}/{len(train_loader)}, Current Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader.dataset)

        # --- Validate one epoch ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch >= args.warmup_epochs:
            scheduler.step()

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")
        else:
            early_stop_counter += 1
            print(f"  -> Val loss did not improve. Counter: {early_stop_counter}/{args.patience}")
            if early_stop_counter >= args.patience:
                print("--- Early stopping triggered ---")
                break

    print("\nTraining complete.")
    print(f"Best model for run {args.run} is saved at {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Rice Disease Classification Model")
    
    # Paths and Run Config
    parser.add_argument('--run', type=int, required=True, help='The experiment run number (1-5). Used as random seed.')
    parser.add_argument('--data_root', type=str, default='./Dhan-Shomadhan', help='Root directory of the dataset.')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Total number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr_warmup', type=float, default=3e-4, help='Learning rate for Phase A (warm-up).')
    parser.add_argument('--lr_finetune', type=float, default=1e-4, help='Learning rate for Phase B (fine-tune).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warm-up epochs.')
    
    # Augmentation and Regularization
    parser.add_argument('--cutmix_alpha', type=float, default=0.8, help='Alpha parameter for CutMix. Set to 0 to disable.')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping.')

    args = parser.parse_args()
    train_model(args)
