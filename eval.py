import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_efficientnet_v2 
from data import RiceDiseaseDataset
from aug import get_val_transforms

def evaluate_model(args):
    """
    Loads a trained model, evaluates it, saves metrics to CSV,
    and saves a confusion matrix heatmap.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Paths ---
    run_results_dir = os.path.join(args.data_root, 'results', f'run_{args.run}')
    model_path = os.path.join(run_results_dir, 'best_model.pt')
    results_csv_path = os.path.join(run_results_dir, 'metrics.csv')
    test_split_file = os.path.join(args.data_root, f'splits/run_{args.run}/test.txt')

    # --- Load Model ---
    model = get_efficientnet_v2(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # --- Prepare Datasets for Scenarios ---
    full_test_dataset = RiceDiseaseDataset(data_root=args.data_root, split_file_path=test_split_file, transform=get_val_transforms())
    
    idx_to_class = {v: k for k, v in full_test_dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    white_indices, field_indices = [], []
    with open(test_split_file, 'r') as f:
        for i, line in enumerate(f.read().strip().split('\n')):
            if not line: continue
            _, _, background = line.strip().split(',')
            if background == "White Background":
                white_indices.append(i)
            elif background == "Field Background":
                field_indices.append(i)

    scenarios = {
        "white_background": Subset(full_test_dataset, white_indices),
        "field_background": Subset(full_test_dataset, field_indices),
        "mixed_background": full_test_dataset
    }
    
    # --- Evaluate Each Scenario ---
    all_results = []
    for scenario_name, scenario_dataset in scenarios.items():
        print(f"\n--- Evaluating scenario: {scenario_name} ---")
        if len(scenario_dataset) == 0:
            print("No samples found for this scenario.")
            continue

        loader = DataLoader(scenario_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # --- Overall Metrics ---
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"  -> Overall Accuracy: {accuracy:.4f}")
        print(f"  -> Macro F1-Score:   {macro_f1:.4f}")

        # --- Per-Class Accuracy Calculation ---
        print("\n  --- Per-Disease Accuracy ---")
        cm = confusion_matrix(all_labels, all_preds)
        
        correct_per_class = cm.diagonal()
        total_per_class = cm.sum(axis=1)
        
        for class_idx, class_name in idx_to_class.items():
            correct = correct_per_class[class_idx]
            total = total_per_class[class_idx]
            
            if total > 0:
                class_acc = correct / total
                print(f"    - {class_name:<15}: {correct}/{total} ({class_acc:.2%})")
            else:
                print(f"    - {class_name:<15}: 0/0 (No samples in test set)")
        
        # --- Generate and Save Confusion Matrix Heatmap ---
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {scenario_name} - Run {args.run}')
        
        cm_path = os.path.join(run_results_dir, f'confusion_matrix_{scenario_name}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"  -> Confusion matrix saved to {cm_path}")
        
        all_results.append({
            'run': args.run,
            'scenario': scenario_name,
            'accuracy': accuracy,
            'macro_f1': macro_f1
        })

    # --- Save Results ---
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv_path, index=False)
    print(f"\nResults for run {args.run} saved to {results_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--run', type=int, required=True, help='The experiment run number to evaluate (1-5).')
    parser.add_argument('--data_root', type=str, default='./Dhan-Shomadhan', help='Root directory of the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    args = parser.parse_args()
    evaluate_model(args)
