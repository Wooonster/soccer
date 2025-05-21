import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

def plot_training_curves(json_file):
    """
    Read training results from JSON file and plot training curves
    
    Args:
        json_file: Path to JSON file containing training results
    """
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract training metrics
    epochs = data['metrics']['epochs']
    train_loss = data['metrics']['train_loss']
    train_acc = data['metrics']['train_acc']
    val_loss = data['metrics']['val_loss']
    val_acc = data['metrics']['val_acc']
    best_epoch = data['best']['epoch']
    best_val_acc = data['best']['val_acc']
    
    # Create output directory if it doesn't exist
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', marker='o', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', marker='s', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    
    # Figure 2: Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', marker='o', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', marker='s', label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch}, Acc={best_val_acc:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300)
    
    # Figure 3: Class-specific Performance Metrics
    plt.figure(figsize=(12, 8))
    
    # Extract precision, recall and F1 for each class
    shot_precision = [data['detailed_metrics'][str(e)]['class_metrics']['precision'][0] for e in epochs]
    shot_recall = [data['detailed_metrics'][str(e)]['class_metrics']['recall'][0] for e in epochs]
    shot_f1 = [data['detailed_metrics'][str(e)]['class_metrics']['f1'][0] for e in epochs]
    
    other_precision = [data['detailed_metrics'][str(e)]['class_metrics']['precision'][1] for e in epochs]
    other_recall = [data['detailed_metrics'][str(e)]['class_metrics']['recall'][1] for e in epochs]
    other_f1 = [data['detailed_metrics'][str(e)]['class_metrics']['f1'][1] for e in epochs]
    
    macro_f1 = [data['detailed_metrics'][str(e)]['macro_metrics']['f1'] for e in epochs]
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, shot_precision, 'b-', marker='o', label='Shot Precision')
    plt.plot(epochs, shot_recall, 'b--', marker='s', label='Shot Recall')
    plt.plot(epochs, shot_f1, 'b:', marker='x', label='Shot F1-Score')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Shot Class Performance Metrics')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, other_precision, 'r-', marker='o', label='Other Precision')
    plt.plot(epochs, other_recall, 'r--', marker='s', label='Other Recall')
    plt.plot(epochs, other_f1, 'r:', marker='x', label='Other F1-Score')
    plt.plot(epochs, macro_f1, 'g-', marker='d', label='Macro Avg F1-Score')
    plt.axvline(x=best_epoch, color='g', linestyle='--')
    plt.title('Other Class and Macro-Average Performance Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=300)
    
    # Figure 4: Error Rate Metrics
    plt.figure(figsize=(10, 6))
    
    # Extract false negative rate and false positive rate
    shot_fnr = [data['detailed_metrics'][str(e)]['class_metrics']['false_negative_rate'][0] for e in epochs]
    shot_fpr = [data['detailed_metrics'][str(e)]['class_metrics']['false_positive_rate'][1] for e in epochs]
    
    plt.plot(epochs, shot_fnr, 'b-', marker='o', label='Shot False Negative Rate (FNR)')
    plt.plot(epochs, shot_fpr, 'r-', marker='s', label='Shot False Positive Rate (FPR)')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Shot Class Error Rates')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_rates.png'), dpi=300)
    
    print(f"Images saved to {output_dir} directory")

if __name__ == "__main__":
    # Specify JSON file path
    json_file = "run.json"
    
    # Plot training curves
    plot_training_curves(json_file) 