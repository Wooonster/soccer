import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch

# ---- Logging Functions ----
def setup_logging(log_dir):
    """Setup logging with a single function that returns all necessary components"""
    # Create directories
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.json"
    
    # Initialize log data structure
    log_data = {
        'timestamp': timestamp,
        'metrics': {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'best': {'val_acc': 0.0, 'epoch': 0}
    }
    
    return log_file, log_data, timestamp

def log_epoch(log_file, log_data, epoch, metrics, best_val_acc=None, best_epoch=None, detailed_metrics=None):
    """Log metrics for one epoch and update best values if provided"""
    # Update metrics lists
    log_data['metrics']['epochs'].append(epoch)
    for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        log_data['metrics'][k].append(metrics[k])
    
    # Update best values if provided
    if best_val_acc is not None:
        log_data['best']['val_acc'] = best_val_acc
    if best_epoch is not None:
        log_data['best']['epoch'] = best_epoch
    
    # Add detailed metrics if provided
    if detailed_metrics is not None:
        if 'detailed_metrics' not in log_data:
            log_data['detailed_metrics'] = {}
        log_data['detailed_metrics'][str(epoch)] = detailed_metrics
    
    # Save to file
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_data

def plot_training(log_data, save_dir):
    """Plot and save training metrics visualization"""
    metrics = log_data['metrics']
    best = log_data['best']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot loss and accuracy
    ax1.plot(metrics['epochs'], metrics['train_loss'], 'b-', label='Train')
    ax1.plot(metrics['epochs'], metrics['val_loss'], 'r-', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(metrics['epochs'], metrics['train_acc'], 'b-', label='Train')
    ax2.plot(metrics['epochs'], metrics['val_acc'], 'r-', label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Mark best epoch if available
    if best['epoch'] > 0:
        ax1.axvline(x=best['epoch'], color='g', linestyle='--')
        ax2.axvline(x=best['epoch'], color='g', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"training_plot_{log_data['timestamp']}.png")
    plt.close()

def calculate_metrics(y_true, y_pred, classes=None):
    """
    Calculate detailed classification metrics:
    - Precision, Recall, F1-score
    - False negative rate (漏判率): 1 - recall
    - False positive rate (错判率): (false positives) / (false positives + true negatives)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names (optional)
    """
    # Compute precision, recall, f1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0, average=None
    )
    
    # Compute macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0, average='macro'
    )
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate false negative rate (漏判率) and false positive rate (错判率)
    false_negative_rate = []  # 漏判率 = 1 - recall
    false_positive_rate = []  # 错判率
    
    n_classes = len(precision)
    for i in range(n_classes):
        # False negative rate = 1 - recall
        fnr = 1 - recall[i]
        false_negative_rate.append(fnr)
        
        # Calculate false positive rate
        if cm.shape[0] > 1:  # Only applicable for binary/multiclass problems
            # False positives = sum of column i except true positive
            false_positives = np.sum(cm[:, i]) - cm[i, i]
            # True negatives = sum of all elements except row i and column i
            true_negatives = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            # False positive rate
            fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            false_positive_rate.append(fpr)
        else:
            false_positive_rate.append(0)
    
    # Prepare results
    metrics_dict = {
        'class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'support': support.tolist()
        },
        'macro_metrics': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
    }
    
    return metrics_dict

def display_metrics(metrics_dict, classes=None):
    """Print metrics in a formatted table"""
    class_metrics = metrics_dict['class_metrics']
    macro_metrics = metrics_dict['macro_metrics']
    
    n_classes = len(class_metrics['precision'])
    
    # Use generic class names if not provided
    if classes is None or len(classes) != n_classes:
        classes = [f'Class {i}' for i in range(n_classes)]
    
    print("\nDetailed Classification Metrics:")
    print("-" * 100)
    header = f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'漏判率(FNR)':<10} | {'错判率(FPR)':<10} | {'Support':<10}"
    print(header)
    print("-" * 100)
    
    for i in range(n_classes):
        row = (f"{classes[i]:<15} | "
               f"{class_metrics['precision'][i]:.4f}     | "
               f"{class_metrics['recall'][i]:.4f}     | "
               f"{class_metrics['f1'][i]:.4f}     | "
               f"{class_metrics['false_negative_rate'][i]:.4f}     | "
               f"{class_metrics['false_positive_rate'][i]:.4f}     | "
               f"{class_metrics['support'][i]}")
        print(row)
    
    print("-" * 100)
    macro_row = (f"{'Macro Average':<15} | "
                f"{macro_metrics['precision']:.4f}     | "
                f"{macro_metrics['recall']:.4f}     | "
                f"{macro_metrics['f1']:.4f}     | "
                f"{'-':<10} | "
                f"{'-':<10} | "
                f"{'-':<10}")
    print(macro_row)
    print("-" * 100)
    
    return

def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Format the text display in each cell
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def validate_model(model, dataloader, criterion, device, classes):
    """Run model validation and calculate detailed metrics"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            # 处理输入数据 - 与训练阶段相同的格式转换
            b, t, c, h, w = inputs.size()
            
            # 转换为[batch*frames, channels, height, width]
            inputs = inputs.view(b * t, c, h, w)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            # 重塑输出并取最后一帧的预测
            num_classes = outputs.size(1)
            outputs = outputs.view(b, t, num_classes)
            final_outputs = outputs[:, -1, :]  # [batch, num_classes]
            
            loss = criterion(final_outputs, labels)
            preds = final_outputs.argmax(dim=1)
            
            running_loss += loss.item() * b
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    total = len(all_labels)
    avg_loss = running_loss / total
    acc = np.sum(np.array(all_preds) == np.array(all_labels)) / total
    metrics = calculate_metrics(all_labels, all_preds, classes)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, metrics, cm, all_labels, all_preds 