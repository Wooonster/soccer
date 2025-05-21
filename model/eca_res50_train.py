import os
from pathlib import Path
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from eca_res50_model import eca_resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Model Loader ----
def load_model(num_classes):
    ckpt = torch.load('./model/eca_resnet50_k3557.pth.tar', map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v

    model = eca_resnet50(k_size=[3,5,5,7], num_classes=1000)
    model.load_state_dict(new_state)
    model.fc = nn.Linear(2048, num_classes)
    return model.to(device)

# ---- Logging Functions ----
def setup_logging(log_dir='logs/eca'):
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

def plot_training(log_data, save_dir='logs/eca'):
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

# ---- Custom Dataset ----
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ---- Build Dataset ----
def build_dataset(data_dir: str, phase: str, transform, num_classes: int):
    """
    从 data_dir/<class>/train 或 /val 中收集图片路径和标签
    仅包含存在 train 和 val 子目录的一级子文件夹作为类别
    phase: 'train' 或 'val'
    """
    data_dir = Path(data_dir)
    # 只筛选出包含 train 和 val 子目录的类别文件夹
    classes = [d.name for d in data_dir.iterdir()
               if d.is_dir() and (d / 'train').is_dir() and (d / 'val').is_dir()]
    classes.sort()
    if len(classes) == 0:
        raise RuntimeError(f"没有检测到任何类别，请检查 {data_dir} 下的子目录结构")
    
    # Ensure class indices start from 0 and don't exceed num_classes-1
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    image_paths = []
    labels = []
    for cls in classes:
        phase_dir = data_dir / cls / phase
        if not phase_dir.exists():
            raise FileNotFoundError(f"目录不存在: {phase_dir}")
        for img_path in phase_dir.iterdir():
            if img_path.is_file():
                image_paths.append(str(img_path))
                labels.append(class_to_idx[cls])
    
    unique_labels = set(labels)
    # 确认标签范围
    if any(l < 0 or l >= num_classes for l in unique_labels):
        print(f"Warning: Found labels {unique_labels} but num_classes={num_classes}")
        print(f"Adjusting num_classes from {num_classes} to {max(unique_labels) + 1}")
        # Instead of raising an error, we'll return the dataset and let the caller decide
        # what to do with the mismatch
    
    print(f"Loaded {phase} dataset: {len(labels)} samples, classes={classes}, label_indices={unique_labels}")
    return CustomImageDataset(image_paths, labels, transform)

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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    total = len(all_labels)
    avg_loss = running_loss / total
    acc = np.sum(np.array(all_preds) == np.array(all_labels)) / total
    metrics = calculate_metrics(all_labels, all_preds, classes)
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, metrics, cm, all_labels, all_preds

if __name__ == '__main__':
    # Configuration
    data_dir = 'figprocess'  # 数据目录，已完成预先的train/val拆分
    batch_size = 64
    num_epochs = 40
    learning_rate = 1e-4
    resume_training = True   # 是否从保存的检查点恢复训练
    
    # Create directories and setup logging
    model_dir = Path('model').resolve()
    log_dir = Path('logs/eca').resolve()
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    save_path = model_dir / 'eca_resnet50_best.pth'
    log_file, log_data, timestamp = setup_logging(log_dir)
    
    print(f"device: {device}")
    
    # Detect classes from directory structure (假设数据已按 <class>/train 和 <class>/val 组织)
    classes = [d.name for d in Path(data_dir).iterdir() 
              if d.is_dir() and (d / 'train').is_dir() and (d / 'val').is_dir()]
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")
    
    # Setup model
    model = load_model(num_classes)
    
    # Resume training from checkpoint if requested
    start_epoch = 1
    if resume_training and save_path.exists():
        try:
            # Load the saved model state
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Resuming training from checkpoint: {save_path}")
            
            # Try to find the last log file to determine start epoch
            log_files = sorted(log_dir.glob("training_log_*.json"), key=os.path.getmtime)
            if log_files:
                with open(log_files[-1], 'r') as f:
                    last_log = json.load(f)
                    if 'metrics' in last_log and 'epochs' in last_log['metrics'] and last_log['metrics']['epochs']:
                        # Start from the next epoch after the last one logged
                        last_epoch = max(last_log['metrics']['epochs'])
                        start_epoch = last_epoch + 1
                        print(f"Resuming from epoch {start_epoch} based on log file")
                        
                        # Update best values from log if available
                        if 'best' in last_log:
                            best = last_log['best']
                            print(f"Loaded best values from log: val_acc={best['val_acc']:.4f}, epoch={best['epoch']}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 1
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ]),
    }
    
    # Datasets and dataloaders
    train_dataset = build_dataset(data_dir, 'train', data_transforms['train'], num_classes)
    val_dataset = build_dataset(data_dir, 'val', data_transforms['val'], num_classes)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    # Track best performance
    best = {'val_acc': 0.0, 'epoch': 0}
    
    # Training loop
    for epoch in range(start_epoch, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        epoch_results = {}
        
        # For collecting all predictions for metrics calculation
        all_labels = []
        all_preds = []
        
        for phase in ['train', 'val']:
            model.train(phase == 'train')
            running = {'loss': 0.0, 'correct': 0, 'total': 0}
            
            # For collecting per-phase predictions
            phase_labels = []
            phase_preds = []
            
            loop = tqdm(dataloaders[phase], desc=f'{phase:5s}', leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    
                    # Backward pass (training only)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # Update statistics
                running['loss'] += loss.item() * inputs.size(0)
                running['correct'] += (preds == labels).sum().item()
                running['total'] += inputs.size(0)
                
                # Collect labels and predictions for metrics calculation
                phase_labels.extend(labels.cpu().numpy())
                phase_preds.extend(preds.cpu().numpy())
                
                # Update progress bar
                metrics = {
                    'loss': running['loss']/running['total'], 
                    'acc': running['correct']/running['total']
                }
                loop.set_postfix(**metrics)
            
            # Phase results
            epoch_results[f'{phase}_loss'] = metrics['loss']
            epoch_results[f'{phase}_acc'] = metrics['acc']
            print(f'  {phase} Loss: {metrics["loss"]:.4f} Acc: {metrics["acc"]:.4f}')
            
            # If validation phase, calculate and display detailed metrics
            if phase == 'val':
                all_labels = phase_labels
                all_preds = phase_preds
                val_metrics = calculate_metrics(all_labels, all_preds, classes)
                display_metrics(val_metrics, classes)
            
            # Save best model (validation phase)
            if phase == 'val' and metrics['acc'] > best['val_acc']:
                best['val_acc'] = metrics['acc']
                best['epoch'] = epoch
                torch.save(model.state_dict(), save_path)
                print(f'  Saved best model (val_acc={best["val_acc"]:.4f})')
        
        # Log epoch results
        detailed_metrics = calculate_metrics(all_labels, all_preds, classes)
        log_epoch(log_file, log_data, epoch, epoch_results, 
                  best['val_acc'], best['epoch'], detailed_metrics)
        print()
    
    # Training summary and visualization
    print(f'Training complete. Best val_acc: {best["val_acc"]:.4f} at epoch {best["epoch"]}')
    plot_training(log_data, log_dir)

    # Validate model and plot confusion matrix
    print("\nFinal model evaluation:")
    val_loss, val_acc, val_metrics, cm, all_labels, all_preds = validate_model(
        model, dataloaders['val'], criterion, device, classes
    )
    display_metrics(val_metrics, classes)
    
    # Save confusion matrix
    cm_save_path = log_dir / f"confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(cm, classes, cm_save_path)
    
    # Save final metrics to json
    final_metrics = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'detailed_metrics': val_metrics,
        'confusion_matrix': cm.tolist()
    }
    with open(log_dir / f"final_metrics_{timestamp}.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nTraining complete. Model saved to {save_path}")
    print(f"Logs and visualizations saved to {log_dir}")
