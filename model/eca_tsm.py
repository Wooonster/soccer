import os
import numpy as np
import torch
import torch.nn as nn
import random
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

from eca_res50_model import eca_resnet50
from temporal_shift import make_temporal_shift, TemporalShift
from training_utils import setup_logging, log_epoch, plot_training, validate_model, display_metrics, plot_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device detected: {device}")

# ------------------------------------------------------------------

class FrameSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Dataset for loading frame sequences with dynamic length for each sequence.
        
        Args:
            root_dir: Path to dataset root (contains 'shot' and 'other' folders)
            transform: Transformations to apply
            train: If True, use training set, else validation set
        """
        self.transform = transform
        self.train = train
        
        split = 'train' if train else 'val'
        
        # Create class paths and mapping
        self.class_paths = [
            os.path.join(root_dir, 'shot', split),
            os.path.join(root_dir, 'other', split)
        ]
        self.class_names = ['shot', 'other']
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        print(f"self.class_paths --> {self.class_paths}")  # ['figprocess/shot/train', 'figprocess/other/train']
        
        # Get all frame sequence directories
        self.sequences = []
        self.class_counts = [0, 0]  # 用于计算每个类别的样本数量
        
        for class_idx, class_path in enumerate(self.class_paths):
            all_files = []
            if os.path.exists(class_path):
                all_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]

            print(f"Got {len(all_files)} image files from {class_path}")
            # Got 8771 image files from figprocess/shot/train, Got 9435 image files from figprocess/other/train
            # Got 2193 image files from figprocess/shot/val, Got 2359 image files from figprocess/other/val

            if all_files:
                shot_groups = {}
                for file in all_files:
                    # file -> 2024-11-16-11-59-07_2_other_101_3032.jpg
                    try:
                        parts = file.split('_')
                        if len(parts) >= 4:
                            shot_key = '_'.join(parts[:-1])
                            # shot_key -> 2024-11-16-11-59-07_2_other_101
                            if shot_key not in shot_groups:
                                shot_groups[shot_key] = []
                            shot_groups[shot_key].append(file)
                    except Exception as e:
                        print(f"Error parsing file {file}: {e}")

                for shot_key, frames in shot_groups.items():
                    if len(frames) > 0:
                        sorted_frames = sorted(frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        # sorted_frames -> xxx_3032.jpg xxx_3033.jpg xxx_3034.jpg
                        self.sequences.append({
                            'path': class_path,
                            'class_idx': class_idx,
                            'frames': sorted_frames,
                            'shot_key': shot_key,
                            'n_frames': len(sorted_frames),
                        })
                        self.class_counts[class_idx] += 1
        
        print(f"Loaded {len(self.sequences)} sequences for {split}")
        print(f"Class distribution: Shot={self.class_counts[0]}, Other={self.class_counts[1]}")
        # Loaded 8354 sequences for train, Class distribution: Shot=366, Other=7988
        # Loaded 2592 sequences for val, Class distribution: Shot=366, Other=2226
        
        # 计算类别权重, 用于加权采样和损失函数
        self.class_weights = [
            len(self.sequences) / (len(self.class_names) * count) if count > 0 else 0
            for count in self.class_counts
        ]
        print(f"Class weights: {self.class_weights}")
        # train -> Class weights: [11.412568306010929, 0.5229093640460691]
        # val -> Class weights: [3.540983606557377, 0.5822102425876011]
        print()
        
        # 生成每个样本的权重, 用于 WeightedRandomSampler
        self.sample_weights = [self.class_weights[seq['class_idx']] for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        frames = seq_info['frames']
        class_idx = seq_info['class_idx']
        path = seq_info['path']
        n_frames = seq_info['n_frames']
        
        # Use all available frames for each sequence (dynamic length)
        sampled_frames = frames  # Simply use all frames in the sequence
        
        # Load and transform frames
        frame_tensors = []
        for frame_name in sampled_frames:
            frame_path = os.path.join(path, frame_name)
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)
        
        # Stack frames along batch dimension
        frames_tensor = torch.stack(frame_tensors)
        
        return frames_tensor, class_idx, n_frames  # Return actual frame count

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    frames_list, labels, frame_counts = zip(*batch)
    
    # Find max length in this batch
    max_len = max(frame_counts)
    
    # Pad sequences to max length by repeating the last frame
    padded_frames = []
    for frames, actual_len in zip(frames_list, frame_counts):
        if actual_len < max_len:
            # Pad by repeating the last frame
            last_frame = frames[-1:].repeat(max_len - actual_len, 1, 1, 1)
            padded_sequence = torch.cat([frames, last_frame], dim=0)
        else:
            padded_sequence = frames
        padded_frames.append(padded_sequence)
    
    # Stack all sequences
    batched_frames = torch.stack(padded_frames)
    batched_labels = torch.tensor(labels)
    
    return batched_frames, batched_labels

def train_val_model(model, train_loader, val_loader, device, n_epochs=20, lr=0.001, class_weights=None):
    """Train and validate the model"""
    # Setup with class-weighted loss
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, device=device)
        print(f"Using class weights for loss: {class_weights_tensor}")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # 差异化 lr, 卷积层（特征提取部分）使用较小 lr, 全连接层（分类器）使用较大 lr
    conv_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            conv_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': conv_params, 'lr': lr * 0.1, 'weight_decay': 1e-5},  # 较小的学习率
        {'params': fc_params, 'lr': lr, 'weight_decay': 1e-4}           # 较大的学习率
    ])
    
    # 使用 ReduceLROnPlateau 动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Logging setup
    log_file, log_data, timestamp = setup_logging(log_dir='logs/tsm')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")):
            # reshape 输入为 TSM 所需的格式
            b, t, c, h, w = inputs.size()
            # -> [batch*frames, channels, height, width]
            inputs = inputs.view(b * t, c, h, w)
            
            # 将数据移到设备上
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # -> [batch*frames, num_classes]
            
            # 取每个序列中最后一帧的预测作为最终预测, loss 计算需要 [batch, num_classes]
            num_classes = outputs.size(1)
            # -> [batch, frames, num_classes]
            outputs = outputs.view(b, t, num_classes)
            # 然后取每个批次的最后一帧预测
            final_outputs = outputs[:, -1, :]  # -> [batch, num_classes]
            
            loss = criterion(final_outputs, targets)  # -> [batch, num_classes]
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * b
            _, predicted = final_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 周期性清理 CUDA 缓存
            if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        val_loss, val_acc, detailed_metrics, cm, _, _ = validate_model(
            model, val_loader, criterion, device, train_loader.dataset.class_names
        )
        
        # F1 score
        shot_f1 = detailed_metrics['class_metrics']['f1'][0]  # 少数类
        macro_f1 = detailed_metrics['macro_metrics']['f1']    # 宏平均
        
        # 使用 F1 分数而非准确率来选择最佳模型
        current_f1 = macro_f1  # 选择使用宏平均 F1
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': current_f1,
            }, f'model/eca_tsm_best_{timestamp}.pth')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        log_data = log_epoch(log_file, log_data, epoch, metrics, best_val_acc, best_epoch, detailed_metrics)
        
        # Print metrics
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"Shot F1={shot_f1:.4f}, Macro F1={macro_f1:.4f}, "
              f"Best F1={best_val_f1:.4f} (Epoch {best_epoch})")
        
        # Display detailed metrics
        display_metrics(detailed_metrics, train_loader.dataset.class_names)
        
        # Plot confusion matrix
        cm_save_path = f"logs/tsm/cm_epoch_{epoch}_{timestamp}.png"
        plot_confusion_matrix(cm, train_loader.dataset.class_names, cm_save_path)
    
    # Final plot
    plot_training(log_data, save_dir='logs/tsm')
    return model, best_val_f1, best_val_acc

# ------------------------------------------------------------------

# add TSM
def insert_tsm(model, n_segment, n_div, place = "blockres"):
    """
    Combine ECA-ResNet50 with Temporal Shift Module.
    
    - residual Shift with 1/4 of the channel shifted
    """
    assert place == "blockres", "only insert in residual"
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model, layer_name)
        blocks = list(stage.children())
        for i, blk in enumerate(blocks):
            shiftd_conv1 = TemporalShift(
                blk.conv1,
                n_segment=n_segment,
                n_div=n_div
            )
            blk.conv1 = shiftd_conv1
        setattr(model, layer_name, nn.Sequential(*blocks))
    return model

def load_model(num_classes=2, n_segment=8, n_div=8):
    """ load pretrained ECA-ResNet50 """
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
    # insert TSM modular
    model = insert_tsm(model, n_segment=n_segment, n_div=n_div)
    return model.to(device)


if __name__ == "__main__":
    # Memory optimization
    torch.backends.cudnn.benchmark = True
    
    # Random seed for reproducibility
    seed = 5525
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Main execution
    # n_frames = 8  # Number of frames to sample per video sequence
    batch_size = 16
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FrameSequenceDataset(
        root_dir='figprocess',
        transform=train_transform,
        train=True
    )
    
    val_dataset = FrameSequenceDataset(
        root_dir='figprocess',
        transform=val_transform,
        train=False
    )
    
    # 使用加权采样器解决类别不平衡
    use_weighted_sampler = True
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False  # 使用采样器时不能设置 shuffle=True
    else:
        sampler = None
        train_shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Find maximum sequence length for TSM configuration
    max_frames = max(seq['n_frames'] for seq in train_dataset.sequences)
    print(f"Maximum sequence length found: {max_frames} frames")
    
    # Load model
    model = load_model(num_classes=2, n_segment=max_frames, n_div=8)
    model = model.to(device)
    
    # Train model with class weights in loss function
    class_weights = train_dataset.class_weights if hasattr(train_dataset, 'class_weights') else None
    
    model, best_val_f1, best_val_acc = train_val_model(
        model, 
        train_loader, 
        val_loader, 
        device,
        n_epochs=30,
        lr=0.0001,
        class_weights=class_weights
    )
    
    print(f"Training complete. Best validation F1: {best_val_f1:.4f}, Best accuracy: {best_val_acc:.4f}")
