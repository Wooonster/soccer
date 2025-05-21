import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(src_root: str, split_ratio: float = 0.2, seed: int = 5525):
    """
    将 figprocess/<class>/ 下的图片随机划分并移动到 figprocess/<class>/train 和 figprocess/<class>/val
    
    Args:
        src_root: 数据根目录，包含各个类别子目录
    """
    random.seed(seed)
    src_root = Path(src_root)
    
    # 确保根目录存在
    if not src_root.exists():
        print(f"错误: 目录 {src_root} 不存在")
        return
    
    # 遍历所有类别目录
    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"警告: 在 {src_root} 下没有找到类别目录")
        return
    
    print(f"开始处理 {len(class_dirs)} 个类别...")
    
    for class_dir in tqdm(class_dirs, desc="处理类别"):
        # 跳过已有的 train/val 子目录
        if class_dir.name in ['train', 'val']:
            continue
        
        # 收集类别目录下的所有图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
            image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"警告: 类别 {class_dir.name} 下没有找到图片文件")
            continue
        
        # 创建 train 和 val 子目录
        train_dir = class_dir / 'train'
        val_dir = class_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # 随机打乱文件列表并划分
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - split_ratio))
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # 移动文件到对应目录
        print(f"  类别 {class_dir.name}: 共 {len(image_files)} 张图片，训练集 {len(train_files)}，验证集 {len(val_files)}")
        
        # 移动文件到训练集目录
        for img in train_files:
            dest = train_dir / img.name
            shutil.move(str(img), str(dest))
        
        # 移动文件到验证集目录
        for img in val_files:
            dest = val_dir / img.name
            shutil.move(str(img), str(dest))
        
        # 验证结果
        actual_train = len(list(train_dir.glob('*.*')))
        actual_val = len(list(val_dir.glob('*.*')))
        print(f"  完成移动: 训练集 {actual_train} 张，验证集 {actual_val} 张")
    
    print("数据集划分完成！")

if __name__ == "__main__":
    # 设置源目录路径
    src_directory = "figprocess"
    
    # 执行数据集划分
    split_dataset(src_directory, split_ratio=0.2, seed=5525)
    
    print("所有操作完成！") 