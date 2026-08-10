"""
Final Test for CMLNN_Jadeite_FlowAlignRGB_260224
硬编码版本 - 直接修改下面的路径即可运行
增加输入时间插值，支持测试不同帧数的输入（如64/96帧）而模型保持32帧
"""

import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# === Project-specific imports ===
from model.Dataset_i import MyDataset
from model.CMLNN_model_Jadeite_2fixed_iu import My_LNN_based_Model_B_CLSM_Fixed3_Trainable as My_Model
#from model.CMLNN_model_Jadeite_260217_2ii_fixed_discrete_iu import CMLNN_Discrete as My_Model

# ==================== 硬编码配置（直接修改这里！）====================
# 检查点路径
CHECKPOINT_PATH = "../new_checkpoints/final_CMLNN_Jadeite_align_i_UCF101_seed42_32frames.pth"  
#CHECKPOINT_PATH = "../new_checkpoints/final_CMLNN_Jadeite_align_i_UCF101_seed71_32frames.pth"  
#CHECKPOINT_PATH = "../new_checkpoints/final_CMLNN_Jadeite_align_i_UCF101_seed113_32frames.pth" 


# 测试参数
TEST_NUM_CLIPS = 3          # 每个视频的测试片段数（可调）
BATCH_SIZE = 2              # 批次大小（显存不够就改小）
NUM_WORKERS = 4             # 数据加载线程数
USE_EMA = True              # 是否使用EMA权重
SAVE_RESULTS = True         # 是否保存结果到文件
RANDOM_SEED =42 # 113# 71#          # 随机种子（与训练一致）

# 数据集相关配置
#DATA_ROOT = "../Dataset/HMDB51/processed"
#NUM_CLASSES = 51            # 类别数（UCF101=101, HMDB51=51）
#TEST_LIST_FILE = "test_RGB_npy_Split1_list.txt"  # 测试集列表文件名（位于 DATA_ROOT 下）
#SUB_FOLDER = 'npy_files'
#--------------------------------------#

DATA_ROOT = "./dataset_npy_keepall/ucf101_processed"
NUM_CLASSES = 101            # 类别数（UCF101=101, HMDB51=51）
TEST_LIST_FILE = "test_RGB_npy_Split01_list.txt"  # 测试集列表文件名（位于 DATA_ROOT 下）
SUB_FOLDER = ''



MODEL_FRAMES = 32 #训练保存的.pth版本
CLIP_LEN =   32  # 80# 64 #  48 # 96#       
# ====================================================================

# ==================== 设置随机种子 ====================
def set_seed(seed):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 测试时数据增强 ====================
class TestGPUTransforms:
    """测试时的GPU数据增强 - 只做中心裁剪，加上时间插值（如有必要）"""
    def __init__(self, target_frames=32):
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).cuda()
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).cuda()
        self.target_frames = target_frames
    
    def center_crop(self, x, target_size=224):
        H, W = x.shape[-2:]
        start_h = (H - target_size) // 2
        start_w = (W - target_size) // 2
        return x[..., start_h:start_h+target_size, start_w:start_w+target_size]
    
    def temporal_interpolate(self, x, target_frames):
        """对时间维度进行线性插值，将任意帧数调整为目标帧数"""
        B, C, T, H, W = x.shape
        if T == target_frames:
            return x
        # 使用三线性插值 (time, height, width)
        x = F.interpolate(x, size=(target_frames, H, W), mode='trilinear', align_corners=False)
        return x
    
    def normalize(self, rgb):
        return (rgb - self.rgb_mean) / self.rgb_std
    
    def __call__(self, rgb, flow):
        if not rgb.is_cuda:
            rgb = rgb.cuda(non_blocking=True)
        if not flow.is_cuda:
            flow = flow.cuda(non_blocking=True)
        
        with torch.no_grad():
            # === 关键：时间维度插值到模型期望的帧数 ===
            rgb = self.temporal_interpolate(rgb, self.target_frames)
            flow = self.temporal_interpolate(flow, self.target_frames)
            
            rgb = self.center_crop(rgb, 224)
            flow = self.center_crop(flow, 224)
            rgb = self.normalize(rgb)
        
        return rgb, flow


# ==================== 加载检查点 ====================
def load_checkpoint(checkpoint_path, model, use_ema=True):
    """加载检查点（无需修改，模型帧数与CLIP_LEN一致）"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 检查点文件不存在: {checkpoint_path}")
    
    print(f"🔄 加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    
    # 处理DataParallel的权重
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 加载模型权重
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    # 如果使用EMA且检查点中有EMA影子权重
    if use_ema and 'ema_shadow' in ckpt:
        print("✅ 使用EMA权重进行测试")
        ema_state_dict = {}
        for name, buf in ckpt['ema_shadow'].items():
            if name in state_dict:
                ema_state_dict[name] = buf
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ema_state_dict, strict=False)
        else:
            model.load_state_dict(ema_state_dict, strict=False)
        print("✅ EMA权重加载完成")
    else:
        print("✅ 使用原始模型权重")
    
    epoch = ckpt.get('epoch', 0)
    print(f"✅ 加载完成 - 轮次: {epoch}")
    return epoch


# ==================== 核心测试函数 ====================
@torch.no_grad()
def test_model(model, test_loader, test_transforms, num_classes):
    """测试模型 - 普通for循环版本（无tqdm）"""
    model.eval()
    if hasattr(model, 'reset_states'):
        model.reset_states()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_batches = len(test_loader)
    print(f"\n📊 开始测试... 共 {total_batches} 个批次")
    
    for batch_idx, (rgb, flow, labels) in enumerate(test_loader, 1):
        if hasattr(model, 'reset_states'):
            model.reset_states()
        
        rgb = rgb.cuda(non_blocking=True)
        flow = flow.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        B = labels.size(0)
        N = test_loader.dataset.num_clips
        
        # 维度处理（与训练代码一致）
        if rgb.dim() == 6:
            B, N, C, T, H, W = rgb.shape
            rgb = rgb.view(B * N, C, T, H, W)
            
            if flow.dim() == 7:
                flow = flow.view(B * N, 1, 2, T, H, W)
                flow = flow.squeeze(1)
            elif flow.dim() == 6:
                if flow.shape[1] == 1:
                    flow = flow.expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
                else:
                    flow = flow.view(B * N, 2, T, H, W)
            elif flow.dim() == 5:
                flow = flow.unsqueeze(1).expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
        
        # 应用测试时增强（内部会自动时间插值到 CLIP_LEN）
        rgb, flow = test_transforms(rgb, flow)
        
        # 模型前向
        logits = model(rgb, flow, reset_states=False)
        
        # 多clip平均
        if N > 1 and logits.dim() == 2:
            logits = logits.view(B, N, -1)
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs.mean(dim=1)
            video_preds = avg_probs.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs
            video_preds = logits.argmax(dim=-1)
        
        # 收集结果
        all_preds.extend(video_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(avg_probs.cpu().numpy())
        
        # 每100个批次显示一次进度
        if batch_idx % 100 == 0:
            current_acc = (video_preds == labels).float().mean().item()
            print(f"  进度: {batch_idx:4d}/{total_batches} 批次, 当前批次准确率: {current_acc:.2%}")
    
    print(f"  完成! 处理了 {len(all_labels)} 个样本")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ==================== 计算指标 ====================
def compute_metrics(all_preds, all_labels, all_probs, num_classes):
    """计算详细指标"""
    print("\n" + "="*60)
    print("📈 计算评估指标...")
    print("="*60)
    
    # 总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes), zero_division=0)
    
    # Macro平均
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Weighted平均
    weighted_precision, weighted_recall, weighted_f1, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Micro平均
    micro_precision, micro_recall, micro_f1, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
    
    # 打印结果
    print(f"\n🎯 总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📊 平均指标:")
    print(f"  Macro 平均 - 精确率: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1: {macro_f1:.4f}")
    print(f"  Weighted平均 - 精确率: {weighted_precision:.4f}, 召回率: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
    print(f"  Micro 平均 - 精确率: {micro_precision:.4f}, 召回率: {micro_recall:.4f}, F1: {micro_f1:.4f}")
    
    # 找出最好和最差的类别
    valid_classes = np.where(support_per_class > 0)[0]
    if len(valid_classes) > 0:
        valid_f1 = f1_per_class[valid_classes]
        best_idx = valid_classes[np.argmax(valid_f1)]
        worst_idx = valid_classes[np.argmin(valid_f1)]
        
        print(f"\n🏆 最佳类别 (F1={f1_per_class[best_idx]:.4f}): 类别 {best_idx}, 支持数={support_per_class[best_idx]}")
        print(f"📉 最差类别 (F1={f1_per_class[worst_idx]:.4f}): 类别 {worst_idx}, 支持数={support_per_class[worst_idx]}")
    
    # sklearn分类报告
    print("\n" + "="*60)
    print("📋 分类报告（sklearn格式）")
    print("="*60)
    print(classification_report(all_labels, all_preds, digits=4))
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }


# ==================== 保存结果 ====================
def save_results(metrics, all_preds, all_labels, all_probs):
    """保存结果到文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path("test_results")
    result_dir.mkdir(exist_ok=True)
    
    # 从检查点路径提取模型名称
    ckpt_name = Path(CHECKPOINT_PATH).stem
    base_name = f"{ckpt_name}_{timestamp}"
    
    # 保存摘要
    summary_file = result_dir / f"{base_name}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CMLNN 最终测试结果\n")
        f.write(f"检查点: {CHECKPOINT_PATH}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"随机种子: {RANDOM_SEED}\n")
        f.write(f"使用EMA: {USE_EMA}\n")
        f.write(f"测试片段数: {TEST_NUM_CLIPS}\n")
        f.write(f"类别数: {NUM_CLASSES}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"总体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n\n")
        f.write(f"Macro 平均 F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted 平均 F1: {metrics['weighted_f1']:.4f}\n")
        f.write(f"Micro 平均 F1: {metrics['micro_f1']:.4f}\n")
    
    # 保存预测结果
    preds_file = result_dir / f"{base_name}_predictions.npz"
    np.savez(preds_file,
             predictions=all_preds,
             labels=all_labels,
             probabilities=all_probs)
    
    print(f"\n💾 结果已保存:")
    print(f"  - 摘要: {summary_file}")
    print(f"  - 预测: {preds_file}")
    
    return summary_file


# ==================== 主程序 ====================
def main():
    """主函数"""
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    print("="*70)
    print("🔬 最终模型测试 - 硬编码版本 (支持任意帧数输入→插值到32帧)")
    print("="*70)
    print(f"检查点: {CHECKPOINT_PATH}")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"测试列表文件: {TEST_LIST_FILE}")
    print(f"测试片段数: {TEST_NUM_CLIPS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"使用EMA: {USE_EMA}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"模型期望帧数: {CLIP_LEN}")
    print("="*70)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 使用设备: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载测试数据路径
    print("\n📂 加载数据路径...")
    test_rgb = load_paths(TEST_LIST_FILE, Path(DATA_ROOT))
    test_flow = [p.replace('_RGB.npy', '_FLOW.npy') for p in test_rgb]
    print(f"测试样本数: {len(test_rgb)}")
    
    # 创建数据集（注意：这里数据集的clip_len是原始存储的帧数，如64/96，我们会在transform中插值）
    test_set = MyDataset(
        test_rgb, test_flow,
        clip_len=CLIP_LEN,  # 这个参数会被Dataset用来读取多少帧？实际上Dataset读取时会根据文件中的帧数返回，我们稍后插值
        training=False,
        num_clips=TEST_NUM_CLIPS
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"数据加载器: {len(test_loader)} batches")
    
    # 创建模型（固定为 CLIP_LEN 帧）
    print("\n🛠️ 创建模型...")
    model = My_Model(num_classes=NUM_CLASSES, num_frames=MODEL_FRAMES )
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"🚀 使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 加载检查点（无需修改）
    load_checkpoint(CHECKPOINT_PATH, model, use_ema=USE_EMA)
    
    # 创建测试增强，传入目标帧数
    test_transforms = TestGPUTransforms(target_frames=MODEL_FRAMES)
    
    # 执行测试
    all_preds, all_labels, all_probs = test_model(model, test_loader, test_transforms, num_classes=NUM_CLASSES)
    
    # 计算指标
    metrics = compute_metrics(all_preds, all_labels, all_probs, num_classes=NUM_CLASSES)
    
    # 保存结果
    if SAVE_RESULTS:
        save_results(metrics, all_preds, all_labels, all_probs)
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)
    print(f"🎯 最终准确率: {metrics['accuracy']:.2%}")
    print(f"📊 Macro F1: {metrics['macro_f1']:.4f}")
    
    return metrics


def load_paths(txt_file, data_root):
    """加载路径列表"""
    p = data_root / txt_file
    with open(p) as f:
        paths = [str(data_root / SUB_FOLDER / line.strip()) for line in f if line.strip()]
    return paths


if __name__ == "__main__":
    main()