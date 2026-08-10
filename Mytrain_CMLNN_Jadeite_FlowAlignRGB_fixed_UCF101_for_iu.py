import os
import time
import random
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# === Project-specific imports ===
from model.Dataset_i import MyDataset


from model.CMLNN_model_Jadeite_fixed_iu import My_LNN_based_Model_B_CLSM_Fixed3_Trainable as My_Model
#from model.CMLNN_model_Jadeite_fixed_adaptive_intensity_iu import My_LNN_based_Model_B_CLSM_Fixed3_Adaptive as My_Model
#from model.CMLNN_model_Jadeite_fixed_discrete_iu import CMLNN_Discrete as My_Model




# --------------------------- 标签加载 ---------------------------
def load_labels_from_file(filename):
    """从文件读取类别列表"""
    with open(filename, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


# --------------------------- Utils ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(42)
#set_seed(71)
set_seed(113)


# --------------------- GPU数据增强（简化版，只做空间变换）---------------------
class OptimizedGPUTransforms:
    """优化的GPU数据增强 - 仅做空间变换"""
    
    def __init__(self, training=True):
        self.training = training
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).cuda()
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).cuda()
        
    def unified_spatial_transform(self, x, target_size=224, scale=(0.8, 1.0)):
        """统一的空间变换 - 同时应用于RGB和Flow"""
        if not self.training:
            H, W = x.shape[-2:]
            start_h = (H - target_size) // 2
            start_w = (W - target_size) // 2
            return x[..., start_h:start_h+target_size, start_w:start_w+target_size]
        
        B, C, T, H, W = x.shape
        
        scale_factor = torch.rand(1, device=x.device) * (scale[1] - scale[0]) + scale[0]
        crop_h = int(H * scale_factor)
        crop_w = int(W * scale_factor)
        crop_h = max(crop_h, target_size)
        crop_w = max(crop_w, target_size)
        
        start_h = torch.randint(0, H - crop_h + 1, (1,), device=x.device).item()
        start_w = torch.randint(0, W - crop_w + 1, (1,), device=x.device).item()
        
        cropped = x[:, :, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        if crop_h != target_size or crop_w != target_size:
            cropped = F.interpolate(cropped.reshape(B*C*T, 1, crop_h, crop_w), 
                                  size=(target_size, target_size), 
                                  mode='bilinear', align_corners=False)
            cropped = cropped.reshape(B, C, T, target_size, target_size)
        
        if torch.rand(1, device=x.device) < 0.5:
            cropped = cropped.flip(-1)
            
        return cropped
    
    def efficient_color_jitter(self, rgb, brightness=0.2, contrast=0.2, saturation=0.2):
        """高效的颜色增强 - 仅应用于RGB"""
        if not self.training:
            return rgb
            
        B, C, T, H, W = rgb.shape
        
        if brightness > 0:
            brightness_factor = torch.rand(1, 1, 1, 1, 1, device=rgb.device) * brightness * 2 + (1 - brightness)
            rgb = rgb * brightness_factor
            
        if contrast > 0:
            contrast_factor = torch.rand(1, 1, 1, 1, 1, device=rgb.device) * contrast * 2 + (1 - contrast)
            mean = rgb.mean(dim=(2, 3, 4), keepdim=True)
            rgb = (rgb - mean) * contrast_factor + mean
            
        if saturation > 0:
            saturation_factor = torch.rand(1, 1, 1, 1, 1, device=rgb.device) * saturation * 2 + (1 - saturation)
            gray = rgb.mean(dim=1, keepdim=True)
            rgb = (rgb - gray) * saturation_factor + gray
            
        return rgb.clamp(0, 1)
    
    def normalize(self, rgb):
        """标准化RGB"""
        return (rgb - self.rgb_mean) / self.rgb_std
    
    def __call__(self, rgb, flow):
        """
        应用空间变换（RGB和Flow同步变换）
        """
        if not rgb.is_cuda:
            rgb = rgb.cuda(non_blocking=True)
        if not flow.is_cuda:
            flow = flow.cuda(non_blocking=True)
        
        with autocast(enabled=False):
            # 同步空间变换（使用相同的变换参数）
            rgb = self.unified_spatial_transform(rgb, 224, (0.8, 1.0))
            flow = self.unified_spatial_transform(flow, 224, (0.8, 1.0))
            
            # 颜色增强（仅RGB）
            rgb = self.efficient_color_jitter(rgb, 0.2, 0.2, 0.2)
            
            # 标准化RGB
            rgb = self.normalize(rgb)
        
        return rgb, flow


# --------------------- Config -----------------------
class FixedLRExperimentConfig:
    """训练配置"""
    def __init__(self,
                 # Optimiser
                 optimizer_type: str = 'AdamW',
                 base_lr: float = 8e-5,
                 weight_decay: float = 1e-3,
                 momentum: float = 0.9,

                 # Training
                 num_epochs: int = 50,
                 accumulation_steps: int = 3,
                 clip_grad_norm: float = 0.3,

                 # Staged unfreezing
                 stage0_min_epochs: int = 8,
                 stage1_min_epochs: int = 20,
                 plateau_eps: float = 5e-3,
                 plateau_window: int = 3,
                 
                 enable_stage2: bool = True,

                 # Data
                 data_root: str = '../mytest06/dataset_npy_keepall/ucf101_processed',
                 clip_len: int = 32,
                 train_num_clips: int = 1,
                 test_num_clips: int = 3,
                 batch_size_train: int = 4,
                 batch_size_test: int = 2,
                 num_workers_train: int = 8,
                 num_workers_test: int = 4,
                 prefetch_factor: int = 2,

                 # Regularisation
                 label_smoothing: float = 0.15,
                 ema_decay: float = 0.999,
                 use_ema_for_eval: bool = True,

                 # Stage-specific learning rates
                 stage0_lr: float = None,
                 stage1_lr: float = None,
                 stage2_lr: float = None,

                 # Misc
                 name: str = 'CMLNN_trainable_ucf101',
                 num_classes: int = 101,
                 
                 # 多GPU支持
                 use_multi_gpu: bool = False,
                 gpu_ids: list = None,
                 
                 # 调制损失权重
                 modulation_loss_weight: float = 0.01,
                 ):
        self.base_lr = base_lr
        
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        self.stage0_min_epochs = stage0_min_epochs
        self.stage1_min_epochs = stage1_min_epochs
        self.plateau_eps = plateau_eps
        self.plateau_window = plateau_window
        
        self.enable_stage2 = enable_stage2

        self.data_root = Path(data_root)
        self.clip_len = clip_len
        self.train_num_clips = train_num_clips
        self.test_num_clips = test_num_clips
        
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        
        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.prefetch_factor = prefetch_factor

        self.label_smoothing = label_smoothing
        self.ema_decay = ema_decay
        self.use_ema_for_eval = use_ema_for_eval

        self.stage0_lr = stage0_lr or base_lr
        self.stage1_lr = stage1_lr or base_lr
        self.stage2_lr = stage2_lr or base_lr

        self.name = name
        self.num_classes = num_classes
        
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = gpu_ids if gpu_ids is not None else list(range(torch.cuda.device_count()))
        self.num_gpus = len(self.gpu_ids) if self.use_multi_gpu else 1
        
        self.modulation_loss_weight = modulation_loss_weight

    def __str__(self):
        """增强配置打印信息"""
        config_str = [
            f"🔧 训练配置摘要:",
            f"  模型名称: {self.name}",
            f"  模型类型: My_LNN_based_Model_B_CLSM_Fixed3_Trainable (可训练tau0/dt)",
            f"  学习率: {self.base_lr:.1e}",
            f"  Batch size: {self.batch_size_train} per GPU",
            f"  总batch size: {self.batch_size_train * self.num_gpus}",
            f"  类别数: {self.num_classes}",
            f"  训练轮次: {self.num_epochs}",
            f"  阶段解冻:",
            f"    - Stage 0 最小轮次: {self.stage0_min_epochs}",
            f"    - Stage 1 最小轮次: {self.stage1_min_epochs}",
            f"    - Stage 2 启用: {'是' if self.enable_stage2 else '否'}",
            f"  优化器: {self.optimizer_type} (wd={self.weight_decay})",
            f"  标签平滑: {self.label_smoothing}",
            f"  EMA衰减: {self.ema_decay}",
            f"  调制损失权重: {self.modulation_loss_weight}",
            f"  光流来源: 从FLOW.npy采样 (与i6.py一致)",  # 修改说明
            f"  多GPU模式: {'是' if self.use_multi_gpu else '否'} ({self.num_gpus} GPUs)",
            f"  数据配置:",
            f"    - 训练clip数: {self.train_num_clips}",
            f"    - 测试clip数: {self.test_num_clips}",
            f"    - 数据加载workers: {self.num_workers_train}",
        ]
        return "\n".join(config_str)


# --------------------- EMA ---------------------
class ModelEMA:
    """EMA模型权重更新"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, buf in model.state_dict().items():
            if buf.dtype.is_floating_point:
                self.shadow[name] = buf.clone().detach()
        print(f"✅ EMA初始化完成 (decay={self.decay})")

    @torch.no_grad()
    def update(self, model: nn.Module):
        decay = self.decay
        msd = model.state_dict()
        for name, buf in msd.items():
            if name in self.shadow and buf.dtype.is_floating_point:
                self.shadow[name].mul_(decay).add_(buf.detach(), alpha=(1.0 - decay))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """加载EMA权重到模型"""
        msd = model.state_dict()
        self.backup = {name: msd[name].clone() for name in self.shadow.keys() if name in msd}
        
        for name, buf in self.shadow.items():
            if name in msd and msd[name].dtype.is_floating_point:
                msd[name].copy_(buf)
        
    @torch.no_grad()
    def restore(self, model: nn.Module):
        """恢复模型的原始权重"""
        msd = model.state_dict()
        for name, buf in self.backup.items():
            if name in msd:
                msd[name].copy_(buf)
        self.backup = {}


# --------------------- Checkpointing ---------------------
def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, 
                   metric_value: float, metric_type: str = 'train_loss', 
                   ema_model: ModelEMA = None):
    """保存检查点"""
    path.parent.mkdir(exist_ok=True, parents=True)
    
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    ckpt = {
        'epoch': epoch,
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        metric_type: metric_value
    }
    
    if ema_model is not None:
        ckpt['ema_shadow'] = ema_model.shadow
        ckpt['ema_decay'] = ema_model.decay
    
    torch.save(ckpt, str(path))
    print(f"💾 检查点已保存到: {path}")


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, 
                   ema_model: ModelEMA = None, device='cuda'):
    """加载检查点"""
    if not path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {path}")
    
    print(f"🔄 加载检查点: {path}")
    ckpt = torch.load(str(path), map_location=device)
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt['model'], strict=False)
    
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    
    if ema_model is not None and 'ema_shadow' in ckpt:
        ema_model.shadow = ckpt['ema_shadow']
        ema_model.decay = ckpt.get('ema_decay', ema_model.decay)
        print(f"✅ 已加载EMA状态 (decay={ema_model.decay})")
    
    epoch = ckpt.get('epoch', 0)
    print(f"✅ 检查点加载完成 - 轮次: {epoch}")
    
    return epoch, ckpt


# --------------------- 【唯一修改】Optimiser ---------------------
def create_optimizer(model: nn.Module, cfg: FixedLRExperimentConfig):
    """创建优化器 - 修复版：正确处理DataParallel"""
    # 关键修复：如果是DataParallel，使用model.module获取原始模型
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    # 获取需要训练的参数
    params = [p for p in actual_model.parameters() if p.requires_grad]
    
    # 调试信息：打印可训练参数数量
    print(f"📊 优化器创建: 找到 {len(params)} 个可训练参数组")
        
    if cfg.optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(params, lr=cfg.base_lr, momentum=cfg.momentum, 
                              weight_decay=cfg.weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_type}")


# --------------------- Data IO ---------------------
def load_paths(txt_file: str, data_root: Path):
    p = data_root / txt_file
    with open(p) as f:
        paths = [str(data_root / line.strip()) for line in f if line.strip()]
    print(f"✅ 从 {txt_file} 加载了 {len(paths)} 个样本路径")
    return paths


# --------------------- Trainer ---------------------
class AdaptiveIntensityTrainer:
    def __init__(self, cfg: FixedLRExperimentConfig):
        self.cfg = cfg
        self.current_epoch = 0
        
        print(f"🔬 训练可调制tau0/dt的自适应强度模型")
        
        # 创建模型
        self.model = My_Model(num_classes=cfg.num_classes,num_frames=cfg.clip_len)
        
        # 多GPU支持
        if cfg.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 使用 {torch.cuda.device_count()} 个GPU进行数据并行")
            device_ids = cfg.gpu_ids[:torch.cuda.device_count()]
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.cuda()
            print(f"✅ 模型已部署到GPU: {device_ids}")
        else:
            self.model = self.model.cuda()
            print("✅ 使用单GPU模式")
        
        print(f"✅ 创建可训练tau0/dt的自适应调制强度模型，类别数: {cfg.num_classes}")
        self.model.train()

        # 解冻策略
        print("\n🔧 应用初始解冻策略...")
        
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
        
        trainable_keywords = [
            'classifier', 'fusion', 'channel_fusion',
            'liquid', 'flow_liquid', 'clsm',
            'enhancer', 'adapter', 'temporal_pool',
            'saliency_fc', 'beta_tau', 'beta_dt',
            'norm', 'bn',
            'proj_out', 'proj_bn',
            'gate_conv', 'output_fusion',
            'time_constant', 'base_time_constant', 'dt', 'base_dt',
            'channel_adapter',
        ]
        
        trainable_count = 0
        frozen_count = 0
        
        for name, p in actual_model.named_parameters():
            should_train = False
            for keyword in trainable_keywords:
                if keyword in name:
                    should_train = True
                    break
            
            if not should_train and any(keyword in name for keyword in ['timesformer', 'front', 'model.']):
                if any(keyword in name for keyword in ['proj_out', 'proj_bn', 'stem']):
                    should_train = True
                else:
                    should_train = False
            
            p.requires_grad = should_train
            
            if should_train:
                trainable_count += 1
            else:
                frozen_count += 1
        
        print(f"\n📊 解冻统计:")
        print(f"  ✅ 可训练参数: {trainable_count}")
        print(f"  ❄️ 冻结参数: {frozen_count}")
        
        print("\n🔍 关键可训练参数检查:")
        for name, p in actual_model.named_parameters():
            if 'base_time_constant' in name or 'base_dt' in name:
                status = "可训练" if p.requires_grad else "冻结"
                print(f"  {name}: {status}, 形状: {p.shape}")

        self.unfreeze_stage = 0
        print("\n✅ 初始状态完成：分类器、融合模块、自适应CLSM等模块已解冻")

        self.stage1_start_epoch = 0
        self._set_freeze_eval()

        # 【使用修复后的优化器创建函数，调用方式完全不变】
        self.optimizer = create_optimizer(self.model, cfg)
        
        self.scaler = GradScaler()
        self.ema = ModelEMA(actual_model, decay=cfg.ema_decay)

        # 数据增强
        self.train_transforms = OptimizedGPUTransforms(training=True)
        self.test_transforms = OptimizedGPUTransforms(training=False)

        # 不再需要 best 检查点路径，仅保留 final
        checkpoint_dir = Path("../new_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        self.final_ckpt_path = checkpoint_dir / f"final_{cfg.name}.pth"
        self.train_loss_hist = []
        self.lr_hist = []
        self.stage_history = []
        self.modulation_loss_hist = []

    def _set_freeze_eval(self):
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
            
        for name, module in actual_model.named_modules():
            params = list(module.parameters())
            if params and all((not p.requires_grad) for p in params):
                module.eval()

    def _train_plateau(self) -> bool:
        W = self.cfg.plateau_window
        if len(self.train_loss_hist) < W:
            return False
        window = self.train_loss_hist[-W:]
        return (max(window) - min(window)) < self.cfg.plateau_eps

    def _maybe_unfreeze(self, epoch: int):
        if self.unfreeze_stage == 0:
            if epoch >= self.cfg.stage0_min_epochs and self._train_plateau():
                print("🎯 进入部分解冻（Stage1）")
                self._unfreeze_to_stage(1)
                self.stage1_start_epoch = epoch

        elif self.unfreeze_stage == 1:
            stage1_epochs = epoch - self.stage1_start_epoch
            if self.cfg.enable_stage2:
                if stage1_epochs >= self.cfg.stage1_min_epochs and self._train_plateau():
                    print("🎯 进入全解冻（Stage2）")
                    self._unfreeze_to_stage(2)
            else:
                if stage1_epochs >= self.cfg.stage1_min_epochs and self._train_plateau():
                    print("📊 Stage 1训练已达到最小轮次，继续训练但不进入Stage 2")

    def _unfreeze_to_stage(self, stage: int):
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
            
        if stage == 1:
            print("🎯 部分解冻TimeSformer后端层...")
            import re
            unfrozen_layers = [8, 9, 10, 11]
            
            for name, p in actual_model.named_parameters():
                if 'model' in name and ('flow_front' in name or 'rgb_front' in name):
                    layer_match = re.search(r'\.layer\.(\d+)\.', name)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                        p.requires_grad = (layer_num in unfrozen_layers)
                    else:
                        p.requires_grad = False
            
            self.unfreeze_stage = 1
            print(f"✅ 部分解冻完成 - TimeSformer最后4层已解冻")

        elif stage == 2:
            if self.cfg.enable_stage2:
                print("🎯 全解冻所有参数...")
                for p in actual_model.parameters():
                    p.requires_grad = True
                self.unfreeze_stage = 2
                print("✅ 全解冻完成 - 所有参数可训练")
            else:
                print("⚠️ Stage 2已禁用，保持Stage 1状态")
                self.unfreeze_stage = 1

        self._set_freeze_eval()
        self._recreate_optimizer_with_stage_lr()

    def _recreate_optimizer_with_stage_lr(self):
        if self.unfreeze_stage == 0:
            stage_lr = self.cfg.stage0_lr
            stage_name = "初始阶段"
        elif self.unfreeze_stage == 1:
            stage_lr = self.cfg.stage1_lr
            stage_name = "部分解冻阶段"
        else:
            stage_lr = self.cfg.stage2_lr
            stage_name = "全解冻阶段"
        
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
            
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in actual_model.parameters())
        
        # 临时配置
        temp_cfg = FixedLRExperimentConfig(
            optimizer_type=self.cfg.optimizer_type,
            base_lr=stage_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum,
            num_epochs=self.cfg.num_epochs,
            accumulation_steps=self.cfg.accumulation_steps,
            clip_grad_norm=self.cfg.clip_grad_norm,
            stage0_min_epochs=self.cfg.stage0_min_epochs,
            stage1_min_epochs=self.cfg.stage1_min_epochs,
            plateau_eps=self.cfg.plateau_eps,
            plateau_window=self.cfg.plateau_window,
            enable_stage2=self.cfg.enable_stage2,
            data_root=str(self.cfg.data_root),
            clip_len=self.cfg.clip_len,
            train_num_clips=self.cfg.train_num_clips,
            test_num_clips=self.cfg.test_num_clips,
            batch_size_train=self.cfg.batch_size_train,
            batch_size_test=self.cfg.batch_size_test,
            num_workers_train=self.cfg.num_workers_train,
            num_workers_test=self.cfg.num_workers_test,
            prefetch_factor=self.cfg.prefetch_factor,
            label_smoothing=self.cfg.label_smoothing,
            ema_decay=self.cfg.ema_decay,
            use_ema_for_eval=self.cfg.use_ema_for_eval,
            stage0_lr=stage_lr,
            stage1_lr=stage_lr,
            stage2_lr=stage_lr,
            name=self.cfg.name,
            num_classes=self.cfg.num_classes,
            use_multi_gpu=self.cfg.use_multi_gpu,
            gpu_ids=self.cfg.gpu_ids,
            modulation_loss_weight=self.cfg.modulation_loss_weight,
        )
        
        # 【使用修复后的优化器创建函数，调用方式完全不变】
        self.optimizer = create_optimizer(self.model, temp_cfg)
        print(f"📊 {stage_name} | 学习率: {stage_lr:.1e} | "
              f"可训练参数: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")

    def train_one_epoch(self, loader_train):
        if hasattr(self, 'current_epoch') and self.current_epoch % 2 == 0:
            print(f"🧹 清理GPU缓存 (epoch {self.current_epoch})")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        total_modulation_loss = 0.0
        
        for step, (rgb, flow, labels) in enumerate(loader_train, start=1):
            step_start_time = time.time()
            
            # 重置状态
            if isinstance(self.model, nn.DataParallel):
                actual_model = self.model.module
            else:
                actual_model = self.model
                
            if hasattr(actual_model, 'reset_states'):
                actual_model.reset_states()
            
            data_time = time.time() - step_start_time

            rgb = rgb.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            original_batch_size = labels.size(0)
            original_labels = labels.clone()

            augment_start = time.time()
            
            # ============= 维度处理 =============
            if rgb.dim() == 6:  # [B, N, C, T, H, W]
                B, N, C, T, H, W = rgb.shape
                rgb = rgb.view(B * N, C, T, H, W)
                
                if flow.dim() == 7:  # [B, N, 1, 2, T, H, W]
                    flow = flow.view(B * N, 1, 2, T, H, W)
                    flow = flow.squeeze(1)  # [B*N, 2, T, H, W]
                elif flow.dim() == 6:  # [B, 1, 2, T, H, W] 或 [B, N, 2, T, H, W]
                    if flow.shape[1] == 1:  # [B, 1, 2, T, H, W]
                        flow = flow.expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
                    else:  # [B, N, 2, T, H, W]
                        flow = flow.view(B * N, 2, T, H, W)
                elif flow.dim() == 5:  # [B, 2, T, H, W]
                    flow = flow.unsqueeze(1).expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
                
                labels_expanded = original_labels.repeat_interleave(N)
                is_multi_clip = True
                
            else:  # [B, C, T, H, W]
                if rgb.dim() == 5:  # [B, C, T, H, W]
                    B, C, T, H, W = rgb.shape
                else:  # [C, T, H, W]
                    rgb = rgb.unsqueeze(0)
                    B, C, T, H, W = rgb.shape
                
                if flow.dim() == 6:  # [B, 1, 2, T, H, W]
                    flow = flow.squeeze(1)  # [B, 2, T, H, W]
                elif flow.dim() == 5:  # [B, 2, T, H, W]
                    pass
                elif flow.dim() == 4:  # [2, T, H, W]
                    flow = flow.unsqueeze(0).expand(B, 2, T, H, W)
                
                labels_expanded = original_labels
                N = 1
                is_multi_clip = False

            # 同步空间变换
            rgb, flow = self.train_transforms(rgb, flow)
            augment_time = time.time() - augment_start

            model_start = time.time()
                
            with autocast():
                # 【关键修改】处理模型返回的tuple
                output = self.model(rgb, flow, reset_states=False)
                if isinstance(output, tuple) and len(output) == 2:
                    logits, modulation_stats = output
                else:
                    logits = output
                    modulation_stats = None
                
                if is_multi_clip and N > 1:
                    logits_per_video = logits.view(B, N, -1)
                    target_per_clip = original_labels.unsqueeze(1).repeat(1, N).view(-1)
                    target_per_clip = target_per_clip.cuda(non_blocking=True)
                    
                    loss_per_clip = F.cross_entropy(
                        logits_per_video.view(-1, logits_per_video.size(-1)), 
                        target_per_clip, 
                        reduction='none'
                    )
                    loss_per_video = loss_per_clip.view(B, N).mean(dim=1)
                    loss_cls = loss_per_video.mean()
                    
                    with torch.no_grad():
                        probs = F.softmax(logits_per_video, dim=-1)
                        avg_probs = probs.mean(dim=1)
                        preds = avg_probs.argmax(dim=-1)
                        batch_correct = (preds == original_labels.cuda()).sum().item()
                        
                else:
                    if self.cfg.label_smoothing > 0:
                        num_classes = logits.size(-1)
                        smooth = self.cfg.label_smoothing
                        with torch.no_grad():
                            true_dist = torch.zeros_like(logits)
                            true_dist.fill_(smooth / (num_classes - 1))
                            true_dist.scatter_(1, labels_expanded.unsqueeze(1), 1.0 - smooth)
                        loss_cls = torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=-1), dim=-1))
                    else:
                        loss_cls = F.cross_entropy(logits, labels_expanded)
                    
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1)
                        batch_correct = (preds == labels_expanded).sum().item()
                
                # 计算调制损失
                modulation_loss = torch.tensor(0.0, device=loss_cls.device)
                if modulation_stats is not None:
                    modulation_loss = actual_model.get_modulation_loss(modulation_stats)
                
                # 总损失
                loss = loss_cls + self.cfg.modulation_loss_weight * modulation_loss

            loss = loss / self.cfg.accumulation_steps
            self.scaler.scale(loss).backward()

            if step % self.cfg.accumulation_steps == 0:
                if self.cfg.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    params = [p for p in actual_model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(params, self.cfg.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(actual_model)

            model_time = time.time() - model_start
            step_time = time.time() - step_start_time

            total_correct += batch_correct
            total_samples += original_batch_size
            total_loss += loss_cls.item()
            total_modulation_loss += modulation_loss.item()

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        avg_modulation_loss = total_modulation_loss / max(len(loader_train), 1)

        self.current_epoch += 1
        self.modulation_loss_hist.append(avg_modulation_loss)
        
        return train_loss, train_acc, avg_modulation_loss

    @torch.no_grad()
    def evaluate(self, loader_test):
        self.model.eval()
        
        if isinstance(self.model, nn.DataParallel):
            actual_model = self.model.module
        else:
            actual_model = self.model
            
        if hasattr(actual_model, 'reset_states'):
            actual_model.reset_states()
        
        if self.cfg.use_ema_for_eval:
            self.ema.apply_to(actual_model)

        total_video_correct = 0
        total_videos = 0
        total_loss = 0.0
        
        for rgb, flow, labels in loader_test:
            if hasattr(actual_model, 'reset_states'):
                actual_model.reset_states()
            
            rgb = rgb.cuda(non_blocking=True)
            flow = flow.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            B = labels.size(0)
            N = self.cfg.test_num_clips
            
            # 维度处理
            if rgb.dim() == 6:  # [B, N, C, T, H, W]
                B, N, C, T, H, W = rgb.shape
                rgb = rgb.view(B * N, C, T, H, W)
                
                if flow.dim() == 7:  # [B, N, 1, 2, T, H, W]
                    flow = flow.view(B * N, 1, 2, T, H, W)
                    flow = flow.squeeze(1)  # [B*N, 2, T, H, W]
                elif flow.dim() == 6:  # [B, 1, 2, T, H, W] 或 [B, N, 2, T, H, W]
                    if flow.shape[1] == 1:  # [B, 1, 2, T, H, W]
                        flow = flow.expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
                    else:  # [B, N, 2, T, H, W]
                        flow = flow.view(B * N, 2, T, H, W)
                elif flow.dim() == 5:  # [B, 2, T, H, W]
                    flow = flow.unsqueeze(1).expand(B, N, 2, T, H, W).reshape(B * N, 2, T, H, W)
            
            rgb, flow = self.test_transforms(rgb, flow)
            
            # 【关键修改】评估时模型只返回logits
            logits = self.model(rgb, flow, reset_states=False)
            
            # 多clip平均
            if N > 1 and logits.dim() == 2:
                logits = logits.view(B, N, -1)
                target_expanded = labels.unsqueeze(1).expand(B, N).reshape(-1)
                logits_flat = logits.view(-1, logits.size(-1))
                loss = F.cross_entropy(logits_flat, target_expanded)
                
                probs = F.softmax(logits, dim=-1)
                avg_probs = probs.mean(dim=1)
                video_preds = avg_probs.argmax(dim=-1)
                
            else:
                loss = F.cross_entropy(logits, labels)
                video_preds = logits.argmax(dim=-1)
            
            video_correct = (video_preds == labels).sum().item()
            total_video_correct += video_correct
            total_videos += B
            total_loss += loss.item() * B

        if self.cfg.use_ema_for_eval:
            self.ema.restore(actual_model)

        test_loss = total_loss / total_videos
        test_acc = total_video_correct / total_videos
        
        return test_loss, test_acc

    # ==================== 修改的 train 方法 ====================
    def train(self, loader_train, loader_test):
        start = time.time()

        # 仅用于记录最终测试结果
        final_test_loss = None
        final_test_acc = None

        for epoch in range(1, self.cfg.num_epochs + 1):
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 训练一个 epoch
            train_loss, train_acc, modulation_loss = self.train_one_epoch(loader_train)

            # 记录训练损失（用于阶段解冻的 plateau 检测）
            self.train_loss_hist.append(train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_hist.append(current_lr)
            self.stage_history.append(self.unfreeze_stage)

            # 检查是否满足解冻条件（只依赖训练损失历史）
            self._maybe_unfreeze(epoch)

            # 只在最后一个 epoch 进行测试
            if epoch == self.cfg.num_epochs:
                print("\n📊 最终评估（最后一个epoch）...")
                final_test_loss, final_test_acc = self.evaluate(loader_test)
                print(f"✅ 最终测试损失: {final_test_loss:.4f}, 最终测试准确率: {final_test_acc:.2%}")

            # 打印进度（训练信息）
            elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
            lr_now = self.optimizer.param_groups[0]['lr']
            if epoch == self.cfg.num_epochs:
                print(f"Epoch {epoch:03d}/{self.cfg.num_epochs} | Time {elapsed} | "
                      f"Train {train_loss:.4f}({train_acc:.2%}) | ModLoss {modulation_loss:.6f} | "
                      f"Test (final) {final_test_loss:.4f}({final_test_acc:.2%}) | "
                      f"Stage {self.unfreeze_stage} | LR {lr_now:.1e}")
            else:
                print(f"Epoch {epoch:03d}/{self.cfg.num_epochs} | Time {elapsed} | "
                      f"Train {train_loss:.4f}({train_acc:.2%}) | ModLoss {modulation_loss:.6f} | "
                      f"Stage {self.unfreeze_stage} | LR {lr_now:.1e}")

        # 训练结束，保存最终检查点（仅 final.pth）
        print(f"\n💾 保存最终检查点到: {self.final_ckpt_path}")
        save_checkpoint(
            self.final_ckpt_path,
            self.model,
            self.optimizer,
            self.cfg.num_epochs,
            final_test_acc if final_test_acc is not None else 0.0,
            'final_accuracy',
            self.ema
        )

        # 返回最终测试结果（用于外部记录）
        return final_test_loss if final_test_loss is not None else float('inf'), \
               final_test_acc if final_test_acc is not None else 0.0
    # ==================== 修改结束 ====================


# --------------------- Runner ---------------------
def run_adaptive_intensity_experiment(cfg: FixedLRExperimentConfig):
    """运行自适应调制强度模型实验 - 使用与i6.py相同的光流加载方式"""
    print("=" * 70)
    print(f"🔬 自适应调制强度模型实验 (可训练tau0/dt + 从FLOW.npy采样)")
    print("=" * 70)

    # 加载RGB和FLOW路径（与i6.py完全一致）
    train_rgb = load_paths('train_RGB_npy_Split01_list.txt', cfg.data_root)
    train_flow = [p.replace('_RGB.npy', '_FLOW.npy') for p in train_rgb]  # 直接替换为_FLOW.npy
    test_rgb = load_paths('test_RGB_npy_Split01_list.txt', cfg.data_root)
    test_flow = [p.replace('_RGB.npy', '_FLOW.npy') for p in test_rgb]

    if len(train_rgb) == 0 or len(train_flow) == 0 or len(test_rgb) == 0 or len(test_flow) == 0:
        raise RuntimeError("数据文件列表为空，请检查 data_root 与 txt 列表。")

    # 直接使用MyDataset
    train_set = MyDataset(
        train_rgb, train_flow, 
        clip_len=cfg.clip_len, 
        training=True, 
        num_clips=cfg.train_num_clips
    )
    test_set = MyDataset(
        test_rgb, test_flow, 
        clip_len=cfg.clip_len, 
        training=False, 
        num_clips=cfg.test_num_clips
    )

    if cfg.use_multi_gpu:
        num_gpus = len(cfg.gpu_ids) if cfg.gpu_ids else torch.cuda.device_count()
        cfg.num_workers_train = min(16, cfg.num_workers_train * num_gpus)
        print(f"📊 多GPU模式: 调整num_workers_train到 {cfg.num_workers_train}")
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.batch_size_train, 
        shuffle=True,
        num_workers=cfg.num_workers_train, 
        persistent_workers=True, 
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.batch_size_test, 
        shuffle=False,
        num_workers=cfg.num_workers_test, 
        pin_memory=True
    )

    print(f"📊 数据加载器配置: {cfg.num_workers_train} workers, prefetch={cfg.prefetch_factor}")
    print(f"📊 光流来源: 从FLOW.npy采样")

    trainer = AdaptiveIntensityTrainer(cfg)
    best_train_loss, best_test_acc = trainer.train(train_loader, test_loader)

    print(f"🎯 自适应调制强度模型结果 - 最佳训练损失: {best_train_loss:.6f}, 最佳测试准确率: {best_test_acc:.2%}")
    return best_test_acc, best_train_loss


# --------------------- Main ---------------------
if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
        print(f"🚀 检测到 {num_gpus} 个GPU，启用多GPU训练")
        gpu_ids = list(range(num_gpus))
    else:
        print(f"🔧 检测到 {num_gpus} 个GPU，使用单GPU训练")
        gpu_ids = [0]
    
    torch.cuda.set_device(gpu_ids[0])
    
    # 配置 - 完全使用原始配置，只保留优化器修复
    cfg = FixedLRExperimentConfig(
        # 优化器
        optimizer_type='AdamW',
        base_lr=8e-5,
        weight_decay=8e-4,#1e-3,  # 保持8e-4
        
        # 训练参数
        num_epochs=20,
        accumulation_steps=3,
        clip_grad_norm=0.3,
        
        # 阶段解冻
        stage0_min_epochs=20,
        stage1_min_epochs=0,
        plateau_eps=0.01,
        plateau_window=2,
        
        # Stage 2开关
        enable_stage2=False,
        
        # 数据配置
        train_num_clips=1,
        test_num_clips=3,
        batch_size_train=4,
        batch_size_test=2,
        num_workers_train=8,
        num_workers_test=4,
        prefetch_factor=2,

        # 阶段学习率
        stage0_lr = None,
        stage1_lr = 8e-5,
        stage2_lr = 1e-5,
        
        # 正则化
        label_smoothing=0.15,
        ema_decay=0.999,
        use_ema_for_eval=True,
        
        # 模型名称 - 修改以区分
        name="CMLNN_Jadeite_align_i_UCF101_seed113_32frames",
        num_classes=101,
        
        # 多GPU设置
        use_multi_gpu=use_multi_gpu,
        gpu_ids=gpu_ids,
        
        # 调制损失权重
        modulation_loss_weight=0.01,
    )
    
    print(cfg)
    print()
    
    print(f"\n{'='*80}")
    print(f"🚀 开始训练可调制tau0/dt模型 (仅优化器修复 - 基于原始版)")
    print(f"{'='*80}")
    
    try:
        final_acc, best_train_loss = run_adaptive_intensity_experiment(cfg)
        print(f"✅ 训练完成: 准确率 = {final_acc:.2%}, 训练损失 = {best_train_loss:.6f}")
        
        results_file = "CMLNN_optimizer_fix_only_based_on_original.txt"
        with open(results_file, 'w') as f:
            f.write("可调制tau0/dt模型训练结果（仅优化器修复 - 基于原始版）\n")
            f.write("="*50 + "\n")
            f.write(f"模型: CMLNN_model_Jadeite_260217_2ii_fixed.py\n")
            f.write(f"配置: 完全使用原始配置，仅优化器创建方式修复\n")
            f.write(f"最佳测试准确率: {final_acc:.2%}\n")
            f.write(f"最佳训练损失: {best_train_loss:.6f}\n")
            f.write(f"训练轮次: {cfg.num_epochs}\n")
            f.write(f"学习率: {cfg.base_lr:.1e}\n")
            f.write(f"Weight decay: {cfg.weight_decay}\n")
        
        print(f"\n💾 结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()