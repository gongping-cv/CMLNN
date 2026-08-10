#sparsity constraint CLSM

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

from transformers import TimesformerModel, TimesformerConfig


# ==================== 基础组件 ====================

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-4, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 2, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 8, channels * 2, 1),
            nn.Sigmoid()
        )
    def forward(self, rgb_feat, flow_feat):
        concat_feat = torch.cat([rgb_feat, flow_feat], dim=1)
        attention_weights = self.channel_attention(concat_feat)
        rgb_weights, flow_weights = torch.split(attention_weights,
                                                [rgb_feat.size(1), flow_feat.size(1)], dim=1)
        return rgb_feat * rgb_weights + flow_feat * flow_weights


# ==================== 关键修改点：稀疏性约束的CLSM调制器 ====================
class CrossModalLiquidStateModulator_Sparse(nn.Module):
    """
    稀疏性约束的CLSM调制器
    仅修改：添加门控机制和稀疏性损失计算
    保持所有其他逻辑不变
    """
    def __init__(self, rgb_channels, sparsity_lambda=0.01):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.sparsity_lambda = sparsity_lambda  # 稀疏性正则化强度
        
        # === 时间卷积层（与原版完全相同）===
        self.temporal_conv = nn.Conv1d(rgb_channels, rgb_channels, kernel_size=3, padding=1)
        
        # === 显著性计算（与原版完全相同）===
        self.saliency_fc = nn.Linear(rgb_channels, 1)
        
        # === 新增：调制门控网络（决定哪些样本应该被调制）===
        self.modulation_gate = nn.Sequential(
            nn.Linear(rgb_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出[0,1]的门控值
        )
        
        # === 调制参数（与原版相同，但使用绝对值确保符号）===
        self.beta_tau = nn.Parameter(torch.tensor(-0.5))  # 初始为负
        self.beta_dt = nn.Parameter(torch.tensor(0.5))    # 初始为正
        
        # === saliency_scale（与原版完全相同）===
        self.saliency_scale = nn.Parameter(torch.tensor(1.0))
        
        # === 初始化 ===
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 显著性计算部分（与原版相同）
        nn.init.normal_(self.saliency_fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.saliency_fc.bias, 0.0)
        nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.temporal_conv.bias, 0.0)
        
        # 门控网络初始化（让初始门控值接近0.5）
        nn.init.normal_(self.modulation_gate[0].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.modulation_gate[0].bias, 0.0)
        nn.init.normal_(self.modulation_gate[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.modulation_gate[2].bias, 0.0)
        
        # saliency_scale（与原版相同）
        nn.init.constant_(self.saliency_scale, 1.0)
    
    def compute_saliency_with_sparsity(self, rgb_features):
        """
        计算显著性，同时返回门控值和稀疏性损失
        新增：门控机制和稀疏性损失计算
        """
        B, C, T, H, W = rgb_features.shape
        
        # 空间池化（与原版相同）
        spatial_pooled = []
        for t in range(T):
            frame_feat = rgb_features[:, :, t, :, :]
            pooled = F.adaptive_avg_pool2d(frame_feat, 1).view(B, C)
            spatial_pooled.append(pooled)
        
        spatial_pooled = torch.stack(spatial_pooled, dim=1)  # [B, T, C]
        
        # 时间卷积增强（与原版相同）
        temporal_features = self.temporal_conv(spatial_pooled.transpose(1, 2)).transpose(1, 2)
        
        # === 计算全局门控值（基于所有帧的平均特征）===
        rgb_global = spatial_pooled.mean(dim=1)  # [B, C]
        gate_value = self.modulation_gate(rgb_global)  # [B, 1]
        
        # 计算显著性（与原版相同）
        saliency_list = []
        for t in range(T):
            s_t_linear = self.saliency_fc(temporal_features[:, t, :]) * self.saliency_scale
            s_t = torch.tanh(s_t_linear)
            saliency_list.append(s_t.squeeze(-1))
        
        saliency = torch.stack(saliency_list, dim=1)  # [B, T]
        
        # === 应用门控：只有门控值大的样本才进行调制 ===
        # 门控值乘到显著性上，实现软选择
        gated_saliency = saliency * gate_value  # [B, T]
        
        # === 计算稀疏性损失：鼓励门控值接近0或1，且显著性稀疏 ===
        # 1. 门控值的二值化损失（接近0或1）
        gate_binary_loss = torch.mean(gate_value * (1 - gate_value))  # 最大化时 gate≈0.5
        
        # 2. 显著性的L1稀疏性损失（鼓励少量帧被调制）
        saliency_sparsity_loss = torch.mean(torch.abs(gated_saliency))
        
        # 总稀疏性损失
        sparsity_loss = self.sparsity_lambda * (gate_binary_loss + saliency_sparsity_loss)
        
        return gated_saliency, gate_value, sparsity_loss
    
    def compute_saliency(self, rgb_features):
        """兼容原版接口（不返回稀疏性损失）"""
        saliency, _, _ = self.compute_saliency_with_sparsity(rgb_features)
        return saliency
    
    def modulate_parameters(self, base_tau, base_dt_scalar, saliency_coeff):
        """
        调制参数 - 与原版完全相同
        接收门控后的显著性系数
        """
        B = saliency_coeff.size(0)
        s = saliency_coeff.view(B, 1, 1, 1)

        beta_tau_eff = -torch.abs(self.beta_tau)
        beta_dt_eff = torch.abs(self.beta_dt)
        
        s_normalized = s  # s已经在[-1,1]范围
        
        tau0 = base_tau.view(1, -1, 1, 1)
        tau = tau0 * (1.0 + beta_tau_eff * s_normalized)
        
        dt_scalar = base_dt_scalar.view(1, 1, 1, 1) * (1.0 + beta_dt_eff * s_normalized)
        
        return tau, dt_scalar


# ==================== ModulatedLiquidCell_Trainable（与原版完全相同） ====================
class ModulatedLiquidCell_Trainable_Sparse(nn.Module):
    """
    可训练的液态单元 - 使用稀疏性约束CLSM
    与原版完全相同，只是改名字以区分
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.recurrent_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        self.base_time_constant = nn.Parameter(torch.ones(out_dim) * 1.5)
        self.base_dt_scalar = nn.Parameter(torch.tensor(0.1))
        
        self.max_dt = 0.3
        self.min_dt = 0.05
        self.max_tau = 3.0
        self.min_tau = 0.5

    def forward(self, x, h_prev, clsm_modulator=None, saliency_coeff=None):
        if clsm_modulator is None:
            raise ValueError("ModulatedLiquidCell_Trainable_Sparse 必须提供 clsm_modulator")
        
        if saliency_coeff is None:
            saliency_coeff = torch.ones(x.size(0), device=x.device) * 0.5
        
        tau, dt_scalar = clsm_modulator.modulate_parameters(
            self.base_time_constant, 
            self.base_dt_scalar,
            saliency_coeff
        )

        dt_scalar = torch.clamp(dt_scalar, self.min_dt, self.max_dt)
        tau = torch.clamp(tau, self.min_tau, self.max_tau)

        Wx = self.input_conv(x)
        Uh = self.recurrent_conv(h_prev)
        preact = Wx + Uh
        h_tilde = torch.tanh(preact)

        h = h_prev + dt_scalar * (-h_prev + h_tilde) / tau
        return h

    def get_modulation_stats(self):
        return {
            'base_tau_mean': self.base_time_constant.mean().item(),
            'base_dt': self.base_dt_scalar.item(),
        }


# ==================== EfficientLiquidBlock_B_CLSM_Trainable_Sparse ====================
class EfficientLiquidBlock_B_CLSM_Trainable_Sparse(nn.Module):
    """
    使用稀疏性约束CLSM的液态块
    仅修改：添加稀疏性损失返回
    """
    def __init__(self, in_channels, hidden_dim=256, num_units=4, rgb_channels=832, sparsity_lambda=0.01):
        super().__init__()

        self.channel_adapter = nn.Sequential(
            BasicConv3d(in_channels, 512, kernel_size=1, stride=1),
            BasicConv3d(512, hidden_dim, kernel_size=1, stride=1),
        )

        # 使用稀疏性版本的液态单元
        self.local_units = nn.ModuleList(
            [ModulatedLiquidCell_Trainable_Sparse(hidden_dim, hidden_dim) for _ in range(num_units)]
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout3d(p=0.3)
        self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # 使用稀疏性约束的CLSM
        self.clsm = CrossModalLiquidStateModulator_Sparse(
            rgb_channels=rgb_channels,
            sparsity_lambda=sparsity_lambda
        )
        
        self.output_fusion = nn.Conv2d(hidden_dim * num_units, hidden_dim, 1)

        nn.init.xavier_uniform_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x, rgb_features=None, states=None):
        x = self.channel_adapter(x)
        B, C, T, H, W = x.shape
        device = x.device

        # 计算显著性和稀疏性损失
        sparsity_loss = torch.tensor(0.0, device=device)
        gate_value = torch.tensor(0.0, device=device)
        
        if rgb_features is not None:
            if (rgb_features.size(2) != T or
                rgb_features.size(3) != H or
                rgb_features.size(4) != W):
                rgb_features = F.interpolate(
                    rgb_features, size=(T, H, W),
                    mode="trilinear", align_corners=False
                )
            saliency, gate_value, sparsity_loss = self.clsm.compute_saliency_with_sparsity(rgb_features)
        else:
            saliency = None

        # 初始化状态（与原版相同）
        if (states is None or
            (not isinstance(states, list)) or
            len(states) != len(self.local_units) or
            states[0].size(0) != B):
            h_locals = [torch.zeros(B, C, H, W, device=device)
                        for _ in self.local_units]
        else:
            h_locals = [s.to(device) for s in states]

        outputs = []
        modulation_stats_list = []

        for t in range(T):
            x_t = x[:, :, t]
            gate = torch.sigmoid(self.gate_conv(x_t))

            unit_outputs = []
            new_h_locals = []
            s_t = saliency[:, t] if saliency is not None else None

            for i, unit in enumerate(self.local_units):
                combined_input = x_t + gate * h_locals[i]
                
                h_i = unit(
                    combined_input, h_locals[i],
                    clsm_modulator=self.clsm,
                    saliency_coeff=s_t
                )
                unit_outputs.append(h_i)
                new_h_locals.append(h_i)

                if i == 0:
                    stats = unit.get_modulation_stats()
                    if s_t is not None:
                        stats['s_t'] = s_t.mean().item()
                    modulation_stats_list.append(stats)

            if len(unit_outputs) > 1:
                combined = torch.cat(unit_outputs, dim=1)
                frame_output = self.output_fusion(combined)
            else:
                frame_output = unit_outputs[0]

            outputs.append(frame_output)
            h_locals = new_h_locals

        out = torch.stack(outputs, dim=2) * self.output_scale
        out = self.dropout(out)
        
        # 返回时增加稀疏性损失和门控值
        return out, h_locals, modulation_stats_list, sparsity_loss, gate_value


# ==================== TimeSformerBackboneFromHF（与原版完全相同） ====================
class TimeSformerBackboneFromHF(nn.Module):
    """
    TimeSformer Backbone
    支持预训练权重加载和通道适配
    """
    def __init__(
        self,
        pretrained_dir: str = None,
        in_chans: int = 3,
        out_channels: int = 832,
        num_frames: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_channels = out_channels

        # =========================
        # 1. 如果提供预训练目录，则尝试加载权重
        # =========================
        if pretrained_dir is not None and os.path.exists(pretrained_dir):
            pretrained_dir = os.path.abspath(pretrained_dir)

            cfg_path = os.path.join(pretrained_dir, "config.json")
            weight_path = os.path.join(pretrained_dir, "pytorch_model.bin")

            if os.path.isfile(cfg_path) and os.path.isfile(weight_path):
                # ---- 加载 config.json ----
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)

                # 记录预训练权重的原始配置
                pretrained_num_channels = cfg_dict.get("num_channels", 3)
                pretrained_num_frames = cfg_dict.get("num_frames", num_frames)

                # 覆盖输入通道数与目标帧数为当前需求
                cfg_dict["num_channels"] = in_chans
                cfg_dict["num_frames"] = num_frames
                config = TimesformerConfig(**cfg_dict)

                # ---- 构建模型 ----
                self.model = TimesformerModel(config)

                # ---- 加载原始 state_dict ----
                raw_state = torch.load(weight_path, map_location="cpu")

                # 去掉 "timesformer." 前缀
                cleaned_state = {}
                for k, v in raw_state.items():
                    if k.startswith("timesformer."):
                        new_k = k[len("timesformer."):]
                    else:
                        new_k = k
                    cleaned_state[new_k] = v

                # 如果当前通道数 != 预训练通道数，做通道适配
                if in_chans != pretrained_num_channels:
                    print(
                        f"[TimeSformer] 适配输入通道: "
                        f"预训练 num_channels={pretrained_num_channels} → 当前 in_chans={in_chans}"
                    )

                    adapted_state = {}
                    for k, v in cleaned_state.items():
                        if not isinstance(v, torch.Tensor):
                            adapted_state[k] = v
                            continue

                        # 处理形状为 [*, C_in, ...] 且 C_in = 预训练通道数 的权重
                        if v.ndim >= 2 and v.shape[1] == pretrained_num_channels:
                            # 对通道取均值
                            v_mean = v.mean(dim=1, keepdim=True)

                            if in_chans == 1:
                                new_v = v_mean
                            elif in_chans == 2:
                                repeat_dims = [1, in_chans] + [1] * (v.ndim - 2)
                                new_v = v_mean.repeat(*repeat_dims)
                            else:
                                repeat_times = math.ceil(in_chans / pretrained_num_channels)
                                repeat_dims = [1, repeat_times] + [1] * (v.ndim - 2)
                                expanded = v.repeat(*repeat_dims)
                                new_v = expanded[:, :in_chans, ...]
                            adapted_state[k] = new_v
                        else:
                            adapted_state[k] = v

                    cleaned_state = adapted_state

                # 如果目标帧数与预训练权重的帧数不同，则对时间嵌入做插值适配
                if num_frames != pretrained_num_frames:
                    print(
                        f"[TimeSformer] 适配时间维度: "
                        f"预训练 num_frames={pretrained_num_frames} → 当前 num_frames={num_frames}"
                    )
                    adapted_state = {}
                    for k, v in cleaned_state.items():
                        if not isinstance(v, torch.Tensor):
                            adapted_state[k] = v
                            continue

                        if ("time_embeddings" in k or "time_embed" in k) and v.ndim == 3:
                            # 常见格式: [1, T, C] 或 [1, C, T]
                            if v.shape[1] == pretrained_num_frames:
                                new_v = F.interpolate(
                                    v.transpose(1, 2), size=num_frames,
                                    mode="linear", align_corners=False
                                ).transpose(1, 2)
                                adapted_state[k] = new_v
                                continue
                            elif v.shape[2] == pretrained_num_frames:
                                new_v = F.interpolate(
                                    v, size=num_frames,
                                    mode="linear", align_corners=False
                                )
                                adapted_state[k] = new_v
                                continue

                        adapted_state[k] = v

                    cleaned_state = adapted_state

                # ---- 加载到模型 ----
                missing, unexpected = self.model.load_state_dict(cleaned_state, strict=False)
                print(f"[TimeSformer] 已从预训练目录加载: {pretrained_dir}")
                print(f"[TimeSformer] 未匹配参数数量: {len(missing)}")
                print(f"[TimeSformer] 多余参数数量: {len(unexpected)}")
            else:
                # 配置文件不存在，使用随机初始化
                config = TimesformerConfig(
                    num_frames=num_frames,
                    image_size=image_size,
                    patch_size=patch_size,
                    num_channels=in_chans,
                )
                self.model = TimesformerModel(config)
                print("[TimeSformer] 未找到预训练权重，随机初始化")
        else:
            # =========================
            # 2. 随机初始化
            # =========================
            config = TimesformerConfig(
                num_frames=num_frames,
                image_size=image_size,
                patch_size=patch_size,
                num_channels=in_chans,
            )
            self.model = TimesformerModel(config)
            print("[TimeSformer] 未使用预训练权重，随机初始化")

        # 输出通道变换
        self.hidden_dim = self.model.config.hidden_size
        self.num_frames = self.model.config.num_frames
        self.img_size = self.model.config.image_size
        self.patch_size = (
            self.model.config.patch_size
            if isinstance(self.model.config.patch_size, int)
            else self.model.config.patch_size[0]
        )

        self.proj_out = nn.Conv3d(self.hidden_dim, out_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(out_channels)

    # =========================
    # 前向
    # =========================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        if T != self.num_frames:
            raise ValueError(
                f"TimeSformerBackboneFromHF 期望输入 {self.num_frames} 帧，但收到 {T} 帧。"
                "请在构建模型时将 num_frames 设为与输入 clip 长度一致的值。"
            )

        # 1) 统一空间尺寸（时间维度保持不变）
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(T, self.img_size, self.img_size),
                            mode="trilinear", align_corners=False)

        # 2) 转换为 TimeSformer 输入格式 [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # 3) 前向传播（时间长度由模型配置 num_frames 显式指定）
        outputs = self.model(pixel_values=x)
        tokens = outputs.last_hidden_state   # [B, 1 + T * num_patches, hidden]

        # 4) 去掉 cls token
        if tokens.size(1) > 1:
            tokens = tokens[:, 1:]

        B, N, C_hid = tokens.shape

        # 5) 动态计算空间 patch 数
        # 注意：这里要使用 self.model.config.image_size 和 self.model.config.patch_size
        # 因为输入可能经过了空间缩放，但 TimeSformer 内部已经按 config 中的尺寸处理
        img_size = self.model.config.image_size
        patch_size = self.model.config.patch_size
        H_p = img_size // patch_size
        W_p = img_size // patch_size
        patches_per_frame = H_p * W_p

        # 6) 动态计算实际帧数（从 token 数推导）
        actual_T = N // patches_per_frame
        # 可选：如果希望帧数与输入一致，可加断言
        # assert actual_T == T, f"Frame mismatch: input {T}, output {actual_T}"

        # 7) reshape 为 [B, actual_T, H_p, W_p, C_hid]
        tokens = tokens.view(B, actual_T, H_p, W_p, C_hid)
        x = tokens.permute(0, 4, 1, 2, 3).contiguous()  # [B, C_hid, actual_T, H_p, W_p]

        # 8) 输出通道映射
        x = self.proj_out(x)
        x = self.proj_bn(x)
        return x


# ==================== 完整模型 - 稀疏性约束版本 ====================
class My_LNN_based_Model_B_CLSM_Fixed3_Sparse(nn.Module):
    """
    完整模型 - 稀疏性约束版本
    基于Fixed3可训练版的最小修改
    """
    def __init__(self, num_classes=101, num_frames=8,sparsity_lambda=0.01):
        super().__init__()
        self.sparsity_lambda = sparsity_lambda
        self.num_frames=num_frames

        pretrained_dir = "./data/weights/timesformer_k400"

        # TimeSformer骨干（与原版相同）
        self.rgb_front = TimeSformerBackboneFromHF(
            pretrained_dir=pretrained_dir,
            in_chans=3,
            out_channels=832,
            num_frames=self.num_frames,
            image_size=224,
            patch_size=16,
        )

        self.flow_front = TimeSformerBackboneFromHF(
            pretrained_dir=pretrained_dir,
            in_chans=2,
            out_channels=832,
            num_frames=self.num_frames,
            image_size=224,
            patch_size=16,
        )

        # RGB增强（与原版相同）
        self.rgb_enhancer = nn.Sequential(
            nn.Conv3d(832, 832, 1),
            nn.ReLU(inplace=True),
        )
        self.rgb_norm = nn.BatchNorm3d(832)

        # 稀疏性约束CLSM-LNN
        self.flow_liquid = EfficientLiquidBlock_B_CLSM_Trainable_Sparse(
            in_channels=832,
            hidden_dim=256,
            rgb_channels=832,
            sparsity_lambda=sparsity_lambda,
        )
        self.flow_norm = nn.BatchNorm3d(256)

        # 通道注意力融合（与原版相同）
        self.channel_fusion_adapter = BasicConv3d(256, 832, kernel_size=1, stride=1)
        self.channel_fusion = ChannelAttentionFusion(832)
        self.fusion_norm = nn.BatchNorm3d(832)

        # 分类头（与原版相同）
        self.pre_classifier_norm = nn.LayerNorm(832)
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(832, 256),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

        self._flow_states = None
        self._sparsity_loss = None
        self._gate_value = None
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                if name.startswith("rgb_front.model") or name.startswith("flow_front.model"):
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm3d):
                if name.startswith("rgb_front.model") or name.startswith("flow_front.model"):
                    continue
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if name.startswith("rgb_front.model") or name.startswith("flow_front.model"):
                    continue
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, CrossModalLiquidStateModulator_Sparse):
                # 稀疏性CLSM有自己的初始化
                pass

    def reset_states(self):
        self._flow_states = None
        self._sparsity_loss = None
        self._gate_value = None

    def get_sparsity_stats(self):
        """获取稀疏性统计信息"""
        if self._gate_value is None:
            return {
                'sparsity_loss': 0.0,
                'gate_value': 0.0,
                'effective_modulation_ratio': 0.0
            }
        
        gate_mean = self._gate_value.mean().item()
        sparsity_loss_val = self._sparsity_loss.item() if isinstance(self._sparsity_loss, torch.Tensor) else 0.0
        
        return {
            'sparsity_loss': sparsity_loss_val,
            'gate_value': gate_mean,
            'effective_modulation_ratio': gate_mean  # 门控值越大，调制越强
        }

    def forward(self, rgb, flow, reset_states=False):
        B = rgb.size(0)
        device = rgb.device

        if reset_states:
            self.reset_states()

        # RGB分支
        rgb_feat = self.rgb_front(rgb)
        rgb_enhanced = self.rgb_enhancer(rgb_feat)
        rgb_feat_for_clsm = self.rgb_norm(rgb_enhanced)

        # Flow分支
        flow_feat = self.flow_front(flow)

        # 状态准备
        if (self._flow_states is None or
            len(self._flow_states) == 0 or
            self._flow_states[0].size(0) != B):
            states = None
        else:
            states = [s.to(device) for s in self._flow_states]

        # 稀疏性约束LNN前向
        flow_liquid_out, self._flow_states, modulation_stats, self._sparsity_loss, self._gate_value = self.flow_liquid(
            flow_feat,
            rgb_features=rgb_feat_for_clsm,
            states=states,
        )
        flow_liquid_out = self.flow_norm(flow_liquid_out)

        # 融合
        flow_adapted = self.channel_fusion_adapter(flow_liquid_out)
        fused = self.channel_fusion(rgb_feat_for_clsm, flow_adapted)
        fused = self.fusion_norm(fused)

        # 分类
        x = F.adaptive_avg_pool3d(fused, (1, 1, 1))
        x = x.flatten(1)
        x = self.pre_classifier_norm(x)
        logits = self.classifier(x)

        if self.training:
            return logits, modulation_stats, self._sparsity_loss
        return logits

    def get_modulation_loss(self, modulation_stats):
        """调制损失计算 - 包含稀疏性损失"""
        # 直接从模型获取β参数
        beta_tau = self.flow_liquid.clsm.beta_tau
        beta_dt = self.flow_liquid.clsm.beta_dt
        
        # 基础β正则化（与原版相同）
        beta_l1_loss = (torch.abs(beta_tau) + torch.abs(beta_dt)) * 0.0001
        
        beta_tau_sign_loss = torch.relu(beta_tau) * 0.001
        beta_dt_sign_loss = torch.relu(-beta_dt) * 0.001
        
        # s_t的动态范围损失（与原版相同）
        s_t_loss = torch.tensor(0.0, device=beta_tau.device)
        if modulation_stats and len(modulation_stats) > 0:
            s_values = []
            for stat in modulation_stats:
                if 's_t' in stat:
                    s_values.append(stat['s_t'])
            if len(s_values) > 1:
                s_std = torch.tensor(s_values, device=beta_tau.device).std()
                s_t_loss = torch.relu(0.2 - s_std) * 0.0001
        
        # 添加稀疏性损失
        sparsity_loss = self._sparsity_loss if self._sparsity_loss is not None else torch.tensor(0.0, device=beta_tau.device)
        
        total_mod_loss = beta_l1_loss + beta_tau_sign_loss + beta_dt_sign_loss + s_t_loss + sparsity_loss
        
        return total_mod_loss



# ==================== 测试验证 ====================
if __name__ == "__main__":
    print("="*80)
    print("TimeSformer 稀疏性约束版本")
    print("基于Fixed3可训练版的最小修改")
    print("核心改进: 添加门控机制，鼓励只有重要帧被调制")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = My_LNN_based_Model_B_CLSM_Fixed3_Sparse(num_classes=101,num_frames=32,sparsity_lambda=0.01).to(device)
    model.train()
    
    # 测试不同帧数
    test_frames = [8, 16, 32]
    print("\n📊 帧数适配测试:")
    
    for T in test_frames:
        rgb = torch.randn(1, 3, T, 224, 224).to(device)
        flow = torch.randn(1, 2, T, 224, 224).to(device)
        
        with torch.no_grad():
            rgb_feat = model.rgb_front(rgb)
            flow_feat = model.flow_front(flow)
            T_out = rgb_feat.shape[2]
            print(f"   输入{T:3d}帧 → 输出{T_out:3d}帧")
    
    # 测试前向传播和梯度
    print("\n📊 前向传播测试:")
    rgb = torch.randn(2, 3, 16, 224, 224).to(device)
    flow = torch.randn(2, 2, 16, 224, 224).to(device)
    
    logits, mod_stats, sparsity_loss = model(rgb, flow, reset_states=True)
    loss = logits.mean() + sparsity_loss
    loss.backward()
    
    print(f"   输出logits形状: {logits.shape}")
    print(f"   分类损失: {logits.mean().item():.4f}")
    print(f"   稀疏性损失: {sparsity_loss.item():.4f}")
    print(f"   总损失: {loss.item():.4f}")
    
    # 获取稀疏性统计
    sparsity_stats = model.get_sparsity_stats()
    print("\n📊 稀疏性统计:")
    print(f"   门控值: {sparsity_stats['gate_value']:.4f}")
    print(f"   有效调制比例: {sparsity_stats['effective_modulation_ratio']:.4f}")
    
    # 检查梯度
    clsm = model.flow_liquid.clsm
    print("\n📊 梯度检查:")
    print(f"   modulation_gate 有梯度: {clsm.modulation_gate[2].weight.grad is not None}")
    print(f"   beta_tau 梯度: {clsm.beta_tau.grad is not None}")
    print(f"   beta_dt 梯度: {clsm.beta_dt.grad is not None}")
    
    print("\n✅ 稀疏性约束版本模型初始化完成")