#adaptive intensity CLSM

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


# ==================== 关键修改点：自适应CLSM调制器 ====================
class CrossModalLiquidStateModulator_Adaptive(nn.Module):
    """
    自适应调制强度的CLSM
    仅修改：将固定β改为动态生成的β
    保持所有其他逻辑不变
    """
    def __init__(self, rgb_channels, hidden_dim=64):
        super().__init__()
        self.rgb_channels = rgb_channels
        
        # === 时间卷积层（与原版完全相同）===
        self.temporal_conv = nn.Conv1d(rgb_channels, rgb_channels, kernel_size=3, padding=1)
        
        # === 显著性计算（与原版完全相同）===
        self.saliency_fc = nn.Linear(rgb_channels, 1)
        
        # === 修改点1：动态β生成网络（替代原版的固定β参数）===
        # β_tau 网络 - 输出范围[-1,1]（保持与原版β_tau的负值特性）
        self.beta_tau_net = nn.Sequential(
            nn.Linear(rgb_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Tanh保证在[-1,1]范围
        )
        
        # β_dt 网络 - 输出范围[0,1]（保持与原版β_dt的正值特性）
        self.beta_dt_net = nn.Sequential(
            nn.Linear(rgb_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Sigmoid保证在[0,1]范围
        )
        
        # === saliency_scale（与原版完全相同）===
        self.saliency_scale = nn.Parameter(torch.tensor(1.0))
        
        # === 初始化 ===
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使初始β接近原版的固定值"""
        # 显著性计算部分（与原版相同）
        nn.init.normal_(self.saliency_fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.saliency_fc.bias, 0.0)
        nn.init.kaiming_normal_(self.temporal_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.temporal_conv.bias, 0.0)
        
        # β网络初始化：让初始输出接近原版的固定值
        # 原版：beta_tau ≈ -0.5, beta_dt ≈ 0.5
        
        # β_tau网络：最后一层权重小，bias设为-0.5左右
        nn.init.normal_(self.beta_tau_net[0].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_tau_net[0].bias, 0.0)
        nn.init.normal_(self.beta_tau_net[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_tau_net[2].bias, -0.5)  # 初始接近-0.5
        
        # β_dt网络：最后一层权重小，bias设为0.5左右
        nn.init.normal_(self.beta_dt_net[0].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_dt_net[0].bias, 0.0)
        nn.init.normal_(self.beta_dt_net[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_dt_net[2].bias, 0.5)  # 初始接近0.5
        
        # saliency_scale（与原版相同）
        nn.init.constant_(self.saliency_scale, 1.0)
    
    def compute_saliency(self, rgb_features):
        """
        计算每帧的显著性系数（与原版完全相同）
        但额外返回全局池化特征用于β生成
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
        
        # 计算显著性（与原版相同）
        saliency_list = []
        for t in range(T):
            s_t_linear = self.saliency_fc(temporal_features[:, t, :]) * self.saliency_scale
            s_t = torch.tanh(s_t_linear)
            saliency_list.append(s_t.squeeze(-1))
        
        saliency = torch.stack(saliency_list, dim=1)  # [B, T]
        
        # === 新增：计算全局RGB特征用于β生成 ===
        # 对时间维度取平均，得到每个样本的全局特征
        rgb_global = spatial_pooled.mean(dim=1)  # [B, C]
        
        return saliency, rgb_global
    
    def compute_dynamic_betas(self, rgb_global):
        """根据全局RGB特征动态生成β值"""
        beta_tau = self.beta_tau_net(rgb_global)  # [B, 1] in [-1,1]
        beta_dt = self.beta_dt_net(rgb_global)    # [B, 1] in [0,1]
        return beta_tau, beta_dt
    
    def modulate_parameters(self, base_tau, base_dt_scalar, saliency_coeff, rgb_global):
        """
        调制参数 - 动态β版本
        保持与原版完全相同的调制逻辑，只是β变成动态的
        """
        B = saliency_coeff.size(0)
        s = saliency_coeff.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 动态生成β
        beta_tau, beta_dt = self.compute_dynamic_betas(rgb_global)  # [B, 1]
        
        # 确保β_tau为负、β_dt为正（与原版的符号约束一致）
        beta_tau_eff = -torch.abs(beta_tau)  # [B, 1] 强制为负
        beta_dt_eff = torch.abs(beta_dt)     # [B, 1] 强制为正
        
        # s已经在[-1,1]范围
        s_normalized = s
        
        # 调制（与原版完全相同）
        tau0 = base_tau.view(1, -1, 1, 1)  # [1, C, 1, 1]
        
        # === 修复点：正确处理beta_tau_eff的维度 ===
        # beta_tau_eff: [B, 1] -> [B, 1, 1, 1] 用于广播
        beta_tau_eff_broadcast = beta_tau_eff.view(B, 1, 1, 1)
        tau = tau0 * (1.0 + beta_tau_eff_broadcast * s_normalized)  # [B, C, 1, 1]
        
        # === 修复点：dt_scalar应该是标量，但这里我们需要保持与原版一致的逻辑 ===
        # 原版：base_dt_scalar是标量，乘以(1 + beta_dt_eff * s)后仍然是标量
        # 但beta_dt_eff现在是[B,1]，所以需要处理一下
        beta_dt_eff_squeezed = beta_dt_eff.squeeze(-1)  # [B]
        s_squeezed = saliency_coeff  # [B]
        
        # dt_scalar应该是[B]，每个样本有自己的dt
        dt_scalar = base_dt_scalar.view(1) * (1.0 + beta_dt_eff_squeezed * s_squeezed)  # [B]
        
        return tau, dt_scalar


# ==================== ModulatedLiquidCell_Trainable（修复forward中的维度问题） ====================
class ModulatedLiquidCell_Trainable_Adaptive(nn.Module):
    """
    可训练的液态单元 - 使用自适应CLSM
    修复：正确处理dt_scalar的维度
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.recurrent_conv = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        self.base_time_constant = nn.Parameter(torch.ones(out_dim) * 1.5)  # [C]
        self.base_dt_scalar = nn.Parameter(torch.tensor(0.1))  # scalar
        
        self.max_dt = 0.3
        self.min_dt = 0.05
        self.max_tau = 3.0
        self.min_tau = 0.5

    def forward(self, x, h_prev, clsm_modulator=None, saliency_coeff=None, rgb_global=None):
        """
        新增rgb_global参数用于动态β计算
        """
        if clsm_modulator is None:
            raise ValueError("必须提供 clsm_modulator")
        
        if saliency_coeff is None:
            saliency_coeff = torch.ones(x.size(0), device=x.device) * 0.5
        
        # 调制参数（现在需要rgb_global）
        tau, dt_scalar = clsm_modulator.modulate_parameters(
            self.base_time_constant, 
            self.base_dt_scalar,
            saliency_coeff,
            rgb_global  # 新增参数
        )

        # tau: [B, C, 1, 1]
        # dt_scalar: [B]
        
        dt_scalar = torch.clamp(dt_scalar, self.min_dt, self.max_dt)  # [B]
        tau = torch.clamp(tau, self.min_tau, self.max_tau)  # [B, C, 1, 1]

        Wx = self.input_conv(x)  # [B, C, H, W]
        Uh = self.recurrent_conv(h_prev)  # [B, C, H, W]
        preact = Wx + Uh
        h_tilde = torch.tanh(preact)  # [B, C, H, W]

        # === 修复点：正确处理dt_scalar的维度 ===
        # dt_scalar: [B] -> [B, 1, 1, 1] 用于广播
        dt_broadcast = dt_scalar.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # h = h_prev + dt * (-h_prev + h_tilde) / tau
        # (-h_prev + h_tilde) / tau: [B, C, H, W]
        # dt_broadcast * 上面: [B, C, H, W]
        h = h_prev + dt_broadcast * ((-h_prev + h_tilde) / tau)  # [B, C, H, W]
        
        return h

    def get_modulation_stats(self):
        return {
            'base_tau_mean': self.base_time_constant.mean().item(),
            'base_dt': self.base_dt_scalar.item(),
        }


# ==================== EfficientLiquidBlock_B_CLSM_Trainable_Adaptive ====================
class EfficientLiquidBlock_B_CLSM_Trainable_Adaptive(nn.Module):
    """
    使用自适应CLSM的液态块
    仅修改：传递rgb_global参数
    """
    def __init__(self, in_channels, hidden_dim=256, num_units=4, rgb_channels=832):
        super().__init__()

        self.channel_adapter = nn.Sequential(
            BasicConv3d(in_channels, 512, kernel_size=1, stride=1),
            BasicConv3d(512, hidden_dim, kernel_size=1, stride=1),
        )

        # 使用自适应版本的液态单元
        self.local_units = nn.ModuleList(
            [ModulatedLiquidCell_Trainable_Adaptive(hidden_dim, hidden_dim) for _ in range(num_units)]
        )

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout3d(p=0.3)
        self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # 使用自适应CLSM
        self.clsm = CrossModalLiquidStateModulator_Adaptive(rgb_channels=rgb_channels)
        
        self.output_fusion = nn.Conv2d(hidden_dim * num_units, hidden_dim, 1)

        nn.init.xavier_uniform_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x, rgb_features=None, states=None):
        x = self.channel_adapter(x)
        B, C, T, H, W = x.shape
        device = x.device

        # 计算显著性和全局RGB特征
        rgb_global = None
        saliency = None
        
        if rgb_features is not None:
            if (rgb_features.size(2) != T or
                rgb_features.size(3) != H or
                rgb_features.size(4) != W):
                rgb_features = F.interpolate(
                    rgb_features, size=(T, H, W),
                    mode="trilinear", align_corners=False
                )
            saliency, rgb_global = self.clsm.compute_saliency(rgb_features)  # [B, T], [B, C]
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
                
                # 传递rgb_global给单元
                h_i = unit(
                    combined_input, h_locals[i],
                    clsm_modulator=self.clsm,
                    saliency_coeff=s_t,
                    rgb_global=rgb_global  # 新增参数
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
        return out, h_locals, modulation_stats_list


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
        # 这里要使用 self.model.config.image_size 和 self.model.config.patch_size
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


# ==================== 完整模型 - 自适应版本 ====================
class My_LNN_based_Model_B_CLSM_Fixed3_Adaptive(nn.Module):
    """
    完整模型 - 自适应调制强度版本
    基于Fixed3可训练版的最小修改
    """
    def __init__(self, num_classes=101,num_frames=8):
        super().__init__()
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

        # 自适应CLSM-LNN（替换为自适应版本）
        self.flow_liquid = EfficientLiquidBlock_B_CLSM_Trainable_Adaptive(
            in_channels=832,
            hidden_dim=256,
            rgb_channels=832,
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
            elif isinstance(m, CrossModalLiquidStateModulator_Adaptive):
                # 自适应CLSM有自己的初始化，不需要额外处理
                pass

    def reset_states(self):
        self._flow_states = None

    def forward(self, rgb, flow, reset_states=False):
        B = rgb.size(0)
        device = rgb.device

        if reset_states:
            self._flow_states = None

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

        # 自适应LNN前向
        flow_liquid_out, self._flow_states, modulation_stats = self.flow_liquid(
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
            return logits, modulation_stats
        return logits

    def get_modulation_loss(self, modulation_stats):
        """调制损失计算 - 确保β网络有梯度流动"""
        # 直接从模型获取β网络
        beta_tau_net = self.flow_liquid.clsm.beta_tau_net
        beta_dt_net = self.flow_liquid.clsm.beta_dt_net
        
        # 对网络权重添加小正则化
        beta_net_loss = 0.0
        for param in beta_tau_net.parameters():
            beta_net_loss += torch.sum(param ** 2) * 0.0001
        for param in beta_dt_net.parameters():
            beta_net_loss += torch.sum(param ** 2) * 0.0001
        
        # s_t的动态范围损失（与原版相同）
        s_t_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if modulation_stats and len(modulation_stats) > 0:
            s_values = []
            for stat in modulation_stats:
                if 's_t' in stat:
                    s_values.append(stat['s_t'])
            if len(s_values) > 1:
                s_std = torch.tensor(s_values, device=next(self.parameters()).device).std()
                s_t_loss = torch.relu(0.2 - s_std) * 0.0001
        
        return beta_net_loss + s_t_loss


# ==================== 测试验证 ====================
if __name__ == "__main__":
    print("="*80)
    print("TimeSformer 自适应调制强度版本（修复版）")
    print("基于Fixed3可训练版的最小修改")
    print("核心改进: 固定β → 动态β (内容自适应)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = My_LNN_based_Model_B_CLSM_Fixed3_Adaptive(num_classes=101, num_frames=32).to(device)
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
    
    logits, mod_stats = model(rgb, flow, reset_states=True)
    loss = logits.mean()
    loss.backward()
    
    print(f"   输出logits形状: {logits.shape}")
    print(f"   损失值: {loss.item():.4f}")
    
    # 检查β网络的梯度
    clsm = model.flow_liquid.clsm
    print("\n📊 β网络梯度检查:")
    print(f"   beta_tau_net 有梯度: {clsm.beta_tau_net[2].weight.grad is not None}")
    print(f"   beta_dt_net 有梯度: {clsm.beta_dt_net[2].weight.grad is not None}")
    
    # 打印调制统计信息
    if mod_stats:
        print("\n📊 调制统计信息 (前3个时间步):")
        for i, stat in enumerate(mod_stats[:3]):
            print(f"   第{i+1}个时间步: base_tau_mean={stat['base_tau_mean']:.3f}, base_dt={stat['base_dt']:.3f}")
            if 's_t' in stat:
                print(f"                s_t={stat['s_t']:.3f}")
    
    print("\n✅ 自适应版本模型（修复版）初始化完成")