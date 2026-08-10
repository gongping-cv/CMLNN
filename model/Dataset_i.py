import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class UCF101Dataset(Dataset):
    def __init__(self, rgb_files, flow_files, clip_len=64, training=True, num_clips=1):
        self.rgb_files = rgb_files
        self.flow_files = flow_files
        self.clip_len = clip_len
        self.training = training
        self.num_clips = num_clips

        # 简化的CPU预处理（只做必要的）
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return min(len(self.rgb_files), len(self.flow_files))

    def sample_clip(self, frames_tensor, T_total):
        clips = []
        
        if self.training or self.num_clips == 1:
            for _ in range(self.num_clips):
                if self.training:
                    start = np.random.randint(0, T_total - self.clip_len + 1)
                else:
                    start = (T_total - self.clip_len) // 2
                clip = frames_tensor[:, start:start + self.clip_len, :, :]
                clips.append(clip)
        else:
            starts = np.linspace(0, T_total - self.clip_len, self.num_clips).astype(int)
            for s in starts:
                clip = frames_tensor[:, s:s + self.clip_len, :, :]
                clips.append(clip)
                
        return clips

    def __getitem__(self, idx):
        # 只做最基本的加载和预处理
        rgb = np.load(self.rgb_files[idx])  # [T, H, W, 3]
        flow = np.load(self.flow_files[idx])  # [T, H, W, 2]
        label = int(self.rgb_files[idx].split('/')[-1].split('_')[0])

        rgb = torch.from_numpy(rgb).permute(3, 0, 1, 2).float() / 255.0
        flow = torch.from_numpy(flow).permute(3, 0, 1, 2).float()

        T_total = rgb.shape[1]

        if T_total < self.clip_len:
            pad_len = self.clip_len - T_total
            pad_rgb = rgb[:, -1:, :, :].repeat(1, pad_len, 1, 1)
            pad_flow = flow[:, -1:, :, :].repeat(1, pad_len, 1, 1)
            rgb = torch.cat([rgb, pad_rgb], dim=1)
            flow = torch.cat([flow, pad_flow], dim=1)
            T_total = self.clip_len

        rgb_clips = self.sample_clip(rgb, T_total)
        flow_clips = self.sample_clip(flow, T_total)

        # 不再在CPU上做增强，直接返回原始数据
        processed_rgb = torch.stack(rgb_clips)
        processed_flow = torch.stack(flow_clips)
        
        return processed_rgb, processed_flow, label
