#生成NPY文件和类别文件
import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import pytz
from multiprocessing import Pool, cpu_count
from pathlib import Path
import uuid
import re

# 配置参数
DATASET_ROOT = Path('Dataset/OlympicSports')
VIDEOS_PATH = DATASET_ROOT / 'videos'  # 原始视频路径，按类别文件夹存放
PROCESSED_PATH = DATASET_ROOT / 'processed'
OUTPUT_PATH = PROCESSED_PATH / 'npy_files'  # NPY文件输出路径
FRAME_SIZE = (224, 224)
NUM_WORKERS = max(1, cpu_count() - 1)

# 全局变量
class_to_idx = {}
class_video_counters = defaultdict(int)

def cleanup_temp_files():
    """清理临时文件目录"""
    temp_dir = OUTPUT_PATH / 'temp'
    if temp_dir.exists():
        for f in temp_dir.glob('*'):
            try:
                if f.is_file():
                    f.unlink()
            except Exception as e:
                print(f"清理失败 {f.name}: {str(e)}")
        print(f"已清理临时目录: {temp_dir}")

def extract_frames(video_path):
    """从视频中提取帧并调整大小"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)

def compute_optical_flow(frames, clip_threshold=20.0):
    """计算光流并归一化到 [-1, 1]"""
    flow_frames = []
    if len(frames) == 0:
        return np.array([], dtype=np.float32)
    
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        # 裁剪并归一化到 [-1, 1]
        flow = np.clip(flow, -clip_threshold, clip_threshold) / clip_threshold
        flow_frames.append(flow.astype(np.float32))
        prev_frame = curr_frame

    # 保持与RGB帧数一致（复制最后一帧光流）
    if flow_frames:
        flow_frames.append(flow_frames[-1])
    else:
        # 如果视频只有一帧，创建全零光流
        h, w = frames[0].shape[:2]
        flow_frames = [np.zeros((h, w, 2), dtype=np.float32)] * len(frames)

    return np.array(flow_frames)

def sanitize_filename(name):
    """清理文件名，移除可能引起问题的字符"""
    # 移除文件名中的特殊字符，只保留字母、数字、下划线和连字符
    return re.sub(r'[^\w\-_]', '_', name)

def process_single_video(args):
    """处理单个视频文件"""
    video_path, class_name, class_idx = args
    try:
        video_path = Path(video_path)
        
        # 获取类内视频编号（三位数，从001开始）
        global class_video_counters
        class_video_counters[class_name] += 1
        video_counter = class_video_counters[class_name]
        video_number = f"{video_counter:03d}"  # 格式化为三位数
        
        # 清理原始文件名
        original_name = video_path.stem
        sanitized_name = sanitize_filename(original_name)
        
        # 生成符合要求的文件名（使用[]包围类别名和原始文件名）
        #rgb_filename = f"{class_idx}_[{class_name}]_[{sanitized_name}]_{video_number}_RGB.npy"
        #flow_filename = f"{class_idx}_[{class_name}]_[{sanitized_name}]_{video_number}_FLOW.npy"
        rgb_filename = f"{class_idx}_[{class_name}]_[{sanitized_name}]_RGB.npy"
        flow_filename = f"{class_idx}_[{class_name}]_[{sanitized_name}]_FLOW.npy"
        
        rgb_file = OUTPUT_PATH / rgb_filename
        flow_file = OUTPUT_PATH / flow_filename

        # 检查文件是否已存在
        if rgb_file.exists() and flow_file.exists():
            print(f"文件已存在，跳过: {rgb_filename}")
            return 1, class_name, video_number

        # 提取帧和计算光流
        print(f"正在处理: {class_name} - {video_number} - {video_path.name}")
        rgb_frames = extract_frames(str(video_path))
        if len(rgb_frames) < 1:
            print(f"跳过：无法提取帧 ({video_path.name})")
            return 0, class_name, video_number
            
        flow_frames = compute_optical_flow(rgb_frames)

        # 使用临时文件确保原子性写入
        temp_dir = OUTPUT_PATH / 'temp'
        temp_prefix = uuid.uuid4().hex
        temp_rgb = temp_dir / f"{temp_prefix}_RGB"
        temp_flow = temp_dir / f"{temp_prefix}_FLOW"
        
        np.save(str(temp_rgb) + '.tmp', rgb_frames, allow_pickle=False)
        np.save(str(temp_flow) + '.tmp', flow_frames, allow_pickle=False)
        
        # 重命名为最终文件名
        Path(str(temp_rgb) + '.tmp.npy').rename(rgb_file)
        Path(str(temp_flow) + '.tmp.npy').rename(flow_file)
        
        print(f"处理完成: {class_name} - {video_number} - {video_path.name}")
        
        return 1, class_name, video_number
        
    except Exception as e:
        print(f"处理失败 {video_path.name}: {str(e)}")
        return 0, class_name, "000"

def generate_class_files():
    """生成类别索引文件"""
    # 扫描视频文件夹，获取所有类别
    class_names = [d.name for d in VIDEOS_PATH.iterdir() if d.is_dir()]
    class_names.sort()
    
    # 创建类别到索引的映射（从0开始）
    global class_to_idx
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # 生成classInd.txt（序号+1）
    class_ind_file = PROCESSED_PATH / 'OlympicSports_classInd.txt'
    with open(class_ind_file, 'w', encoding='utf-8') as f:
        for idx, cls in enumerate(class_names, 1):  # 从1开始编号
            f.write(f"{idx} {cls}\n")
    print(f"类别索引文件已生成: {class_ind_file}")
    
    # 生成labels.txt（只有类别名称）
    labels_file = PROCESSED_PATH / 'OlympicSports_labels.txt'
    with open(labels_file, 'w', encoding='utf-8') as f:
        for cls in class_names:
            f.write(f"{cls}\n")
    print(f"标签文件已生成: {labels_file}")
    
    return class_names, class_to_idx

def get_all_videos():
    """获取所有视频文件"""
    videos = []
    class_names, class_to_idx = generate_class_files()
    
    for class_name in class_names:
        class_path = VIDEOS_PATH / class_name
        if class_path.exists():
            for video_file in class_path.glob('*.avi'):
                videos.append((video_file, class_name, class_to_idx[class_name]))
    
    return videos

def generate_hmdb51_npy():
    """生成HMDB51的NPY文件"""
    # 创建输出目录
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / 'temp').mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # 获取所有视频文件
    print("正在扫描视频文件...")
    all_videos = get_all_videos()
    
    if not all_videos:
        print("未找到任何视频文件！")
        return
    
    print(f"找到 {len(all_videos)} 个视频文件")
    
    # 重置计数器
    global class_video_counters
    class_video_counters = defaultdict(int)
    
    successful_count = 0
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_video, all_videos),
            total=len(all_videos),
            desc="处理视频文件"
        ))
    
    successful_count = sum(result[0] for result in results)
    
    print(f"\n处理完成: {successful_count}/{len(all_videos)}")
    print(f"NPY文件已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    print(f"当前工作目录: {Path.cwd()}")
    print(f"视频路径: {VIDEOS_PATH.resolve()}")
    print(f"输出路径: {OUTPUT_PATH.resolve()}")

    # 检查路径是否存在
    if not VIDEOS_PATH.exists():
        raise FileNotFoundError(f"视频路径不存在: {VIDEOS_PATH}")

    # 测试写入权限
    test_file = OUTPUT_PATH / 'write_test.tmp'
    try:
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"无法写入输出目录: {OUTPUT_PATH}") from e

    cleanup_temp_files()
    start_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        generate_hmdb51_npy()
    except Exception as e:
        print(f"程序出错: {str(e)}")
        raise
    finally:
        cleanup_temp_files()

    end_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time}")