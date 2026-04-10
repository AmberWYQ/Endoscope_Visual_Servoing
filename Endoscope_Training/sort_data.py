import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from configs import config

def organize_dataset(src_dir, dest_dir):
    # 创建目标总目录
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取所有 mp4 文件
    video_files = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    
    video_count = 0
    for video_file in video_files:
        video_count += 1
        # 获取不带后缀的文件名 (例如 "1")
        file_id = os.path.splitext(video_file)[0]
        print("file_id: ", file_id)
        new_id = file_id.replace("video_", "data_")
        csv_file = f"{new_id}.csv"
        
        # 检查对应的 CSV 是否存在
        src_video_path = os.path.join(src_dir, video_file)
        src_csv_path = os.path.join(src_dir, csv_file)
        
        if os.path.exists(src_csv_path):
            # 1. 为这组数据创建一个独立的文件夹 (例如 "video_1")
            folder_id = f"{video_count:02}"
            folder_name = f"video_{folder_id}"
            target_folder = os.path.join(dest_dir, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            # 2. 移动并重命名文件 (统一命名方便后续脚本处理)
            shutil.copy(src_video_path, os.path.join(target_folder, "video.mp4"))
            shutil.copy(src_csv_path, os.path.join(target_folder, "data.csv"))
            
            print(f"✅ 已整理: {folder_name}")
        else:
            print(f"⚠️ 跳过: 找不到 {video_file} 对应的 CSV 文件")

def process_all_data(base_dir):
    all_dfs = []
    video_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    target_fps = config.CameraConfig.fps  # <--- 這裡設定你想要的固定 FPS
    print("target_fps: ", target_fps)
    for folder in tqdm(video_folders, desc="正在處理影片並對齊數據"):
        video_path = os.path.join(base_dir, folder, 'video.mp4')
        csv_path = os.path.join(base_dir, folder, 'data.csv')
        frames_dir = os.path.join(base_dir, folder, 'frames')
        
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
            
        # 1. 獲取影片資訊
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0: original_fps = 30 # 防止讀取失敗
        
        # 計算步長：例如 60fps 轉 20fps，步長就是 3 (每 3 幀取 1 幀)
        stride = max(1, round(original_fps / target_fps))
        print(f"\n影片 {folder}: 原始 FPS={original_fps:.2f}, 目標 FPS={target_fps}, 使用步長={stride}")

        # 2. 抽幀邏輯
        frame_idx = 0
        saved_count = 0
        saved_frame_indices = [] # 記錄我們到底存了原始影片的哪幾幀

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 只有符合步長的幀才保存
            if frame_idx % stride == 0:
                img_name = f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, img_name), frame)
                saved_frame_indices.append(frame_idx) # 記錄這個索引
                saved_count += 1
            frame_idx += 1
        cap.release()
        
        # 3. 讀取 CSV 並進行「時間對齊」
        df = pd.read_csv(csv_path)
        
        # 核心對齊邏輯：
        # 我們保存了影片的第 [0, stride, 2*stride...] 幀
        # 所以 CSV 也必須取對應的第 [0, stride, 2*stride...] 行
        valid_indices = [i for i in saved_frame_indices if i < len(df)]
        df_aligned = df.iloc[valid_indices].copy()
        
        # 修正圖片路徑：現在圖片編號是 0, 1, 2...對應我們存下來的圖
        image_paths = [os.path.join(frames_dir, f"frame_{i:05d}.jpg") for i in range(len(df_aligned))]
        df_aligned['image_path'] = image_paths
        
        all_dfs.append(df_aligned)
        print(f"整理完成: 提取了 {len(df_aligned)} 個樣本")

    # 4. 合并
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(os.path.join(base_dir, 'master_bc_data.csv'), index=False)
        print(f"\n全部處理完成！總樣本數: {len(master_df)}")


# --- 使用设置 ---
# 修改为你存放原始文件的路径
source_folder = '/Users/weiyuqing/Desktop/endo_low_level_control/collected_data' 
# 修改为你想要生成的结构化路径
output_folder = '/Users/weiyuqing/Desktop/dataset'

organize_dataset(source_folder, output_folder)

process_all_data(output_folder)