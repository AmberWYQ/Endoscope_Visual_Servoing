import pandas as pd
import cv2
import os
import subprocess

def fast_sync_video_csv(video_path, csv_path, output_dir):
    # 1. 创建输出目录
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 2. 读取 CSV 数据
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    print(f"CSV 包含 {num_rows} 条记录")

    # 3. 使用 FFmpeg 快速抽帧 (确保帧数与 CSV 行数大致匹配)
    # 计算视频时长
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # 自动计算需要抽取的帧率 (使其等于 CSV 总行数 / 视频时长)
    target_fps = num_rows / duration
    print(f"视频时长: {duration:.2f}s, 原帧率: {fps}, 目标同步帧率: {target_fps:.2f}")

    # 调用 FFmpeg 抽帧
    print("正在抽帧，请稍候...")
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={target_fps}',
        '-q:v', '2',  # 高质量保存
        os.path.join(frames_dir, 'frame_%05d.jpg'),
        '-y'
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # 4. 关联图片路径到 CSV
    # 获取实际抽取的图片列表
    extracted_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    # 对齐长度（防止由于精度导致的 1-2 帧误差）
    min_len = min(len(extracted_frames), num_rows)
    df = df.iloc[:min_len].copy()
    df['image_path'] = [os.path.abspath(os.path.join(frames_dir, f)) for f in extracted_frames[:min_len]]

    # 5. 保存新的 CSV
    new_csv_path = os.path.join(output_dir, "bc_ready_data.csv")
    df.to_csv(new_csv_path, index=False)
    print(f"完成！新 CSV 已保存至: {new_csv_path}")
    print(f"最终样本数: {min_len}")


if __name__ == "__main__":
    # 使用示例
    fast_sync_video_csv(
        video_path="/Users/weiyuqing/Desktop/endo_low_level_control/example/video_20251219-182747.mp4", 
        csv_path="/Users/weiyuqing/Desktop/endo_low_level_control/example/data_20251219-182747.csv", 
        output_dir="./frames"
    )