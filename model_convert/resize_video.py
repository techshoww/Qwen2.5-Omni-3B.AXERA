import subprocess
import os

def resize_video(input_path, output_path, width, height):
    """
    修改视频的分辨率并保持音频不变。
    
    参数:
        input_path (str): 输入视频文件路径。
        output_path (str): 输出视频文件路径。
        width (int): 目标宽度。
        height (int): 目标高度。
    """
    # 构造 ffmpeg 命令
    command = [
        "ffmpeg",
        "-i", input_path,          # 输入文件
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",  # 调整分辨率并填充黑边
        "-c:a", "copy",            # 复制音频流，不重新编码
        output_path                # 输出文件
    ]
    
    try:
        # 执行命令
        subprocess.run(command, check=True)
        print(f"视频已成功调整为 {width}x{height} 并保存到 {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"处理视频时出错: {e}")

# 示例调用
if __name__ == "__main__":
    input_file = "1.mp4"  # 输入视频文件路径
    output_file = "2.mp4"  # 输出视频文件路径
    target_width = 308
    target_height = 308

    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在！")
    else:
        resize_video(input_file, output_file, target_width, target_height)