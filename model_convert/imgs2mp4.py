import cv2
import os

def images_to_mp4(image_folder, output_video, fps=2):
    """
    将指定文件夹中的图片转换为MP4视频
    :param image_folder: 图片文件夹路径（需按文件名顺序排列）
    :param output_video: 输出视频路径（必须以.mp4结尾）
    :param fps: 视频帧率，默认30帧/秒
    """
    # 获取所有图片并按文件名排序
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()  # 确保按文件名顺序处理[2,4](@ref)
    # images = images[75:90]

    # 读取第一张图片以获取尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    H,W,_ = first_image.shape
    # height, width, _ = first_image.shape
    height = H
    width = W
    size = (width, height)

    # 定义视频编码器（MP4需使用'mp4v'编码）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, size)

    # 逐帧写入图片
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        # frame = frame[H-512:H, 0:512]
        video.write(frame)  # 写入视频帧[1,4](@ref)

    video.release()
    print(f"视频已生成：{output_video}")

# 示例调用
if __name__ == "__main__":
    images_to_mp4(
        image_folder="demo_cv308",  # 替换为你的图片文件夹路径
        output_video="demo.mp4",
        fps=2
    )