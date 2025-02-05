# #收集数据集，按下x就会拍一张照片并保存到该目录的下一级目录picture下，并且输入图像尺寸imgsz=640,按下q退出
import cv2
import os

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_photo_counter(directory):
    """获取下一个照片计数器的值"""
    existing_files = os.listdir(directory)
    photo_files = [f for f in existing_files if f.endswith('.jpg')]
    if not photo_files:
        return 0
    last_photo = max(photo_files, key=lambda f: int(f.split('.')[0]))
    return int(last_photo.split('.')[0]) + 1

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头是否连接正确！")
        return

    # 设置图像尺寸
    imgsz = 640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imgsz)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imgsz)

    # 确保保存照片的目录存在
    picture_dir = "picture"
    ensure_directory_exists(picture_dir)

    # 获取下一个照片计数器的值
    photo_counter = get_next_photo_counter(picture_dir)

    print("按 'x' 键拍照并保存，按 'q' 键退出程序。")

    while True:
        # 从摄像头捕获一帧图像
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面，退出程序。")
            break

        # 上下反转图像
        frame = cv2.flip(frame, 0)  # 0 表示上下反转
        # 左右反转图像
        frame = cv2.flip(frame, 1)  # 1 表示左右反转

        # 显示图像
        cv2.imshow('Camera Feed', frame)

        # 检测按键
        key = cv2.waitKey(1) & 0xFF

        # 按下 'x' 键拍照并保存
        if key == ord('x'):
            # 生成文件名
            filename = os.path.join(picture_dir, f"{photo_counter}.jpg")
            # 保存图像
            cv2.imwrite(filename, frame)
            print(f"照片已保存到 {filename}")
            # 递增照片计数器
            photo_counter += 1

        # 按下 'q' 键退出
        elif key == ord('q'):
            break

    # 释放摄像头资源并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


