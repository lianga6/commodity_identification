import cv2

def test_camera():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0 表示树莓派的默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头是否连接正确！")
        return

    print("摄像头已成功打开，按 'q' 键退出程序。")

    while True:
        # 从摄像头捕获一帧图像
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面，退出程序。")
            break

        # 显示图像
        cv2.imshow('Camera Feed', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头测试结束，资源已释放。")

if __name__ == "__main__":
    test_camera()