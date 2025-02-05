from ultralytics import YOLO

if __name__ == '__main__':
    # 设置环境变量
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只显示 NVIDIA 的独立 GPU

    # 加载训练模型
    a = YOLO('last.pt')

    # 开始训练模型
    a.train(
        data='D:\picture\CI3\data.yaml',  # 数据集配置文件路径
        lr0=0.001,          # 初始学习率
        # lrf=0.1,            # 学习率衰减率
        imgsz=640,         # 训练图片尺寸，官方推荐640
        batch=8,          # 每次训练的批次大小
        epochs=300,        # 训练轮数
        device='0',        # 训练设备，指定使用第一个 GPU（此时只有 NVIDIA 的独立 GPU 可用）
        amp=False,           # 混合精度训练
        augment=True  # 开启数据增强
    )

    print('训练完成')

