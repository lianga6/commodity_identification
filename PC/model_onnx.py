# 模型转换（.pt模型转换为.onnx格式）
from ultralytics import YOLO

# 加载模型
model = YOLO(r"runs/detect/train/weights/best.pt")

# 导出模型
model.export(
    format="onnx",          # 导出为 ONNX 格式
    imgsz=640,              # 输入图像尺寸
    half=False,              # 启用 FP16 量化
    int8=False,             # 是否启用 INT8 量化（根据需求启用）
    dynamic=True,           # 启用动态输入尺寸
    simplify=True,          # 简化模型
    opset=12                # 指定 ONNX opset 版本
)
