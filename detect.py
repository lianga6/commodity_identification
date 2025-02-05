import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# 定义类别字典，将类别ID映射到类别名称
classes = {
    0: '可乐', 1: '东鹏特饮', 2: '红笔' ,3: '黑笔', 4 :'旺旺雪饼'
}

class YOLOv8:
    """YOLOv8目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        初始化YOLOv8类的实例。
        
        参数:
            onnx_model: ONNX模型的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 加载类别名称
        self.classes = classes

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 用于存储检测到的类别ID
        self.detected_classes = []

        # 加载中文字体
        self.font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 字体路径
        try:
            self.font = ImageFont.truetype(self.font_path, 20)  # 字体大小20
        except Exception as e:
            print(f"无法加载字体文件: {self.font_path}，请检查路径是否正确。")
            raise e

    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别ID对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名称和置信度分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 使用PIL绘制中文字符
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 计算文本尺寸
        text_bbox = draw.textbbox((0, 0), label, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算文本位置
        text_x = int(x1)
        text_y = int(y1) - text_height if int(y1) - text_height > 0 else int(y1) + 10
        
        # 绘制文本背景
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill=tuple(map(int, color))
        )
        
        # 绘制文本
        draw.text((text_x, text_y), label, font=self.font, fill=(0, 0, 0))  # 黑色文本
        
        # 将PIL图像转换回OpenCV图像
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    def postprocess(self, input_image, output):
        """
        对模型的输出进行后处理，以提取边界框、置信度分数和类别ID。
        """
        # 转置并压缩输出以匹配预期的形状
        outputs = np.transpose(np.squeeze(output[0]))

        # 获取输出数组中的行数
        rows = outputs.shape[0]

        # 用于存储检测到的边界框、置信度分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组中的每一行
        for i in range(rows):
            # 从当前行中提取类别分数
            classes_scores = outputs[i][4:]

            # 找到类别分数中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大值大于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取具有最高分数的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行中提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算缩放后的边界框坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 将类别ID、置信度分数和边界框坐标添加到各自的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大值抑制以过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 清空检测到的类别列表
        self.detected_classes = []

        # 遍历非极大值抑制后选择的索引
        for i in indices:
            # 获取对应于索引的边界框、置信度分数和类别ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # 将检测到的类别ID添加到列表中
            self.detected_classes.append(class_id)

            # 在输入图像上绘制检测结果
            input_image = self.draw_detections(input_image, box, score, class_id)

        # 返回修改后的输入图像
        return input_image

    def preprocess(self, img):
        """
        在执行推理之前预处理输入图像。
        """
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = img.shape[:2]

        # 将图像的颜色空间从BGR转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以匹配输入形状
        img = cv2.resize(img, (self.input_width, self.input_height))

        # 通过除以255.0来规范化图像数据
        image_data = np.array(img) / 255.0

        # 转置图像，使通道维度为第一个维度
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度以匹配预期的输入形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def main(self, img):
        """
        使用ONNX模型执行推理并返回带有绘制检测结果的输出图像。
        """
        # 使用ONNX模型创建推理会话并指定执行提供程序
        session = ort.InferenceSession(self.onnx_model, providers=["CPUExecutionProvider"])

        # 获取模型输入
        model_inputs = session.get_inputs()

        # 打印模型输入形状以确认
        print(f"Model Input Shape: {model_inputs[0].shape}")

        # 手动指定输入宽度和高度（YOLOv8 默认为 640x640）
        self.input_width = 416
        self.input_height = 416

        # 预处理图像数据
        img_data = self.preprocess(img)

        # 使用预处理后的图像数据进行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # 对输出进行后处理以获得输出图像
        return self.postprocess(img, outputs)  # 输出图像


if __name__ == "__main__":
    # 固定参数
    model_path = "best1.onnx"  # 模型路径
    conf_thres = 0.6          # 置信度阈值
    iou_thres = 0.5           # IoU 阈值

    # 使用指定的参数创建YOLOv8类的实例
    detection = YOLOv8(model_path, conf_thres, iou_thres)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break

        # 翻转图像（上下和水平翻转）
        flipped_frame = cv2.flip(frame, -1)  # -1 表示同时水平和垂直翻转

        # 执行目标检测
        output_image = detection.main(flipped_frame)

        # 在窗口中显示输出图像
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", output_image)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()