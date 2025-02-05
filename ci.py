
import cv2
import numpy as np
from collections import defaultdict
from detect import YOLOv8
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库

# 配置参数
MODEL_PATH = "best2.onnx"
CONF_THRES = 0.8
IOU_THRES = 0.5

# 商品价格表（元/件）
PRICES = {
    "可乐": 3.0,
    "东鹏特饮": 5.0,
    "红笔": 2.0,
    "黑笔": 2.5,
    "旺旺雪饼": 1.0
}

# 界面参数
PANEL_WIDTH = 400  # 信息面板宽度
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 中文字体路径

class PaymentSystem:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        # 初始化YOLOv8检测器
        self.detector = YOLOv8(MODEL_PATH, CONF_THRES, IOU_THRES)
        
        # 状态变量
        self.shopping_cart = defaultdict(int)
        self.total_price = 0.0

        # 加载中文字体
        self.font = ImageFont.truetype(FONT_PATH, 24)  # 字体大小24

    def process_frame(self, frame):
        """处理单帧图像并返回检测结果"""
        # 执行目标检测
        flipped = cv2.flip(frame, -1)
        processed_img = self.detector.main(flipped)
        return processed_img

    def update_shopping_cart(self, current_detections):
        """更新购物车信息"""
        # 统计当前帧检测结果
        frame_counts = defaultdict(int)
        for class_id in current_detections:
            class_name = self.detector.classes[class_id]
            frame_counts[class_name] += 1
        
        # 更新累计购物车（实际应用应添加去重逻辑）
        self.shopping_cart.clear()
        for name, count in frame_counts.items():
            self.shopping_cart[name] += count

        # 计算总价
        self.total_price = sum(PRICES[name] * count 
                              for name, count in self.shopping_cart.items())

    def draw_chinese_text(self, img, text, position, font, color):
        """
        在图像上绘制中文字符
        """
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制中文字符
        draw.text(position, text, font=font, fill=color)
        
        # 将PIL图像转换回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_info_panel(self, height):
        """创建右侧信息面板"""
        panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
        
        # 标题
        panel = self.draw_chinese_text(panel, "智能收银系统", (20, 40), self.font, (255, 255, 255))
        
        # 价格信息
        y_pos = 100
        panel = self.draw_chinese_text(panel, "当前商品:", (20, y_pos), self.font, (200, 200, 255))
        y_pos += 50
        
        for name, count in self.shopping_cart.items():
            text = f"{name} x {count} (单价: {PRICES[name]} 元)"
            panel = self.draw_chinese_text(panel, text, (30, y_pos), self.font, (150, 255, 150))
            y_pos += 40
        
        # 总价信息
        y_pos += 30
        total_text = f"应付总额: {self.total_price:.2f} 元"
        panel = self.draw_chinese_text(panel, total_text, (20, y_pos), self.font, (100, 255, 100))
        
        return panel

    def run(self):
        """主运行循环"""
        # #开启全屏模式
        # cv2.namedWindow("Automatic Checkout System", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Automatic Checkout System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取视频帧")
                break

            try:
                # 处理图像并获取检测结果
                processed_img = self.process_frame(frame)
                
                # 获取当前检测结果
                current_detections = self.detector.detected_classes
                self.update_shopping_cart(current_detections)

                # 创建组合界面
                h, w = processed_img.shape[:2]
                info_panel = self.create_info_panel(h)
                combined = np.hstack((processed_img, info_panel))

                # 显示结果
                cv2.imshow("Automatic Checkout System", combined)

                # 退出机制
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"处理过程中发生错误: {str(e)}")
                break

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = PaymentSystem()
    system.run()


