import cv2
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict
from detect import YOLOv8
from PIL import Image, ImageDraw, ImageFont

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
        self.current_cart = defaultdict(int)       # 当前批次商品
        self.frame_counts = defaultdict(int)       # 当前帧检测到的数量
        self.total_cart = defaultdict(int)         # 累计确认商品
        self.total_price = 0.0                     # 累计总金额
        self.transaction_active = True             # 交易是否处于活动状态

        # 加载中文字体
        self.font = ImageFont.truetype(FONT_PATH, 24)
        self.small_font = ImageFont.truetype(FONT_PATH, 18)

    def process_frame(self, frame):
        """处理单帧图像并返回检测结果"""
        flipped = cv2.flip(frame, -1)
        processed_img = self.detector.main(flipped)
        return processed_img

    def update_shopping_cart(self, current_detections):
        """更新当前帧商品计数"""
        self.frame_counts = defaultdict(int)
        for class_id in current_detections:
            class_name = self.detector.classes[class_id]
            self.frame_counts[class_name] += 1

    def draw_chinese_text(self, img, text, position, font, color):
        """在图像上绘制中文字符"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_info_panel(self, height):
        """创建右侧信息面板"""
        panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
        
        # 标题
        panel = self.draw_chinese_text(panel, "智能收银系统", (20, 20), self.font, (255, 255, 255))
        panel = self.draw_chinese_text(panel, "操作提示：", (20, 60), self.small_font, (200, 200, 255))
        panel = self.draw_chinese_text(panel, "Z-确认批次  X-总结账  C-新交易  Q-退出", (30, 90), self.small_font, (150, 200, 255))

        y_pos = 140
        
        # 当前批次商品（仅显示当前帧的商品）
        panel = self.draw_chinese_text(panel, "当前批次商品:", (20, y_pos), self.small_font, (200, 200, 255))
        y_pos += 40
        for name, count in self.frame_counts.items():
            price = PRICES[name] * count
            text = f"{name} x {count} = {price:.1f}元"
            panel = self.draw_chinese_text(panel, text, (30, y_pos), self.small_font, (150, 255, 150))
            y_pos += 30
        
        # 已确认商品
        panel = self.draw_chinese_text(panel, "已确认商品:", (20, y_pos), self.small_font, (200, 200, 255))
        y_pos += 40
        for name, count in self.total_cart.items():
            price = PRICES[name] * count
            text = f"{name} x {count} = {price:.1f}元"
            panel = self.draw_chinese_text(panel, text, (30, y_pos), self.small_font, (150, 255, 150))
            y_pos += 30
        
        # 总价信息
        y_pos += 20
        total_text = f"应付总额: {self.total_price:.2f} 元"
        panel = self.draw_chinese_text(panel, total_text, (20, y_pos), self.font, (100, 255, 100))
        
        return panel

    def confirm_current_batch(self):
        """确认当前批次"""
        if not self.transaction_active:
            return  # 交易不处于活动状态时，不执行任何操作

        # 将当前帧的商品数量累加到当前批次购物车
        for name, count in self.frame_counts.items():
            self.current_cart[name] += count
        self.frame_counts.clear()
        
        # 将当前批次购物车的商品累加到已确认商品
        for name, count in self.current_cart.items():
            self.total_cart[name] += count
        
        # 清空当前批次购物车
        self.current_cart.clear()
        
        # 更新应付总额
        self.total_price = sum(PRICES[name] * count for name, count in self.total_cart.items())
        print("当前批次已确认，已累加到已确认商品")

    def save_transaction(self):
        """保存交易记录"""
        if not self.transaction_active:
            return  # 交易不处于活动状态时，不执行任何操作

        if not self.total_cart:
            print("没有需要保存的交易记录")
            return

        # 计算总结账金额
        self.total_price = sum(PRICES[name] * count for name, count in self.total_cart.items())
        
        # 保存到CSV文件
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        items = dict(self.total_cart)
        total = self.total_price
        
        # 写入CSV文件
        filename = "transaction_history.csv"
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Items", "Total"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({"Timestamp": timestamp, "Items": str(items), "Total": total})
        
        print(f"交易已保存至 {filename}")
        self.transaction_active = False  # 交易结束，关闭交易状态

    def reset_transaction(self):
        """重置交易"""
        self.total_cart.clear()
        self.current_cart.clear()
        self.frame_counts.clear()
        self.total_price = 0.0
        self.transaction_active = True   # 重新开启交易状态
        print("交易已重置，开始新交易")

    def run(self):
        """主运行循环"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取视频帧")
                    break

                # 处理图像并获取检测结果
                processed_img = self.process_frame(frame)
                self.update_shopping_cart(self.detector.detected_classes)

                # 创建组合界面
                h, w = processed_img.shape[:2]
                info_panel = self.create_info_panel(h)
                combined = np.hstack((processed_img, info_panel))

                # 显示结果
                cv2.imshow("Automatic Checkout System", combined)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('z'):
                    self.confirm_current_batch()
                elif key == ord('x'):
                    self.save_transaction()
                elif key == ord('c'):
                    self.reset_transaction()
                elif key == ord('q'):
                    break

        finally:
            # 释放资源
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = PaymentSystem()
    system.run()