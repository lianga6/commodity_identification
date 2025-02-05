# 系统介绍
该系统名为商品自动识别结账系统，系统包括商品识别模块与付款提示模块。通过摄像头采集柜台商品图像，借助OpenCV进行图像处理，利用yolov8模型精准识别商品种类。系统自动计算商品总价，并在显示屏上提示顾客应付款金额。该系统旨在简化超市结账流程，提升购物效率与顾客体验，拟在树莓派上部署运行，实现高效、便捷的商品识别与付款功能。
# 代码文件说明
- PC文件夹：这个文件夹的文件是在你的电脑端训练模型的，其中train.py是用yolov8训练模型的代码；model_onnx.py是将训练好的pt格式的模型转化为可在树莓派上部署的onnx格式的模型；jtt.py是将json格式的标注文件转化为yolov8训练所需要的txt格式的标注文件的脚本（针对多边形）。
- 其余代码：其余代码都是在树莓派上运行的代码，其中collect_pic.py是用来收集所需要的图片数据，按x拍照，按q退出；camera_test.py是用来测试摄像头是否好用的代码；detect.py主要作用是解析onnx模型的网络，方便使用；主程序是ci.py。

# 该项目衍生的博文
- https://blog.csdn.net/pai_da_xing8/article/details/145185421
- https://blog.csdn.net/pai_da_xing8/article/details/145311505
- https://blog.csdn.net/pai_da_xing8/article/details/145392890
- https://blog.csdn.net/pai_da_xing8/article/details/145385427
