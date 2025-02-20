# 商品自动识别结账系统
# 产品背景
在传统超市的自助结账中，顾客需要一件一件地扫描商品条形码，既费时又麻烦。据统计，每位顾客平均结账时间超过3分钟，其中大部分时间都花在扫描条码上。尤其在购物高峰时段，排队结账更是让人头疼。

该系统解决了这些问题。它采用智能识别技术，可以在1秒内同时识别4到6件商品，速度比传统方式快两倍以上。如果商品太多，还可以分批识别，一次结账。更棒的是，系统完全不需要条形码，省去了贴条码的人力物力，为超市节省了大量成本。

此外，我们还引入了智能超市语音助手，通过调用DeepSeek-v3模型，为顾客提供实时语音支持。无论是询问商品信息、了解促销活动，还是获得结账指引，语音助手都能快速响应，让购物体验更加轻松愉快。

系统还会自动记录每笔交易的时间、商品信息和金额，方便管理者查看每天的销售情况。这套集智能识别、语音交互和数据管理于一体的系统，不仅让顾客结账更快捷，还能帮助超市实现智能化管理，提升整体运营效率。
 # 系统介绍
本系统为基于计算机视觉技术的智能商品识别结账系统，由商品识别模块、交互控制模块及交易管理模块构成。系统通过调用摄像头实时采集商品图像，采用OpenCV进行图像预处理，结合YOLOv8深度学习模型实现高精度商品分类识别。优化后的系统支持多批次商品处理模式，用户可通过'z'键逐批锁定商品清单，'x'键触发智能结账并生成结构化交易日志（含时间戳、商品明细、数量及金额），'v'键'唤出智能超市助手，c'键实现交易状态重置以及语音助手的重置，'q'键安全退出程序。系统部署于树莓派硬件平台，提供可视化交互界面，显著提升收银效率并实现交易数据数字化管理，适用于商超、便利店等零售场景。

# 视频演示（1.5倍速）
[观看视频](https://github.com/user-attachments/assets/efa1eca9-9619-48ae-86c4-f61e495c46e7 "点我观看演示视频（1.5倍速）")

#项目流程
运行./run_ci.sh后
- 按下 'z' 键时，确认当前批次的商品。
- 再次按下 'z' 键时，确认这批次的商品。
- 当所有批次都识别完后，按下 'x' 键时，开始结账，并保存当前交易记录到日志中。
- 再按下 'c' 键之前的任意时刻，都可通过按下 'v' 键与超市助手对话。
- 按下 'c' 键后开始下一次交易，超市助手初始化。
- 按下 'q' 键时，退出程序。

# 代码文件说明
- PC文件夹：这个文件夹的文件是在你的电脑端训练模型的，其中train.py是用yolov8训练模型的代码；model_onnx.py是将训练好的pt格式的模型转化为可在树莓派上部署的onnx格式的模型；jtt.py是将json格式的标注文件转化为yolov8训练所需要的txt格式的标注文件的脚本（针对多边形）。
- 其余代码：其余代码都是在树莓派上运行的代码，其中collect_pic.py是用来收集所需要的图片数据，按x拍照，按q退出；camera_test.py是用来测试摄像头是否好用的代码；detect.py主要作用是解析onnx模型的网络，方便使用；DS系列都是语音助手，其中DS10是最好的一个；ci2.py是商品识别与结账系统的代码。
- 注意：（流式的更顺畅，没有副作用；智能静音检测能提高与智能体的沟通速度，但是要进行一步噪声阈值检测，也就是具体环境需要调试一下）
   - DS系列与ci2.py都是主文件，我是用一个bash脚本同时让他们在后台运行，也就是说，其实这两个系统可以独立工作。
   - DS10.py是流式的，还有智能静音检测。
   - DS11.py是流式的，没有智能静音检测。
   - DS01.py是非流式的，有智能静音检测。
   - DS00.py是流式的，没有智能静音检测。

#API资源
- deepseek的API
- dashscope的API： https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key

# 模型说明
- best1.onnx:只包含“可乐”和“东鹏特饮”的识别。
- best2.onnx：包含“可乐”，“东鹏特饮”，“黑笔”，“红笔”以及“旺旺雪饼”的识别。

# 该项目衍生的博文
- https://blog.csdn.net/pai_da_xing8/article/details/145185421
- https://blog.csdn.net/pai_da_xing8/article/details/145311505
- https://blog.csdn.net/pai_da_xing8/article/details/145392890
- https://blog.csdn.net/pai_da_xing8/article/details/145385427

# 改动
比第二版多了智能语音助手。

# 版权问题
若用于商业用途需经过本人的同意，参照License

