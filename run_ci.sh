#!/bin/bash

# 并行运行两个Python脚本
echo "启动并行任务..."
sudo ~/venv/myenv/bin/python ~/projects/CI/DS10.py &#看情况选择你想选择的语音助手模式
~/venv/myenv/bin/python ~/projects/CI/ci2.py &

# 等待所有后台任务完成
wait
echo "所有并行任务执行完毕"


# 赋予脚本执行权限：
# chmod +x run_scripts.sh
# 运行脚本：
# ./run-ci.sh
