#!/bin/bash

# 并行运行两个Python脚本
echo "启动并行任务..."
~/venv/myenv/bin/python ~/projects/CI/ci2.py 
echo "所有并行任务执行完毕"

# 赋予脚本执行权限：
# chmod +x run_scripts.sh
# 运行脚本：
# ./run.sh
