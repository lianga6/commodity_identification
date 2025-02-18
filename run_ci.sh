#!/bin/bash

# 并行运行两个Python脚本
echo "启动并行任务..."
sudo ~/venv/myenv/bin/python ~/projects/CI/DS.py &
~/venv/myenv/bin/python ~/projects/CI/ci2.py &

# 等待所有后台任务完成
wait
echo "所有并行任务执行完毕"

