#!/bin/bash
# 快速测试 Teacher Trajectory Pipeline

cd /home/qisheng/agent/AgentEvolver
source ~/anaconda3/bin/activate agentevolver

echo "Running quick test..."
python tests/quick_test_teacher.py

echo ""
echo "Test completed."

