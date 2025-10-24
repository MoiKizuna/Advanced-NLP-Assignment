#!/bin/bash
# Advanced NLP Assignment 1 - 快速克隆脚本

echo "🎯 Advanced NLP Assignment 1 - 快速设置"
echo "========================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo "❌ 错误: 未找到Git，请先安装Git"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 项目根目录: $PROJECT_ROOT"

# 运行Python克隆工具
echo "🚀 启动克隆工具..."
python3 "$SCRIPT_DIR/clone_tool.py" --target-dir "$(dirname "$PROJECT_ROOT")"

echo ""
echo "✅ 设置完成！"
echo ""
echo "📖 查看作业答案:"
echo "   cat '$PROJECT_ROOT/Q1/Q1_Answers.md'"
echo ""
echo "🧪 运行测试:"
echo "   cd '$PROJECT_ROOT'"
echo "   python3 test_transformer.py"
echo ""
echo "📚 开始学习:"
echo "   1. 阅读 Q1_Answers.md 了解理论"
echo "   2. 完成 Q2_transformer_model.py 中的TODO"
echo "   3. 运行测试验证实现"
