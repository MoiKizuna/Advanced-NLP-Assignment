#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced NLP Assignment 1 - 克隆工具
用于快速设置和运行Transformer模型项目
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class AssignmentCloner:
    def __init__(self):
        self.project_name = "Advanced-NLP-Assignment"
        self.github_url = "https://github.com/MoiKizuna/Advanced-NLP-Assignment.git"
        
    def clone_repository(self, target_dir=None):
        """克隆GitHub仓库到本地"""
        if target_dir is None:
            target_dir = os.getcwd()
            
        target_path = Path(target_dir) / self.project_name
        
        if target_path.exists():
            print(f"目录 {target_path} 已存在，跳过克隆")
            return target_path
            
        try:
            print(f"正在克隆仓库到 {target_path}...")
            subprocess.run([
                "git", "clone", self.github_url, str(target_path)
            ], check=True)
            print("✅ 仓库克隆成功！")
            return target_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 克隆失败: {e}")
            return None
    
    def setup_environment(self, project_path):
        """设置Python环境和依赖"""
        project_path = Path(project_path)
        
        # 创建虚拟环境
        venv_path = project_path / "venv"
        if not venv_path.exists():
            print("创建Python虚拟环境...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        
        # 激活虚拟环境并安装依赖
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            activate_script = venv_path / "bin" / "activate"
            pip_path = venv_path / "bin" / "pip"
        
        # 安装依赖
        requirements = [
            "torch>=1.9.0",
            "numpy>=1.21.0",
            "matplotlib>=3.3.0",
            "tqdm>=4.62.0",
            "jupyter>=1.0.0"
        ]
        
        print("安装Python依赖...")
        for req in requirements:
            try:
                subprocess.run([str(pip_path), "install", req], check=True)
                print(f"✅ 已安装 {req}")
            except subprocess.CalledProcessError as e:
                print(f"❌ 安装 {req} 失败: {e}")
    
    def create_run_script(self, project_path):
        """创建运行脚本"""
        project_path = Path(project_path)
        
        # 创建测试脚本
        test_script = project_path / "test_transformer.py"
        test_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer模型测试脚本
"""

import torch
import torch.nn as nn
from Assignment_1.Q2_transformer_model import build_transformer

def test_transformer():
    """测试Transformer模型的基本功能"""
    print("🚀 开始测试Transformer模型...")
    
    # 模型参数
    src_vocab = 1000
    tgt_vocab = 1000
    src_len = 50
    tgt_len = 50
    embed_dim = 256
    N = 4
    heads = 8
    hidden_dim = 512
    dropout = 0.1
    
    # 构建模型
    print("构建Transformer模型...")
    model = build_transformer(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_len=src_len,
        tgt_len=tgt_len,
        embed_dim=embed_dim,
        N=N,
        heads=heads,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    print(f"✅ 模型构建成功！参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 2
    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))
    
    # 创建掩码
    src_mask = torch.ones(batch_size, src_len, src_len)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    
    print("测试前向传播...")
    try:
        # 编码
        enc_out = model.encode(src, src_mask)
        print(f"✅ 编码输出形状: {enc_out.shape}")
        
        # 解码
        dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
        print(f"✅ 解码输出形状: {dec_out.shape}")
        
        # 投影
        output = model.project(dec_out)
        print(f"✅ 最终输出形状: {output.shape}")
        
        print("🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查模型实现中的TODO部分")

if __name__ == "__main__":
    test_transformer()
'''
        
        with open(test_script, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 创建README
        readme_path = project_path / "README.md"
        readme_content = '''# Advanced NLP Assignment 1

## 项目结构

```
Advanced-NLP-Assignment/
├── Assignment 1/
│   ├── assignment_1.pdf          # 作业要求文档
│   ├── Q1/
│   │   └── Q1_Answers.md         # Q1答案文档
│   └── Q2_transformer_model.py  # Transformer模型实现
├── PPT/                          # 课程PPT
└── README.md                     # 项目说明
```

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/MoiKizuna/Advanced-NLP-Assignment.git
cd Advanced-NLP-Assignment
```

### 2. 设置环境
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\\Scripts\\activate  # Windows

pip install torch numpy matplotlib tqdm jupyter
```

### 3. 运行测试
```bash
python test_transformer.py
```

## 作业内容

### Q1: Transformer理论问题
查看 `Assignment 1/Q1/Q1_Answers.md` 了解详细答案。

### Q2: Transformer模型实现
在 `Assignment 1/Q2_transformer_model.py` 中完成TODO部分：
- PositionalEncoding.forward()
- FeedForward.forward()
- MultiHeadAttention.forward()
- Residual.forward()
- EncoderLayer.forward()

## 学习资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - PyTorch官方教程

## 贡献

欢迎提交Issue和Pull Request！
'''
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ 运行脚本和README创建完成！")
    
    def run(self, target_dir=None):
        """主运行函数"""
        print("🎯 Advanced NLP Assignment 1 - 克隆工具")
        print("=" * 50)
        
        # 克隆仓库
        project_path = self.clone_repository(target_dir)
        if project_path is None:
            return
        
        # 设置环境
        self.setup_environment(project_path)
        
        # 创建运行脚本
        self.create_run_script(project_path)
        
        print("\n🎉 项目设置完成！")
        print(f"项目路径: {project_path}")
        print("\n下一步:")
        print("1. cd " + str(project_path))
        print("2. source venv/bin/activate  # 激活虚拟环境")
        print("3. python test_transformer.py  # 运行测试")

def main():
    parser = argparse.ArgumentParser(description="Advanced NLP Assignment 1 克隆工具")
    parser.add_argument("--target-dir", "-t", help="目标目录路径", default=None)
    
    args = parser.parse_args()
    
    cloner = AssignmentCloner()
    cloner.run(args.target_dir)

if __name__ == "__main__":
    main()
