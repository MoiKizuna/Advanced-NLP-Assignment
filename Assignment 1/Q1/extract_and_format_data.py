import json
import argparse
import os
from typing import List, Dict, Any

def extract_conversations_from_jsonl(input_file: str, output_file: str, max_samples: int = None):
    """
    从 JSONL 文件中提取对话数据并转换为 ChatML 格式
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出的文本文件路径
        max_samples: 最大处理样本数，None 表示处理所有
    """
    
    chatml_conversations = []
    processed_count = 0
    
    print(f"开始处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and processed_count >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                
                # 检查是否有对话数据
                if 'conversations' not in data:
                    continue
                    
                conversations = data['conversations']
                if not conversations or len(conversations) < 2:
                    continue
                
                # 构建 ChatML 格式的对话
                chatml_text = ""
                for conv in conversations:
                    if conv['from'] == 'human':
                        chatml_text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
                    elif conv['from'] == 'gpt':
                        chatml_text += f"<|im_start|>assistant\n{conv['value']}<|im_end|>\n"
                
                chatml_conversations.append(chatml_text.strip())
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"已处理 {processed_count} 个对话")
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                continue
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in chatml_conversations:
            f.write(conv + '\n\n')
    
    print(f"处理完成！共提取 {processed_count} 个对话")
    print(f"输出文件: {output_file}")
    return processed_count

def create_training_corpus(input_file: str, output_file: str, sample_size: int = 50000):
    """
    创建用于 BPE 训练的语料库
    
    Args:
        input_file: JSONL 输入文件
        output_file: 输出训练文件
        sample_size: 采样大小
    """
    
    print(f"创建训练语料库，采样大小: {sample_size}")
    
    # 提取对话数据
    temp_file = "temp_chatml.txt"
    extract_conversations_from_jsonl(input_file, temp_file, sample_size)
    
    # 读取并处理文本，移除特殊标记，只保留纯文本
    training_texts = []
    
    with open(temp_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 按对话分割
        conversations = content.split('<|im_start|>')
        
        for conv in conversations:
            if not conv.strip():
                continue
                
            # 移除 ChatML 标记，只保留文本内容
            lines = conv.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('<|im_end|>') and not line.startswith('user') and not line.startswith('assistant'):
                    # 清理文本
                    clean_text = line.replace('<|im_end|>', '').strip()
                    if clean_text:
                        training_texts.append(clean_text)
    
    # 写入训练文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in training_texts:
            f.write(text + '\n')
    
    # 清理临时文件
    os.remove(temp_file)
    
    print(f"训练语料库创建完成: {output_file}")
    print(f"包含 {len(training_texts)} 个文本片段")

def main():
    parser = argparse.ArgumentParser(description='从 JSONL 文件提取对话数据')
    parser.add_argument('--input', '-i', required=True, help='输入的 JSONL 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出的文本文件路径')
    parser.add_argument('--sample-size', '-s', type=int, default=50000, 
                       help='采样大小 (默认: 50000)')
    parser.add_argument('--format', choices=['chatml', 'training'], default='training',
                       help='输出格式: chatml 或 training (默认: training)')
    
    args = parser.parse_args()
    
    if args.format == 'chatml':
        extract_conversations_from_jsonl(args.input, args.output, args.sample_size)
    else:
        create_training_corpus(args.input, args.output, args.sample_size)

if __name__ == '__main__':
    main()
