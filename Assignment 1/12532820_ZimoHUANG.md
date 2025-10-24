

# **阅读在线版本：**[12535820_黄子墨](https://iyrna6v2lz.feishu.cn/wiki/Ik1cwsOngiqkNukHyolcIdC0nyb)





# **Q1: BPE 算法解释与代码演示**

1. ## **BPE 算法解释**

   

字节对编码（Byte Pair Encoding, BPE）是一种常见的子词（subword）分词算法。它的核心思想是通过迭代地合并语料库中出现频率最高的相邻字节对（或字符对），来动态地创建词汇表。这种方法能够有效地平衡词汇表大小和编码序列长度，并很好地处理未登录词（Out-of-Vocabulary, OOV）问题。

其主要步骤如下：

1. **初始化词汇表**：将初始词汇表设置为语料库中所有出现过的单个字符。
2. **统计词频**：统计语料库中每个单词的出现频率，并将单词拆分为字符序列。
3. **迭代合并**：
   1. 统计所有相邻字符对的出现频率，找到频率最高的一对（例如 "e" 和 "s"）。
   2. 将这对字符合并成一个新的子词单元（"es"）。
   3. 将这个新的子词单元添加到词汇表中。
4. **重复**：重复步骤3，直到词汇表大小达到预设的阈值，或没有可合并的字符对为止。
   1. 

1. ##  **BPE 工作原理代码演示**

```Python
import re
import collections

def get_stats(vocab):
    """
    功能：统计语料库中所有相邻字符对的出现频率。
    输入：一个字典，键是拆分成字符的单词，值是该单词的频率。
    返回：一个字典，键是相邻的字符对（元组），值是这对组合出现的总频率。
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    功能：在语料库中，将指定的字符对合并成一个新的单元。
    输入：
        - pair: 需要合并的字符对，例如 ('e', 's')。
        - v_in: 当前的语料库字典。
    返回：一个新的语料库字典，其中最高频的字符对已被合并。
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


corpus = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

num_merges = 5

print(f"原始语料: {corpus}\n")


for i in range(num_merges):
    # 统计当前语料中所有相邻字符对的频率
    pairs = get_stats(corpus)
    
    if not pairs:
        break
    
    # 找到出现频率最高的那一对
    best_pair = max(pairs, key=pairs.get)
    
    # 合并操作
    corpus = merge_vocab(best_pair, corpus)  
```

1. ## **训练并应用 BPE 分词器**

使用 subword-nmt 工具包训练一个词汇表大小为 10,000 的 BPE (Byte Pair Encoding) 模型，并将其应用于中英文混合对话数据的分词任务。

- **操作系统**: Linux 6.11.0-17-generic
- **硬件配置**: 24 核 CPU, NVIDIA RTX 3090 (24GB 显存)
- **软件环境**: Python 3, subword-nmt 工具包
- **数据集**: 科技研究领域对话数据 (130,501 条记录)
  - 

### 文件结构

```Plain
/home/limx/Desktop/Advanced-NLP-Assignment/
├── extract_and_format_data.py          # 数据提取脚本
├── training_corpus.txt                 # 训练语料库 (40.7 MB)
├── bpe_codes.txt                       # BPE 规则文件 (101 KB)
├── tokenized_corpus.txt                # 分词结果文件
└── subword-nmt/                        # subword-nmt 工具包
    ├── learn_bpe.py                    # BPE 训练脚本
    └── apply_bpe.py                    # BPE 应用脚本
```

### 实验步骤

#### 步骤 1: 数据预处理

使用自定义的数据提取脚本从 JSONL 文件中提取对话数据并转换为训练语料库。

暂时无法在飞书文档外展示此内容

**执行命令**:

```Bash
python3 extract_and_format_data.py \
    --input "industry_instruction_semantic_cluster_dedup_科技_科学研究_valid_train (1).jsonl" \
    --output training_corpus.txt \
    --sample-size 50000 \
    --format training
```

**处理结果**:

- 成功提取 50,000 个对话，生成ChatML格式的数据集
- 生成训练语料库包含 141,244 个文本片段
- 文件大小: 40.7 MB
  - 

**数据示例**:

```Plain
在5G网络开发过程中，有哪些常见的安全挑战需要解决？
在5G网络开发过程中，有几个常见的安全挑战需要解决。首先是网络架构的安全性，包括如何保护核心网络、边缘网络和终端设备的安全。其次是用户数据的隐私保护，如何确保用户的个人信息和通信内容在传输和存储过程中的安全性。另外，物联网设备的安全性也是一个重要的挑战，因为5G网络将连接大量的物联网设备，如何保护这些设备免受攻击和滥用是一个关键问题。此外，虚拟化和云计算技术的应用也带来了新的安全挑战，如如何保护虚拟化环境的安全和云服务提供商的安全。总的来说，在5G网络开发过程中，需要解决这些常见的安全挑战，以确保网络的安全和可靠性。
How does the clustering of microearthquakes help seismologists understand what initiates and governs an earthquake, and what insights do researchers like Ross Stein hope to gain from this?
The clustering of microearthquakes helps seismologists understand what initiates and governs an earthquake by providing insights into the stress transfer and fault interaction processes. By analyzing the locations and patterns of microearthquakes, researchers like Ross Stein hope to gain a better understanding of how earthquakes are triggered and how stress is transferred along faults. This knowledge can help improve earthquake forecasting and hazard assessment. Stein's research focuses on the interaction between earthquakes and faults, and he hopes to learn more about the underlying mechanisms that control earthquake behavior.
......
```

#### 步骤 2: BPE 模型训练

使用 subword-nmt 的 learn_bpe.py 脚本训练 BPE 模型。

**执行命令**:

```Bash
python3 subword-nmt/learn_bpe.py \
    --input training_corpus.txt \
    --output bpe_codes.txt \
    --symbols 10000 \
    --min-frequency 2 \
    --verbose
```

**训练参数**:

- 词汇表大小: 10,000
- 最小频率阈值: 2
- 训练数据: 141,244 行文本
  - 

**训练结果**:

- 成功生成 10,000 条 BPE 合并规则
- 训练时间: 约 1 分钟
  - 

**BPE 规则示例**:

```Plain
#version: 0.2
t h
i n
a n
t i
e n
th e</w>
e r
o n
r e
...
```

#### 步骤 3: 分词应用

使用训练好的 BPE 模型对训练数据进行分词。

**执行命令**:

```Bash
python3 subword-nmt/apply_bpe.py \
    --input training_corpus.txt \
    --codes bpe_codes.txt \
    --output tokenized_corpus.txt
```

**分词结果**:

- 成功对 141,245 行文本进行分词
- 输出文件包含子词标记的文本

**示例1**

```Plain
在5G网络开发过程中，有哪些常见的安全挑战需要解决？
```

分词后：

```Plain
在@@ 5G@@ 网络@@ 开发@@ 过程@@ 中，@@ 有@@ 哪些@@ 常@@ 见@@ 的@@ 安全@@ 挑战@@ 需要@@ 解决@@ ？
```

**示例2**

```Plain
How does the clustering of microearthquakes help seismologists understand what initiates and governs an earthquake?
```

分词后：

```Plain
How does the c@@ lu@@ ster@@ ing of micro@@ ear@@ th@@ qu@@ a@@ k@@ es help se@@ is@@ mo@@ log@@ i@@ sts under@@ st@@ and what initi@@ ates and govern@@ s an ear@@ th@@ qu@@ a@@ ke@@ ?
```

# Q2: Transformer 模型实现

本实验完成了 `transformer_model.py` 文件中缺失的 5 个核心函数实现，构建了一个完整的 Transformer 模型。该模型包含了位置编码、多头注意力机制、前馈网络、残差连接和编码器层等关键组件。

## 实现的核心组件

1. ### **PositionalEncoding (位置编码)**

**功能**: 为输入序列添加位置信息，解决 Transformer 无法感知序列位置的问题。

```Python
def forward(self, x):
    # Add positional encoding to input x, then apply dropout
    x = x + self.pe[:, :x.size(1)]  # Add positional encoding
    return self.dropout(x)
```

1. ### **FeedForward (前馈网络)**

**功能**: 实现 Transformer 中的两层全连接网络，提供非线性变换能力。

```Python
def forward(self, x):
    # Implement two linear layers with ReLU and Dropout
    x = self.fc1(x)  # First linear layer
    x = self.relu(x)  # ReLU activation
    x = self.drop(x)  # Dropout
    x = self.fc2(x)  # Second linear layer
    return x
```

1. ### **MultiHeadAttention (多头注意力)**

**功能**: 实现多头自注意力机制，允许模型同时关注不同位置的信息。

```Python
# Compute attention scores
# 1. Multiply Q and K^T, then scale by sqrt(d_k)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

# 2. If mask is provided, use masked_fill to set ignored positions to -1e9
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

# 3. Apply softmax to get attention weights (sum = 1)
attn = torch.softmax(scores, dim=-1)
attn = self.drop(attn)
```

1. ###  **Residual** **(残差连接)**

**功能**: 实现残差连接和层归一化，解决深层网络的梯度消失问题。

```Python
def forward(self, x, sublayer):
    # Apply LayerNorm on sublayer(x), then add residual connection with input x
    return x + self.drop(sublayer(self.norm(x)))
```

1. ### **EncoderLayer (****编码器****层)**

**功能**: 组合自注意力和前馈网络，构成一个完整的编码器层。

```Python
def forward(self, x, mask):
    # Apply self-attention + residual, then feed-forward + residual
    x = self.res_layers[0](x, lambda x: self.self_attn(x, x, x, mask))
    x = self.res_layers[1](x, self.ffn)
    return x
```

# **Q3: Bind Network**

![img](https://iyrna6v2lz.feishu.cn/space/api/box/stream/download/asynccode/?code=NjY3YzQxYjllOGZhMGZkMjk3NjFlYTc4MTQ5MjUwOTNfRmJxTHBrRHpMWjB6d2xkR3B2SFMwV3J1Q1h0RGdzemZfVG9rZW46VlZHeGJKZHNzb3pQa214ZldmMWNPdFRjbmFlXzE3NjEzMDA2MDU6MTc2MTMwNDIwNV9WNA)

暂时无法在飞书文档外展示此内容

## 核心功能：

- **Bind Network 类** - 完整的 PyTorch `nn.Module` 实现
- **网络架构** - 严格按照论文图3的结构实现
- **三个重复块** - 每个块包含 Norm、W1、W2、W3 和残差连接
- **门控机制** - W1+SiLU 和 W2 的并行路径设计
- **RMSNorm** - 高效的归一化层实现

```Plain
✓ Bind Network 测试通过!
输入形状: torch.Size([2, 1, 1024])
输出形状: torch.Size([2, 1, 4096])
总参数数量: 608,301,056
```