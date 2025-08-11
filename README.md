# Advanced Fine-Tuning Framework for LLMs with Differentiated Loss

![](loss.png)

## ✨ 项目特色

*   **🚀 差异化微调 (DFT) Loss**: 创新的自定义损失函数，它根据模型对正确Token的预测置信度（`p_correct`）动态调整损失权重。这使得模型能够更专注于巩固“已学好”的知识，同时避免被“过难”的样本带偏，从而实现更稳定、高效的收敛。
*   **⚡ 高性能与高效率**:
    *   **分布式训练**: 深度集成 **DeepSpeed ZeRO-3**，支持在多GPU上进行大规模模型训练，极大优化了显存占用。
    *   **Flash Attention 2**: 内置支持 `flash_attention_2`，显著提升长序列（如8K+）训练的速度和效率。
    *   **梯度检查点 (Gradient Checkpointing)**: 有效减少训练过程中的显存消耗。
    *   **BF16/FP16 混合精度**: 加速训练过程，同时保持模型性能。
*   **📦 健壮的数据处理**:
    *   **ChatML 格式**: 专为处理 `ChatML` 格式的对话数据而设计。
    *   **多源数据融合**: 能够自动加载、验证和合并来自多个不同JSON文件的训练数据，并处理schema不一致的问题。
    *   **高效预处理**: 支持多进程数据预处理，加快数据准备速度。
*   **📝 全面的日志与监控**:
    *   **分布式日志**: 内置自定义的分布式日志记录器，确保在多卡环境中日志清晰、不冗余。
    *   **训练指标监控**: 深度集成 **WandB** 或 **SwanLab** 等实验跟踪工具，实时监控`loss`、`grad_norm`以及自定义的 `avg_p_correct` 等关键指标。
*   **🔧 灵活配置**: 所有训练参数（模型、数据、DFT参数、训练参数等）均通过命令行进行配置，清晰易用。

## 🧠 DFT Loss 工作原理

## 1. SFT 的标准公式与梯度

### 1.1 SFT损失函数

标准的 SFT 损失为 token-level 交叉熵损失（以一个“专家数据对”分布 D 为期望）：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[%20-\log%20\pi_\theta(y^*|x)%20\right]" alt="SFT Loss">
</p>

- <img src="https://latex.codecogs.com/svg.latex?x" style="vertical-align: middle;" alt="x">：输入（如问题、指令）
- <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt="y^*">：专家答案（ground-truth 标签）
- <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt="πθ(y*|x)">：模型参数 <img src="https://latex.codecogs.com/svg.latex?\theta" style="vertical-align: middle;" alt="θ"> 下，输出 <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt="y^*"> 的概率

### 1.2 SFT的梯度

对 <img src="https://latex.codecogs.com/svg.latex?\theta" style="vertical-align: middle;" alt="θ"> 求梯度：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\nabla_\theta\log%20\pi_\theta(y^*|x)\right]" alt="SFT Loss Gradient">
</p>

---

## 2. RL 策略梯度的标准形式

RL 的目标是最大化期望奖励：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20r(x,%20y)%20]" alt="RL Objective">
</p>

- <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)" style="vertical-align: middle;" alt="r(x, y)">：奖励函数，衡量 <img src="https://latex.codecogs.com/svg.latex?(x,%20y)" style="vertical-align: middle;" alt="(x, y)"> 的好坏

其**策略梯度**为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\cdot%20r(x,%20y)%20]" alt="RL Policy Gradient">
</p>

---

## 3. 用重要性采样把SFT的梯度写成RL形式

### 3.1 重新写SFT梯度

SFT的期望是在 <img src="https://latex.codecogs.com/svg.latex?(x,%20y^*)%20\sim%20\mathcal{D}" style="vertical-align: middle;" alt=""> 上，但RL是在 <img src="https://latex.codecogs.com/svg.latex?(x,%20y)%20\sim%20\pi_\theta" style="vertical-align: middle;" alt=""> 上。我们希望把SFT的梯度也写成“采样于 <img src="https://latex.codecogs.com/svg.latex?\pi_\theta" style="vertical-align: middle;" alt=""> 并带权重”的形式。

**关键技巧：**  
用重要性采样重写：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{y^*%20\sim%20p^*}%20[f(y^*)]%20=%20\mathbb{E}_{y%20\sim%20\pi_\theta}%20\left[%20\frac{p^*(y)}{\pi_\theta(y)}%20f(y)%20\right]" alt="IS Trick">
</p>

在SFT里，<img src="https://latex.codecogs.com/svg.latex?p^*(y)" style="vertical-align: middle;" alt=""> 可以看作是“专家分布”，但数据集D是有限的离散点，所以我们可以写成：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20[%20-\nabla_\theta%20\log%20\pi_\theta(y^*|x)%20]%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x}%20\mathbb{E}_{y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\cdot%20(%20-\nabla_\theta%20\log%20\pi_\theta(y|x)%20)%20\right]" alt="SFT IS expansion">
</p>

- <img src="https://latex.codecogs.com/svg.latex?1[y%20=%20y^*]" style="vertical-align: middle;" alt="1[y = y^*]">：指示函数，采样的 <img src="https://latex.codecogs.com/svg.latex?y" style="vertical-align: middle;" alt=""> 等于 <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt=""> 时为1，否则为0

**即**：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\cdot%20(%20-\nabla_\theta%20\log%20\pi_\theta(y|x)%20)%20\right]" alt="SFT IS RL Form">
</p>

---

### 3.2 重新整理为RL策略梯度结构

我们可以对上式做如下整理：

定义：

- **隐式奖励**：<img src="https://latex.codecogs.com/svg.latex?r(x,%20y)%20=%201[y%20=%20y^*]" style="vertical-align: middle;" alt="">
- **重要性权重**：<img src="https://latex.codecogs.com/svg.latex?w(y|x)%20=%20\frac{1}{\pi_\theta(y|x)}" style="vertical-align: middle;" alt="">

则，上式等价于：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20-%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20w(y|x)%20\cdot%20r(x,%20y)%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\right]" alt="SFT as RL Policy Gradient">
</p>

这就是将SFT的梯度**转化为RL策略梯度形式**，唯一的区别在于 <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)" style="vertical-align: middle;" alt=""> 和 <img src="https://latex.codecogs.com/svg.latex?w(y|x)" style="vertical-align: middle;" alt=""> 的定义。

---

## 4. SFT的“隐式奖励问题”分析

注意到：

- <img src="https://latex.codecogs.com/svg.latex?w(y|x)%20=%20\frac{1}{\pi_\theta(y|x)}" style="vertical-align: middle;" alt="">
- <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)%20=%201[y%20=%20y^*]" style="vertical-align: middle;" alt="">

即：**只有生成了专家答案 <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt=""> 才有奖励1，否则为0，但这个奖励会被 <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> 放大。**

- 如果 <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> 很小（模型原本不认为 <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt=""> 是好答案），<img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> 就很大，导致梯度爆炸，优化不稳定，泛化变差。

---

## 5. DFT的修正：消除 <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt=""> 的影响

**核心思想：**  
既然 <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> 带来坏影响，直接在损失函数/梯度中乘以 <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt="">，即可将其“抵消”。

### 5.1 修正后的梯度（DFT梯度）

令：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{DFT}}(\theta)%20=%20\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20\cdot%20\text{sg}(\pi_\theta(y^*|x))" alt="DFT Gradient">
</p>

- <img src="https://latex.codecogs.com/svg.latex?\text{sg}(\cdot)" style="vertical-align: middle;" alt="sg(.)">：stop-gradient 算子，表示这个项**不参与反向传播**，仅作为数值权重

具体写法（展开）：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y^*|x)\right]" alt="Expanded DFT Gradient">
</p>

### 5.2 反推DFT的损失函数

因为

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20\Big(%20-%20\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\log%20\pi_\theta(y^*|x)%20\Big)%20=%20-%20\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y^*|x)" alt="DFT Loss Derivation">
</p>

所以，DFT的损失是：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\log%20\pi_\theta(y^*|x)\right]" alt="DFT Loss">
</p>

### 5.3 Token-level DFT损失

NLP任务中答案通常是token序列，DFT在token级别推广为：

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-%20\sum_{t=1}^{|y^*|}%20\text{sg}(\pi_\theta(y^*_t%20|%20y^*_{<t},%20x))%20\cdot%20\log%20\pi_\theta(y^*_t%20|%20y^*_{<t},%20x)\right]" alt="Token-level DFT Loss">
</p>

- <img src="https://latex.codecogs.com/svg.latex?y^*_t" style="vertical-align: middle;" alt="y^*_t">：专家答案的第 <img src="https://latex.codecogs.com/svg.latex?t" style="vertical-align: middle;" alt="t"> 个token
- <img src="https://latex.codecogs.com/svg.latex?y^*_{<t}" style="vertical-align: middle;" alt="y^*_{<t}">：专家答案前 <img src="https://latex.codecogs.com/svg.latex?t-1" style="vertical-align: middle;" alt="t-1"> 个token

---

## 6. 推导总结流程回顾

1. **SFT的交叉熵损失与梯度**
2. **用重要性采样把SFT梯度重写到模型策略分布**
3. **发现SFT实际上等价于一个reward很稀疏、且被 <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt=""> 放大权重的RL**
4. **分析 <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt=""> 带来的训练不稳定/泛化问题**
5. **提出DFT：乘以 <img src="https://latex.codecogs.com/svg.latex?\pi_\theta" style="vertical-align: middle;" alt=""> 抵消这个放大，损失函数得到修正**

---

## 最终公式总结

### SFT损失与梯度

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[%20-\log%20\pi_\theta(y^*|x)%20\right]" alt="SFT Loss">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\nabla_\theta\log%20\pi_\theta(y^*|x)\right]" alt="SFT Loss Gradient">
</p>

### RL策略梯度

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\cdot%20r(x,%20y)%20]" alt="RL Policy Gradient">
</p>

### 用重要性采样重写SFT梯度

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20-%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\right]" alt="SFT RL Form">
</p>

### DFT损失（token-level，论文公式9）

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-%20\sum_{t=1}^{|y^*|}%20\text{sg}(\pi_\theta(y^*_t%20|%20y^*_{<t},%20x))%20\cdot%20\log%20\pi_\theta(y^*_t%20|%20y^*_{<t},%20x)\right]" alt="DFT Token Loss">
</p>

> 其中 <img src="https://latex.codecogs.com/svg.latex?\text{sg}(\cdot)" style="vertical-align: middle;" alt="sg(.)"> 表明权重处不参与梯度计算。



DFT Loss的核心思想是：**让模型更关注它有把握学对的东西**。

传统的交叉熵损失对所有Token一视同仁。而DFT通过一个`dft_alpha`参数来调整这一行为。其核心逻辑如下：

```python
# 1. 计算模型预测正确Token的概率 p_correct
with torch.no_grad():
    probs = F.softmax(shift_logits_flat, dim=-1)
    p_correct = probs.gather(1, correct_labels.unsqueeze(-1)).squeeze(-1)

# 2. 根据 p_correct 和 dft_alpha 计算损失权重
# 当 p_correct -> 1 (模型很自信), dft_weight -> 1
# 当 p_correct -> 0 (模型不自信), dft_weight -> (1 - dft_alpha)
dft_weight = p_correct * self.dft_alpha + (1 - self.dft_alpha)

# 3. 将权重应用到原始损失上
loss_flat = original_loss_flat * dft_weight

# 4. 计算最终平均损失
loss = loss_flat.sum() / num_valid_tokens
```

通过这种方式，模型预测越不准的“困难”样本，其损失权重就越低，从而避免了这些样本产生过大的梯度来干扰模型的整体收敛进程。

## ⚙️ 环境准备

1.  克隆本仓库：
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  建议使用 `conda` 创建一个虚拟环境：
    ```bash
    conda create -n dft_trainer python=3.10
    conda activate dft_trainer
    ```

3.  安装依赖。请确保您的环境中已安装与您的CUDA版本匹配的PyTorch。
    ```bash
    # requirements.txt
    
    # 核心依赖
    torch --pre "torch>=2.1.0"
    transformers "transformers>=4.40.0"
    datasets "datasets>=2.18.0"
    deepspeed "deepspeed>=0.14.0"
    
    # 加速与效率
    accelerate "accelerate>=0.29.0"
    flash-attn --pre "flash-attn>=2.5.0" --no-build-isolation
    
    # 工具
    sentencepiece # for tokenization
    protobuf # for tokenization
    
    # 实验跟踪 (可选)
    swanlab
    wandb
    ```
    执行安装：
    ```bash
    pip install -r requirements.txt
    ```

## 📚 数据准备

本项目使用标准的 `JSONL` 文件格式，每一行是一个JSON对象。每个JSON对象必须包含一个 `messages` 字段，其内容遵循 **ChatML** 格式。

**数据格式示例 (`data.jsonl`)**:
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "你好，请介绍一下自己。"}, {"role": "assistant", "content": "你好！我是一个大型语言模型，很高兴为您服务。"}]}
{"messages": [{"role": "user", "content": "帮我写一首关于春天的诗。"}, {"role": "assistant", "content": "当然。春风拂面绿芽新，细雨如丝润物频。田野芬芳蜂蝶舞，江山如画醉游人。"}]}
```
**关键点**:
*   `messages` 是一个列表，包含多个对话轮次。
*   每个对话轮次是一个字典，包含 `role` 和 `content`。
*   为了计算损失，`messages` 列表中**必须至少包含一个 `role` 为 `assistant` 的轮次**，因为只有 `assistant` 的回复才会被计入损失计算。

## 🚀 如何开始训练

训练通过一个启动脚本来管理，该脚本配置了所有必要的参数。

### 1. DeepSpeed 配置

项目需要一个DeepSpeed配置文件。这里提供一个适用于 **ZeRO-3** 的模板 `ds_config/zero3.json`。

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "wall_clock_breakdown": false,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none"
        },
        "offload_optimizer": {
            "device": "none"
        },
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_prefetch_bucket_size": 1e7,
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

### 2. 编写启动脚本

创建一个 `train.sh` 脚本来启动训练任务。

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 训练数据文件列表
DATA_FILES=(
    "/path/to/your/data_part1.json"
    "/path/to/your/data_part2.json"
    # ... more data files
)

# 输出和日志目录
OUTPUT_DIR="output_model"
LOG_DIR="logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_$(date +%F_%H%M%S).log"

deepspeed --num_gpus=8 train_dft_fixed.py \
    --model_name_or_path /path/to/your/base_model \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --trust_remote_code True \
    --data_files "${DATA_FILES[@]}" \
    --max_length 8192 \
    --preprocessing_num_workers 8 \
    --validation_split_percentage 2.0 \
    --enable_gradient_checkpointing True \
    --dft_alpha 0.7 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 16 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_only_model True \
    --report_to swanlab \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --deepspeed ./ds_config/zero3.json \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${LOG_DIR} \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False

echo "Training started in background. Log file: ${LOG_FILE}"
```

### 3. 运行训练

将脚本设置为可执行并运行：

```bash
chmod +x train.sh
nohup ./train.sh > ${LOG_FILE} 2>&1 &
```

您可以通过以下命令实时查看日志：
```bash
tail -f ${LOG_FILE}
```

## 📈 监控与结果

*   **命令行日志**: 训练进度、loss等信息会实时输出到您指定的日志文件中。
*   **实验跟踪平台**: 如果您配置了 `report_to` 为 `swanlab` 或 `wandb`，您可以访问对应的平台UI，查看所有指标的图表化展示，包括：
    *   `train/loss`: 训练集损失，应稳步下降。
    *   `eval/loss`: 验证集损失，是衡量模型泛化能力的关键。
    *   `train/grad_norm`: 梯度范数，用于判断训练稳定性。
    *   `train/train/avg_p_correct`: **DFT核心指标**，反映模型对正确Token的平均预测概率，应稳步上升。
    *   `train/train/dft_alpha`: 您设置的超参数，用于验证配置是否生效。

## 🤝 贡献

欢迎对本项目进行贡献！如果您有任何想法、建议或发现了bug，请随时提交 Pull Request 或创建 Issue。

## 📄 许可证

本项目采用 [Apache 2.0 License](LICENSE) 开源许可证。
