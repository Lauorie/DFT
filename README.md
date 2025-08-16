<p align="right">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a>
</p>

# Advanced Fine-Tuning Framework for LLMs with Differentiated Loss

![](loss.png)

## ✨ Features

*   **🚀 Differentiated Fine-Tuning (DFT) Loss**: An innovative custom loss function that dynamically adjusts loss weights based on the model's confidence (`p_correct`) in predicting the correct token. This allows the model to focus on consolidating "well-learned" knowledge while avoiding being misled by "overly difficult" samples, resulting in more stable and efficient convergence.
*   **⚡ High Performance and Efficiency**:
    *   **Distributed Training**: Deep integration with **DeepSpeed ZeRO-3**, enabling large-scale model training on multiple GPUs and optimizing memory usage.
    *   **Flash Attention 2**: Built-in support for `flash_attention_2` significantly boosts training speed and efficiency for long sequences (e.g., 8K+ tokens).
    *   **Gradient Checkpointing**: Effectively reduces memory consumption during training.
    *   **BF16/FP16 Mixed Precision**: Accelerates training while maintaining model performance.
*   **📦 Robust Data Processing**:
    *   **ChatML Format**: Designed specifically for handling dialogue data in the `ChatML` format.
    *   **Multi-Source Data Fusion**: Automatically loads, validates, and merges training data from multiple JSON files, handling schema inconsistencies.
    *   **Efficient Preprocessing**: Supports multi-process data preprocessing for faster data preparation.
*   **📝 Comprehensive Logging and Monitoring**:
    *   **Distributed Logging**: Custom distributed logger ensures clear, non-redundant logs in multi-GPU settings.
    *   **Training Metrics Monitoring**: Deep integration with **WandB** or **SwanLab** for real-time tracking of `loss`, `grad_norm`, and custom metrics like `avg_p_correct`.
*   **🔧 Flexible Configuration**: All training parameters (model, data, DFT parameters, training settings, etc.) are configured via command line for clarity and ease of use.

---

## 🧠 How DFT Loss Works

### 1. Standard SFT Loss and Gradient

#### 1.1 SFT Loss Function

The standard Supervised Fine-Tuning (SFT) loss is the token-level cross-entropy (expectation over expert data pairs D):

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[%20-\log%20\pi_\theta(y^*|x)%20\right]" alt="SFT Loss">
</p>

- <img src="https://latex.codecogs.com/svg.latex?x" style="vertical-align: middle;" alt="x">: Input (e.g., question, instruction)
- <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt="y^*">: Expert answer (ground truth label)
- <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt="πθ(y*|x)">: Probability of output <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt="y^*"> under model parameters <img src="https://latex.codecogs.com/svg.latex?\theta" style="vertical-align: middle;" alt="θ">

#### 1.2 SFT Gradient

Gradient with respect to <img src="https://latex.codecogs.com/svg.latex?\theta" style="vertical-align: middle;" alt="θ">:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\nabla_\theta\log%20\pi_\theta(y^*|x)\right]" alt="SFT Loss Gradient">
</p>

---

### 2. Standard RL Policy Gradient

RL aims to maximize expected reward:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20r(x,%20y)%20]" alt="RL Objective">
</p>

- <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)" style="vertical-align: middle;" alt="r(x, y)">: Reward function measuring the quality of <img src="https://latex.codecogs.com/svg.latex?(x,%20y)" style="vertical-align: middle;" alt="(x, y)">

The **policy gradient** is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\cdot%20r(x,%20y)%20]" alt="RL Policy Gradient">
</p>

---

### 3. Expressing SFT Gradient in RL Form with Importance Sampling

#### 3.1 Rewriting SFT Gradient

SFT's expectation is over <img src="https://latex.codecogs.com/svg.latex?(x,%20y^*)%20\sim%20\mathcal{D}" style="vertical-align: middle;" alt="">, while RL's is over <img src="https://latex.codecogs.com/svg.latex?(x,%20y)%20\sim%20\pi_\theta" style="vertical-align: middle;" alt="">. We aim to write SFT's gradient as "sampled from <img src="https://latex.codecogs.com/svg.latex?\pi_\theta" style="vertical-align: middle;" alt=""> with weighting".

**Key trick:**  
Rewrite using importance sampling:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{y^*%20\sim%20p^*}%20[f(y^*)]%20=%20\mathbb{E}_{y%20\sim%20\pi_\theta}%20\left[%20\frac{p^*(y)}{\pi_\theta(y)}%20f(y)%20\right]" alt="IS Trick">
</p>

In SFT, <img src="https://latex.codecogs.com/svg.latex?p^*(y)" style="vertical-align: middle;" alt=""> is the "expert distribution". For the discrete dataset D:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20[%20-\nabla_\theta%20\log%20\pi_\theta(y^*|x)%20]%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x}%20\mathbb{E}_{y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\cdot%20(%20-\nabla_\theta%20\log%20\pi_\theta(y|x)%20)%20\right]" alt="SFT IS expansion">
</p>

- <img src="https://latex.codecogs.com/svg.latex?1[y%20=%20y^*]" style="vertical-align: middle;" alt="1[y = y^*]">: Indicator function, 1 if sampled <img src="https://latex.codecogs.com/svg.latex?y" style="vertical-align: middle;" alt=""> equals <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt="">, else 0.

**Thus:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\cdot%20(%20-\nabla_\theta%20\log%20\pi_\theta(y|x)%20)%20\right]" alt="SFT IS RL Form">
</p>

---

#### 3.2 Rewriting in RL Policy Gradient Structure

Define:

- **Implicit reward**: <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)%20=%201[y%20=%20y^*]" style="vertical-align: middle;" alt="">
- **Importance weight**: <img src="https://latex.codecogs.com/svg.latex?w(y|x)%20=%20\frac{1}{\pi_\theta(y|x)}" style="vertical-align: middle;" alt="">

So, the above is equivalent to:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20-%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20w(y|x)%20\cdot%20r(x,%20y)%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\right]" alt="SFT as RL Policy Gradient">
</p>

This rewrites the SFT gradient in the **RL policy gradient form**, differing only in the definitions of <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)" style="vertical-align: middle;" alt=""> and <img src="https://latex.codecogs.com/svg.latex?w(y|x)" style="vertical-align: middle;" alt="">.

---

### 4. Analysis: The SFT “Implicit Reward Problem”

Note:

- <img src="https://latex.codecogs.com/svg.latex?w(y|x)%20=%20\frac{1}{\pi_\theta(y|x)}" style="vertical-align: middle;" alt="">
- <img src="https://latex.codecogs.com/svg.latex?r(x,%20y)%20=%201[y%20=%20y^*]" style="vertical-align: middle;" alt="">

**Only when the model generates the expert answer <img src="https://latex.codecogs.com/svg.latex?y^*" style="vertical-align: middle;" alt=""> does it get reward 1; otherwise 0. However, this reward is amplified by <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt="">.**

- If <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> is small, then <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> is large, leading to gradient explosion, unstable optimization, and poor generalization.

---

### 5. DFT Correction: Eliminating <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt=""> Amplification

**Core idea:**  
Since <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> causes instability, directly multiply by <img src="https://latex.codecogs.com/svg.latex?\pi_\theta(y^*|x)" style="vertical-align: middle;" alt=""> in the loss/gradient to "cancel" its effect.

#### 5.1 Corrected Gradient (DFT Gradient)

Let:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{DFT}}(\theta)%20=%20\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20\cdot%20\text{sg}(\pi_\theta(y^*|x))" alt="DFT Gradient">
</p>

- <img src="https://latex.codecogs.com/svg.latex?\text{sg}(\cdot)" style="vertical-align: middle;" alt="sg(.)">: stop-gradient operator (does not backpropagate), used as a numeric weight.

Expanded:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y^*|x)\right]" alt="Expanded DFT Gradient">
</p>

#### 5.2 DFT Loss Function

Since

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20\Big(%20-%20\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\log%20\pi_\theta(y^*|x)%20\Big)%20=%20-%20\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\nabla_\theta%20\log%20\pi_\theta(y^*|x)" alt="DFT Loss Derivation">
</p>

So, the DFT loss is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\text{sg}(\pi_\theta(y^*|x))%20\cdot%20\log%20\pi_\theta(y^*|x)\right]" alt="DFT Loss">
</p>

#### 5.3 Token-Level DFT Loss

For NLP, DFT is extended to token-level as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-%20\sum_{t=1}^{|y^*|}%20\text{sg}(\pi_\theta(y^*_t%20|%20y^*_{<t},%20x))%20\cdot%20\log%20\pi_\theta(y^*_t%20|%20y^*_{<t},%20x)\right]" alt="Token-level DFT Loss">
</p>

- <img src="https://latex.codecogs.com/svg.latex?y^*_t" style="vertical-align: middle;" alt="y^*_t">: The t-th token of the answer
- <img src="https://latex.codecogs.com/svg.latex?y^*_{<t}" style="vertical-align: middle;" alt="y^*_{<t}">: The first t-1 tokens of the answer

---

### 6. Recap of the Derivation Process

1. **SFT cross-entropy loss and gradient**
2. **Rewrite SFT gradient onto model policy distribution using importance sampling**
3. **Recognize SFT is equivalent to an RL with sparse reward, amplified by <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt="">**
4. **Analyze instability/generalization issues caused by <img src="https://latex.codecogs.com/svg.latex?1/\pi_\theta" style="vertical-align: middle;" alt="">**
5. **Propose DFT: multiply by <img src="https://latex.codecogs.com/svg.latex?\pi_\theta" style="vertical-align: middle;" alt=""> to neutralize amplification and correct the loss**

---

## Summary of Key Formulas

### SFT Loss and Gradient

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[%20-\log%20\pi_\theta(y^*|x)%20\right]" alt="SFT Loss">
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-\nabla_\theta\log%20\pi_\theta(y^*|x)\right]" alt="SFT Loss Gradient">
</p>

### RL Policy Gradient

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20J(\theta)%20=%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20[%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\cdot%20r(x,%20y)%20]" alt="RL Policy Gradient">
</p>

### SFT Gradient in RL Form via Importance Sampling

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\nabla_\theta%20L_{\mathrm{SFT}}(\theta)%20=%20-%20\mathbb{E}_{x%20\sim%20\mathcal{D}_x,\,%20y%20\sim%20\pi_\theta(\cdot|x)}%20\left[%20\frac{1[y%20=%20y^*]}{\pi_\theta(y|x)}%20\nabla_\theta%20\log%20\pi_\theta(y|x)%20\right]" alt="SFT RL Form">
</p>

### DFT Loss (Token-Level, Eq. 9 in the Paper)

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?L_{\mathrm{DFT}}(\theta)%20=%20\mathbb{E}_{(x,%20y^*)%20\sim%20\mathcal{D}}%20\left[-%20\sum_{t=1}^{|y^*|}%20\text{sg}(\pi_\theta(y^*_t%20|%20y^*_{<t},%20x))%20\cdot%20\log%20\pi_\theta(y^*_t%20|%20y^*_{<t},%20x)\right]" alt="DFT Token Loss">
</p>

> Where <img src="https://latex.codecogs.com/svg.latex?\text{sg}(\cdot)" style="vertical-align: middle;" alt="sg(.)"> indicates no gradient flow through the weight.

---

The core idea of DFT Loss: **Let the model focus more on what it is confident about learning correctly**.

Traditional cross-entropy treats all tokens equally. DFT, via a `dft_alpha` parameter, adjusts this behavior:

```python
# 1. Compute model's probability for the correct token, p_correct
with torch.no_grad():
    probs = F.softmax(shift_logits_flat, dim=-1)
    p_correct = probs.gather(1, correct_labels.unsqueeze(-1)).squeeze(-1)

# 2. Compute the loss weight using p_correct and dft_alpha
# If p_correct -> 1 (very confident), dft_weight -> 1
# If p_correct -> 0 (not confident), dft_weight -> (1 - dft_alpha)
# High p_correct ⇒ high weight ⇒ reinforce "easy/already learned" tokens (like self-paced, but prefers easy samples)
# Low p_correct ⇒ low weight ⇒ downweight hard samples' gradients
dft_weight = p_correct * self.dft_alpha + (1 - self.dft_alpha)

# 3. Apply weight to the original loss
loss_flat = original_loss_flat * dft_weight

# 4. Compute the final mean loss
loss = loss_flat.sum() / num_valid_tokens
```

In this way, the less confident (hard) samples contribute less to the loss, preventing them from dominating the gradients and disrupting convergence.

---

## ⚙️ Environment Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  (Recommended) Create a conda virtual environment:
    ```bash
    conda create -n dft_trainer python=3.10
    conda activate dft_trainer
    ```

3.  Install dependencies. Make sure your environment has PyTorch matching your CUDA version.
    ```bash
    # requirements.txt

    # Core dependencies
    torch --pre "torch>=2.1.0"
    transformers "transformers>=4.40.0"
    datasets "datasets>=2.18.0"
    deepspeed "deepspeed>=0.14.0"

    # Acceleration and efficiency
    accelerate "accelerate>=0.29.0"
    flash-attn --pre "flash-attn>=2.5.0" --no-build-isolation

    # Utilities
    sentencepiece # for tokenization
    protobuf # for tokenization

    # Experiment tracking (optional)
    swanlab
    wandb
    ```
    To install:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📚 Data Preparation

This project uses standard `JSONL` files, where each line is a JSON object. Each object must have a `messages` field in **ChatML** format.

**Data Example (`data.jsonl`):**
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, please introduce yourself."}, {"role": "assistant", "content": "Hello! I am a large language model, happy to assist you."}]}
{"messages": [{"role": "user", "content": "Write me a poem about spring."}, {"role": "assistant", "content": "Sure. Spring breeze caresses sprouting green, soft rain moistens all unseen. Fields are fragrant, bees and butterflies dance, the land a painting, inviting all to glance."}]}
```
**Key Points:**
*   `messages` is a list of dialogue turns.
*   Each turn is a dict with `role` and `content`.
*   There **must be at least one `assistant` turn** in `messages` to compute the loss (only `assistant` replies are included in loss computation).

---

## 🚀 Getting Started with Training

Training is managed via a launch script that configures all necessary parameters.

### 1. DeepSpeed Config

Prepare a DeepSpeed config file, e.g. `ds_config/zero3.json` for **ZeRO-3**:

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

### 2. Launch Script

Create a `train.sh` script to launch training:

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# List of training data files
DATA_FILES=(
    "/path/to/your/data_part1.json"
    "/path/to/your/data_part2.json"
    # ... more data files
)

# Output and log directories
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

### 3. Start Training

Make the script executable and run:

```bash
chmod +x train.sh
nohup ./train.sh > ${LOG_FILE} 2>&1 &
```

To monitor logs in real-time:
```bash
tail -f ${LOG_FILE}
```

---

## 📈 Monitoring & Results

*   **Console Logs**: Training progress and loss are output to your specified log file.
*   **Experiment Tracking**: If `report_to` is set to `swanlab` or `wandb`, you can view all metrics and charts in the platform UI, including:
    *   `train/loss`: Training loss (should decrease steadily).
    *   `eval/loss`: Validation loss (key for generalization).
    *   `train/grad_norm`: Gradient norm (for training stability).
    *   `train/train/avg_p_correct`: **DFT core metric**; average model confidence on correct tokens (should increase steadily).
    *   `train/train/dft_alpha`: The hyperparameter you set (for verifying configuration).

---

## 🤝 Contribution

Contributions are welcome! If you have ideas, suggestions, or find bugs, feel free to submit a Pull Request or open an Issue.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
