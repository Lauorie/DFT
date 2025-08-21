import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import logging
import os
from pathlib import Path
from typing import List, Optional
import torch.distributed as dist
from contextlib import contextmanager
import traceback
import numpy as np
import shutil
import re

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
from datasets import load_dataset, Dataset, concatenate_datasets, disable_progress_bar, enable_progress_bar
from dataclasses import dataclass, field

# --- 分布式日志管理 ---
class DistributedLogger:
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.logger.setLevel(logging.DEBUG if self.is_main_process else logging.WARNING)
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            f'%(asctime)s - [RANK {self.rank}/{self.world_size}] - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_file and self.is_main_process:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str, force_all_ranks: bool = False):
        if self.is_main_process or force_all_ranks:
            self.logger.info(msg)
    
    def warning(self, msg: str, force_all_ranks: bool = False):
        if self.is_main_process or force_all_ranks:
            self.logger.warning(msg)
    
    def error(self, msg: str, force_all_ranks: bool = True):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        if self.is_main_process:
            self.logger.debug(msg)

dist_logger = DistributedLogger(__name__, "train_distributed.log")

@contextmanager
def progress_bar_context(show_progress: bool = True):
    if show_progress:
        enable_progress_bar()
    else:
        disable_progress_bar()
    try:
        yield
    finally:
        if not show_progress:
            enable_progress_bar()

# --- 参数定义 ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    torch_dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="flash_attention_2")
    trust_remote_code: bool = field(default=True)

@dataclass
class DataArguments:
    data_files: List[str] = field(metadata={"help": "Training data files"})
    eval_data_files: Optional[List[str]] = field(default=None)
    max_length: int = field(default=8192)
    preprocessing_num_workers: int = field(default=8)
    validation_split_percentage: float = field(default=5.0)
    max_eval_samples: Optional[int] = field(default=1000)
    debug_data_processing: bool = field(default=False)

@dataclass
class DFTArguments:
    enable_gradient_checkpointing: bool = field(default=True)
    dft_alpha: float = field(default=1.0)
    use_simple_dft: bool = field(default=True)

@dataclass 
class LoggingArguments:
    reduce_logging: bool = field(default=True)
    log_metrics_steps: int = field(default=100)

# --- DFT Trainer 完全修复版 ---
class DFTTrainer(Trainer):
    def __init__(self, dft_alpha: float = 1.0, log_metrics_steps: int = 100, use_simple_dft: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dft_alpha = dft_alpha
        self.log_metrics_steps = log_metrics_steps
        self.use_simple_dft = use_simple_dft
        self.is_main_process = self.args.local_rank <= 0
        # 统一保存策略：使用 transformers 内置 TrainingArguments.save_only_model
        self.save_only_model = bool(getattr(self.args, "save_only_model", False))
        # 手动最优指标与checkpoint跟踪（兼容 DeepSpeed + save_only_model 禁用 load_best_model_at_end 的场景）
        self.best_metric_name = getattr(self.args, "metric_for_best_model", "eval_loss")
        self.greater_is_better = getattr(self.args, "greater_is_better", False)
        self._tracked_best_metric = None
        self._tracked_best_checkpoint = None
        self._ckpt_paths_by_step = {}
        
        if not self.is_main_process:
            hf_logging.set_verbosity_error()

    def _save_checkpoint(self, model, trial, metrics=None):
        """如果设置为仅保存模型，则在每个 checkpoint 只保存模型与分词器。"""
        if not self.save_only_model:
            return super()._save_checkpoint(model, trial, metrics)
        
        # 仅保存模型权重
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        # 记录本步checkpoint目录
        try:
            self._ckpt_paths_by_step[int(self.state.global_step)] = output_dir
        except Exception:
            pass
        
        # 使用内部 _save，确保兼容 deepspeed/fp16 等场景
        self._save(output_dir)
        # 同步保存 TrainerState，供 _rotate_checkpoints 与最优模型选择使用
        try:
            if self.is_world_process_zero():
                # 兼容不同版本 transformers：直接写 trainer_state.json
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        except Exception:
            pass
        
        # 在保存时若拿到评估指标，手动更新最优
        try:
            if isinstance(metrics, dict):
                metric_value = None
                if self.best_metric_name in metrics:
                    metric_value = metrics[self.best_metric_name]
                else:
                    alt_key = self.best_metric_name if self.best_metric_name.startswith("eval_") else f"eval_{self.best_metric_name}"
                    if alt_key in metrics:
                        metric_value = metrics[alt_key]
                if metric_value is not None:
                    self._update_best_from_metric(int(self.state.global_step), float(metric_value))
        except Exception:
            pass
        
        # 保存 tokenizer
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass
        
        # 轮转旧 checkpoint（强制保留最佳checkpoint）
        try:
            self._rotate_checkpoints_keep_best()
        except Exception:
            pass

    def log(self, logs: dict, *args, **kwargs) -> None:
        """拦截日志，在出现 eval 指标时手动更新最优checkpoint。"""
        super().log(logs, *args, **kwargs)
        try:
            if not isinstance(logs, dict):
                return
            metric_value = None
            if self.best_metric_name in logs:
                metric_value = logs[self.best_metric_name]
            else:
                alt_key = self.best_metric_name if self.best_metric_name.startswith("eval_") else f"eval_{self.best_metric_name}"
                if alt_key in logs:
                    metric_value = logs[alt_key]
            if metric_value is not None:
                self._update_best_from_metric(int(self.state.global_step), float(metric_value))
        except Exception:
            pass

    def _update_best_from_metric(self, step: int, metric_value: float) -> None:
        is_better = (
            self._tracked_best_metric is None
            or (self.greater_is_better and metric_value > self._tracked_best_metric)
            or (not self.greater_is_better and metric_value < self._tracked_best_metric)
        )
        if not is_better:
            return
        self._tracked_best_metric = float(metric_value)
        best_path = self._ckpt_paths_by_step.get(step, os.path.join(self.args.output_dir, f"checkpoint-{step}"))
        self._tracked_best_checkpoint = best_path
        # 同步到 TrainerState
        self.state.best_metric = float(metric_value)
        self.state.best_model_checkpoint = best_path
        try:
            if os.path.isdir(best_path):
                with open(os.path.join(best_path, "BEST"), "w", encoding="utf-8") as f:
                    f.write(f"{self.best_metric_name}={self._tracked_best_metric}\n")
        except Exception:
            pass

    def _rotate_checkpoints_keep_best(self):
        """自定义轮转：遵循 save_total_limit，且永不删除最优 checkpoint。"""
        save_total_limit = getattr(self.args, "save_total_limit", None)
        if save_total_limit is None or save_total_limit <= 0:
            return

        output_dir = Path(self.args.output_dir)
        if not output_dir.exists():
            return

        ckpt_dirs = []
        pattern = re.compile(r"^checkpoint-(\d+)$")
        for item in output_dir.iterdir():
            if item.is_dir():
                m = pattern.match(item.name)
                if m:
                    step = int(m.group(1))
                    ckpt_dirs.append((step, item))

        if len(ckpt_dirs) <= save_total_limit:
            return

        # 按 step 升序（最老在前）
        ckpt_dirs.sort(key=lambda x: x[0])

        best_ckpt = getattr(self.state, "best_model_checkpoint", None)
        best_path = Path(best_ckpt).resolve() if best_ckpt else None

        # 需要保留：最优（若存在）+ 最新的若干个直到总数等于 save_total_limit
        keep_paths = set()
        if best_path is not None:
            keep_paths.add(best_path)
        remaining = save_total_limit - len(keep_paths)
        if remaining > 0:
            for _, p in ckpt_dirs[-remaining:]:
                keep_paths.add(p.resolve())

        for _, path in ckpt_dirs:
            if path.resolve() in keep_paths:
                continue
            try:
                shutil.rmtree(path)
                dist_logger.info(f"轮转删除checkpoint: {path}")
            except Exception as e:
                dist_logger.warning(f"删除checkpoint失败 {path}: {e}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """完全修复的compute_loss，确保维度匹配"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 获取vocab size
        vocab_size = logits.shape[-1]
        
        # Shift操作：确保logits和labels对齐
        # logits: [batch, seq_len, vocab] -> [batch, seq_len-1, vocab]
        # labels: [batch, seq_len] -> [batch, seq_len-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 展平用于loss计算
        # shift_logits: [batch * (seq_len-1), vocab]
        # shift_labels: [batch * (seq_len-1)]
        shift_logits_flat = shift_logits.view(-1, vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        
        # 计算基础loss
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_flat = loss_fct(shift_logits_flat, shift_labels_flat)
        
        if self.use_simple_dft and self.dft_alpha > 0:
            # DFT: 按预测概率加权
            with torch.no_grad():
                probs = F.softmax(shift_logits_flat, dim=-1)
                # 获取正确token的概率
                valid_mask = shift_labels_flat != -100
                gather_labels = shift_labels_flat.clone()
                gather_labels[~valid_mask] = 0  # 将padding位置设为0以避免gather错误
                
                p_correct = probs.gather(1, gather_labels.unsqueeze(-1)).squeeze(-1)
                p_correct = p_correct * valid_mask.float()  # mask掉padding
                
                # DFT权重
                dft_weight = p_correct * self.dft_alpha + (1 - self.dft_alpha)
                dft_weight = dft_weight * valid_mask.float()
            
            # 应用DFT权重
            loss_flat = loss_flat * dft_weight
        
        # 计算平均loss
        valid_tokens = (shift_labels_flat != -100).sum()
        if valid_tokens > 0:
            loss = loss_flat.sum() / valid_tokens
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 记录指标
        if model.training and self.state.global_step % self.log_metrics_steps == 0 and self.is_main_process:
            with torch.no_grad():
                if self.use_simple_dft and valid_tokens > 0 and 'p_correct' in locals():
                    avg_p_correct = p_correct[valid_mask].mean().item()
                    self.log({
                        "train/avg_p_correct": avg_p_correct,
                        "train/dft_alpha": self.dft_alpha,
                    })
        
        return (loss, outputs) if return_outputs else loss

# --- 数据处理类 ---
class DataProcessor:
    def __init__(self, tokenizer, max_length: int, debug: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        self.debug = debug
        self.process_stats = {"success": 0, "failed": 0, "total": 0}
    
    def _validate_messages_format(self, messages):
        if not isinstance(messages, list) or len(messages) == 0:
            return False
        
        has_assistant = False
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg.get('role') == 'assistant':
                has_assistant = True
        
        return has_assistant
    
    def load_and_validate_datasets(self, data_files: List[str], dataset_type: str = "train") -> Dataset:
        datasets = []
        total_samples = 0
        
        for file_path in data_files:
            if not Path(file_path).exists():
                dist_logger.warning(f"{dataset_type}数据文件不存在: {file_path}")
                continue
            
            try:
                dataset = load_dataset("json", data_files=file_path, split="train")
                
                if "messages" not in dataset.column_names:
                    dist_logger.warning(f"文件 {Path(file_path).name} 不包含messages字段")
                    continue
                
                def filter_valid(example):
                    return self._validate_messages_format(example.get("messages", []))
                
                initial_len = len(dataset)
                dataset = dataset.filter(filter_valid)
                final_len = len(dataset)
                
                if final_len == 0:
                    dist_logger.warning(f"文件 {Path(file_path).name} 没有有效数据")
                    continue
                
                if final_len < initial_len:
                    dist_logger.info(f"文件 {Path(file_path).name}: {initial_len} -> {final_len} 样本")
                
                # 仅保留训练所需列，避免不同文件其他字段（content/title/…）的schema冲突
                try:
                    dataset = dataset.select_columns(["messages"]) if "messages" in dataset.column_names else dataset
                except Exception:
                    # 某些datasets版本无select_columns则用remove_columns
                    cols_to_remove = [c for c in dataset.column_names if c != "messages"]
                    if cols_to_remove:
                        dataset = dataset.remove_columns(cols_to_remove)
                
                datasets.append(dataset)
                total_samples += final_len
                
            except Exception as e:
                dist_logger.error(f"加载文件失败 {file_path}: {e}")
                continue
        
        if not datasets:
            if dataset_type == "eval":
                return None
            raise ValueError(f"没有成功加载任何{dataset_type}数据文件")
        
        combined_dataset = concatenate_datasets(datasets)
        dist_logger.info(f"{dataset_type}数据集加载完成，总样本数: {total_samples}")
        
        return combined_dataset

    def preprocess_chatml_function(self, examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for messages in examples["messages"]:
            try:
                if not self._validate_messages_format(messages):
                    self.process_stats["failed"] += 1
                    continue
                
                processed = self._process_single_conversation(messages)
                if processed:
                    for key in model_inputs.keys():
                        model_inputs[key].append(processed[key])
                    self.process_stats["success"] += 1
                else:
                    self.process_stats["failed"] += 1
                    
            except Exception as e:
                self.process_stats["failed"] += 1
                if self.debug:
                    dist_logger.debug(f"处理失败: {str(e)[:100]}")
                continue
            
            self.process_stats["total"] += 1
        
        if self.is_main_process and self.process_stats["total"] % 100 == 0:
            dist_logger.info(
                f"Processing Data: total {self.process_stats['total']}, "
                f"Success: {self.process_stats['success']}, "
                f"Failed: {self.process_stats['failed']}"
            )
        
        return model_inputs

    def _process_single_conversation(self, messages):
        try:
            # 找assistant位置
            assistant_idx = -1
            for i, msg in enumerate(messages):
                if msg.get('role') == 'assistant':
                    assistant_idx = i
                    break
            
            if assistant_idx == -1:
                return None
            
            # 应用chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            if not text:
                return None
            
            # tokenize
            full_tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens.get("attention_mask", [1] * len(input_ids))
            
            # 处理labels
            labels = input_ids.copy()
            
            if assistant_idx > 0:
                prompt_messages = messages[:assistant_idx]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                prompt_tokens = self.tokenizer(
                    prompt_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                prompt_len = len(prompt_tokens["input_ids"])
                
                # Mask prompt部分
                for i in range(min(prompt_len, len(labels))):
                    labels[i] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as e:
            if self.debug:
                dist_logger.debug(f"处理对话错误: {str(e)[:200]}")
            return None

# --- 评估指标：token级别准确率 ---
def compute_token_level_accuracy(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # 统一转为批次列表以降低内存并兼容不拼接场景
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # labels 也可能是列表
    if isinstance(predictions, (list, tuple)):
        pred_batches = predictions
        label_batches = labels if isinstance(labels, (list, tuple)) else [labels] * len(pred_batches)
        total_correct = 0
        total_valid = 0
        for p, l in zip(pred_batches, label_batches):
            if isinstance(p, tuple):
                p = p[0]
            if p.ndim == 3:  # logits
                p = np.argmax(p, axis=-1)
            # 对齐
            if p.ndim == 2 and l.ndim == 2:
                p_shift = p[:, :-1]
                l_shift = l[:, 1:]
            else:
                p_shift = p
                l_shift = l
            mask = l_shift != -100
            total_valid += int(mask.sum())
            total_correct += int(((p_shift == l_shift) & mask).sum())
        acc = 0.0 if total_valid == 0 else total_correct / total_valid
        return {"acc": float(acc)}

    # 非列表：numpy数组
    if predictions.ndim == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions

    if pred_ids.ndim == 2 and labels.ndim == 2:
        pred_shift = pred_ids[:, :-1]
        labels_shift = labels[:, 1:]
    else:
        pred_shift = pred_ids
        labels_shift = labels

    mask = labels_shift != -100
    valid = mask.sum()
    if valid == 0:
        return {"acc": 0.0}
    correct = (pred_shift == labels_shift) & mask
    acc = float(correct.sum() / valid)
    return {"acc": acc}

# 在评估阶段即时压缩logits，避免累计巨大三维张量
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # 直接返回 argmax 的 token ids (int32)，大幅降低内存
    return torch.argmax(logits, dim=-1).to(torch.int32)

# --- 主函数 ---
def setup_distributed_logging():
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank != 0:
            logging.getLogger().setLevel(logging.ERROR)
            hf_logging.set_verbosity_error()
            disable_progress_bar()

def setup_model_and_tokenizer(model_args: ModelArguments):
    dist_logger.info(f"加载模型: {model_args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    # 强制使用不包含 <think> 注入的模板，避免模型在训练时学习到无意义的思维标签
    try:
        clean_tpl_path = os.path.join(os.path.dirname(__file__).rsplit('/', 1)[0], 'no_think_chat_template.jinja')
        if os.path.exists(clean_tpl_path):
            with open(clean_tpl_path, 'r', encoding='utf-8') as f:
                tokenizer.chat_template = f.read()
            dist_logger.info("已应用干净模板: no_think_chat_template.jinja")
        else:
            # 兼容直接运行目录为 DFT-Train/ 的情况
            alt_path = os.path.join(os.path.dirname(__file__), 'no_think_chat_template.jinja')
            if os.path.exists(alt_path):
                with open(alt_path, 'r', encoding='utf-8') as f:
                    tokenizer.chat_template = f.read()
                dist_logger.info("已应用干净模板: no_think_chat_template.jinja (alt)")
            else:
                dist_logger.warning("未找到 no_think_chat_template.jinja，继续使用默认模板")
    except Exception as e:
        dist_logger.warning(f"应用干净模板失败: {e}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 对于训练，使用left padding以避免Flash Attention问题
    tokenizer.padding_side = 'left'
    
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping.get(model_args.torch_dtype, torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code
    )
    
    return model, tokenizer

def setup_datasets(data_args: DataArguments, tokenizer, training_args: TrainingArguments):
    processor = DataProcessor(tokenizer, data_args.max_length, data_args.debug_data_processing)
    
    # 加载训练数据
    dist_logger.info("加载训练数据...")
    train_raw = processor.load_and_validate_datasets(data_args.data_files, "train")
    
    # 处理验证数据
    eval_raw = None
    if data_args.eval_data_files:
        dist_logger.info("加载验证数据...")
        eval_raw = processor.load_and_validate_datasets(data_args.eval_data_files, "eval")
    elif data_args.validation_split_percentage > 0:
        dist_logger.info(f"分割 {data_args.validation_split_percentage}% 作为验证集")
        split = train_raw.train_test_split(
            test_size=data_args.validation_split_percentage / 100.0,
            seed=training_args.seed
        )
        train_raw = split["train"]
        eval_raw = split["test"]
    
    # 预处理
    dist_logger.info(f"预处理训练数据: {len(train_raw)} 样本")
    processor.process_stats = {"success": 0, "failed": 0, "total": 0}
    
    with progress_bar_context(processor.is_main_process):
        train_processed = train_raw.map(
            processor.preprocess_chatml_function,
            batched=True,
            batch_size=10,
            remove_columns=train_raw.column_names,
            num_proc=min(4, data_args.preprocessing_num_workers),
            load_from_cache_file=False
        )
        
        train_processed = train_processed.filter(lambda x: len(x["input_ids"]) > 0)
    
    dist_logger.info(f"训练数据处理完成: {len(train_processed)} 样本")
    
    if len(train_processed) == 0:
        raise ValueError("预处理后训练数据为空")
    
    # 预处理验证数据
    eval_processed = None
    if eval_raw is not None and len(eval_raw) > 0:
        if data_args.max_eval_samples and len(eval_raw) > data_args.max_eval_samples:
            eval_raw = eval_raw.select(range(data_args.max_eval_samples))
        
        dist_logger.info(f"预处理验证数据: {len(eval_raw)} 样本")
        processor.process_stats = {"success": 0, "failed": 0, "total": 0}
        
        with progress_bar_context(processor.is_main_process):
            eval_processed = eval_raw.map(
                processor.preprocess_chatml_function,
                batched=True,
                batch_size=10,
                remove_columns=eval_raw.column_names,
                num_proc=min(4, data_args.preprocessing_num_workers),
                load_from_cache_file=False
            )
            
            eval_processed = eval_processed.filter(lambda x: len(x["input_ids"]) > 0)
        
        if len(eval_processed) == 0:
            dist_logger.warning("验证数据为空")
            eval_processed = None
        else:
            dist_logger.info(f"验证数据处理完成: {len(eval_processed)} 样本")
    
    return train_processed, eval_processed

def main():
    setup_distributed_logging()
    
    parser = HfArgumentParser((ModelArguments, DataArguments, DFTArguments, LoggingArguments, TrainingArguments))
    model_args, data_args, dft_args, logging_args, training_args = parser.parse_args_into_dataclasses()

    # 避免wandb交互
    if "wandb" in training_args.report_to:
        os.environ["WANDB_MODE"] = "offline"
    
    if training_args.local_rank <= 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 自动设置评估策略
    if data_args.eval_data_files or data_args.validation_split_percentage > 0:
        if training_args.eval_strategy == "no":
            training_args.eval_strategy = "steps"
    else:
        training_args.eval_strategy = "no"
    
    # 确保保存最优模型与指标选择生效
    if training_args.eval_strategy != "no":
        # 若用户未指定，则默认使用 eval_acc 作为最佳指标（存在 compute_metrics）
        if not getattr(training_args, "metric_for_best_model", None):
            training_args.metric_for_best_model = "eval_acc"
        if getattr(training_args, "greater_is_better", None) is None:
            training_args.greater_is_better = True
        # 处理 DeepSpeed + save_only_model 与 load_best_model_at_end 的冲突
        has_deepspeed = bool(getattr(training_args, "deepspeed", None))
        save_only_model = bool(getattr(training_args, "save_only_model", False))
        if has_deepspeed and save_only_model:
            if getattr(training_args, "load_best_model_at_end", False):
                dist_logger.warning("DeepSpeed 与 save_only_model 同时使用，将禁用 load_best_model_at_end，由自定义逻辑跟踪最优checkpoint。")
            training_args.load_best_model_at_end = False
        else:
            if not getattr(training_args, "load_best_model_at_end", False):
                training_args.load_best_model_at_end = True
    
    try:
        model, tokenizer = setup_model_and_tokenizer(model_args)
        
        if dft_args.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            dist_logger.info("启用梯度检查点")
        
        train_dataset, eval_dataset = setup_datasets(data_args, tokenizer, training_args)
        
        training_args.group_by_length = False
        
        if training_args.local_rank > 0:
            training_args.report_to = []
        
        # 评估阶段内存优化：不拼接logits，启用低精度评估，限制累计步数
        if hasattr(training_args, "eval_do_concat_logits"):
            training_args.eval_do_concat_logits = False
        if hasattr(training_args, "bf16") and training_args.bf16:
            setattr(training_args, "bf16_full_eval", True)
        if not getattr(training_args, "eval_accumulation_steps", 0):
            training_args.eval_accumulation_steps = 1

        # 创建trainer
        trainer = DFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_token_level_accuracy,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            dft_alpha=dft_args.dft_alpha,
            log_metrics_steps=logging_args.log_metrics_steps,
            use_simple_dft=dft_args.use_simple_dft,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
        )

        dist_logger.info(f"Start training - DFT alpha={dft_args.dft_alpha}")
        trainer.train()
        
        # 训练完成后输出最优 checkpoint 信息
        try:
            best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
            best_metric = getattr(trainer.state, "best_metric", None)
            metric_name = getattr(training_args, "metric_for_best_model", "eval_loss")
            if best_ckpt is not None:
                dist_logger.info(f"Training completed, best checkpoint: {best_ckpt} | {metric_name}={best_metric}")
            else:
                dist_logger.warning("Training completed, but no best checkpoint information (possibly no evaluation or no evaluation strategy)")
        except Exception:
            pass

        trainer.save_model(training_args.output_dir)
        # 确保将已应用的 chat_template 一并保存，供推理阶段加载
        try:
            tokenizer.save_pretrained(training_args.output_dir)
            dist_logger.info(f"Tokenizer saved: {training_args.output_dir}")
        except Exception as e:
            dist_logger.warning(f"Failed to save tokenizer: {e}")
        dist_logger.info(f"Model saved: {training_args.output_dir}")
        
    except Exception as e:
        dist_logger.error(f"Training failed: {e}")
        dist_logger.error(f"Details: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()