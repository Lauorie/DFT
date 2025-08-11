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
        
        # 使用内部 _save，确保兼容 deepspeed/fp16 等场景
        self._save(output_dir)
        
        # 保存 tokenizer
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:
                pass
        
        # 轮转旧 checkpoint
        try:
            # 不同版本签名不同，优先无参调用
            self._rotate_checkpoints()
        except TypeError:
            try:
                self._rotate_checkpoints(use_mtime=False)
            except Exception:
                pass

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
                f"处理进度: 总计 {self.process_stats['total']}, "
                f"成功 {self.process_stats['success']}, "
                f"失败 {self.process_stats['failed']}"
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
    
    try:
        model, tokenizer = setup_model_and_tokenizer(model_args)
        
        if dft_args.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            dist_logger.info("启用梯度检查点")
        
        train_dataset, eval_dataset = setup_datasets(data_args, tokenizer, training_args)
        
        training_args.group_by_length = False
        
        if training_args.local_rank > 0:
            training_args.report_to = []
        
        # 创建trainer
        trainer = DFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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

        dist_logger.info(f"开始训练 - DFT alpha={dft_args.dft_alpha}")
        trainer.train()
        
        trainer.save_model(training_args.output_dir)
        dist_logger.info(f"模型已保存: {training_args.output_dir}")
        
    except Exception as e:
        dist_logger.error(f"训练失败: {e}")
        dist_logger.error(f"详情: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()