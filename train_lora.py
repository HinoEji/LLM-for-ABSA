"""Finetune LoRA Qwen Instruct cho task extraction và chọn best checkpoint bằng F1 dev."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from llm_extraction.utils import (
    TaskConfig,
    build_messages,
    compute_micro_f1,
    convert_raw_record,
    load_task_config,
    parse_model_output,
    positive_int,
    read_records,
)


IGNORE_INDEX = -100

def parse_args() -> argparse.Namespace:
    """Parse tham số command line cho finetune LoRA.

    Returns:
        Namespace cấu hình huấn luyện.
    """
    parser = argparse.ArgumentParser(description="Finetune LoRA Qwen Instruct cho task extraction.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", required=True, help="Train JSONL có text/labels hoặc messages format.")
    parser.add_argument("--validation_file", required=True, help="Dev JSONL có text/labels hoặc messages format.")
    parser.add_argument("--task_config", default="configs/asqp.json", help="File JSON config chứa prompt và schema label.")
    parser.add_argument("--output_dir", default="outputs/qwen-extraction-lora")
    parser.add_argument("--max_seq_length", type=positive_int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=positive_int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=positive_int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=1)
    parser.add_argument("--eval_steps", type=positive_int, default=1)
    parser.add_argument("--eval_strategy", type=str, default="epoch"),
    parser.add_argument("--save_strategy", type=str, default="epoch"),
    parser.add_argument("--steps", type=positive_int, default=1)
    parser.add_argument("--save_steps", type=positive_int, default=1)
    parser.add_argument("--logging_strategy", type=str, default="epoch")
    parser.add_argument("--logging_steps", type=positive_int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lora_r", type=positive_int, default=16)
    parser.add_argument("--lora_alpha", type=positive_int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--generation_max_new_tokens", type=positive_int, default=256)
    parser.add_argument("--generation_batch_size", type=positive_int, default=4)
    parser.add_argument("--use_4bit", action="store_true", help="Bật QLoRA 4-bit nếu môi trường hỗ trợ bitsandbytes.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true", help="Dùng cho một số mô hình có hỗ trợ chế độ thinking")
    return parser.parse_args()

args = parse_args()



@dataclass
class DataCollatorForCausalLM:
    """Pad input_ids/attention_mask/labels cho causal LM."""

    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Tạo batch tensor từ các feature đã tokenize.

        Args:
            features: Danh sách feature có input_ids, attention_mask và labels.

        Returns:
            Batch tensor đã pad.
        """
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ExtractionTrainer(Trainer):
    """Trainer thêm generation-based F1 vào mỗi lần evaluate."""

    def __init__(
        self,
        *args: Any,
        eval_records: list[dict[str, Any]],
        task_config: TaskConfig,
        generation_max_new_tokens: int,
        generation_batch_size: int,
        **kwargs: Any,
    ) -> None:
        """Khởi tạo trainer với dữ liệu dev raw để generate.

        Args:
            eval_records: Dev records chứa text và labels gold.
            task_config: Cấu hình prompt/schema của task.
            generation_max_new_tokens: Số token tối đa sinh khi evaluate.
            generation_batch_size: Batch size generation trên dev.
            kwargs: Tham số truyền tiếp cho transformers.Trainer.
        """
        super().__init__(*args, **kwargs)
        self.eval_records = eval_records
        self.task_config = task_config
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_batch_size = generation_batch_size

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate loss bằng Trainer gốc và F1 bằng generation.

        Args:
            eval_dataset: Dataset tokenized tùy chọn.
            ignore_keys: Key bỏ qua khi evaluate.
            metric_key_prefix: Prefix metric, thường là eval.

        Returns:
            Metrics gồm eval_loss và eval_f1.
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        f1_metrics = self._evaluate_generation()
        metrics[f"{metric_key_prefix}_precision"] = f1_metrics["precision"]
        metrics[f"{metric_key_prefix}_recall"] = f1_metrics["recall"]
        metrics[f"{metric_key_prefix}_f1"] = f1_metrics["f1"]
        metrics[f"{metric_key_prefix}_tp"] = f1_metrics["tp"]
        metrics[f"{metric_key_prefix}_fp"] = f1_metrics["fp"]
        metrics[f"{metric_key_prefix}_fn"] = f1_metrics["fn"]
        self.log(metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_generation(self) -> dict[str, float]:
        """Generate trên dev và tính micro-F1 exact match."""
        was_training = self.model.training
        self.model.eval()
        predictions: list[list[dict[str, Any]]] = []
        references: list[list[dict[str, Any]]] = []

        for start in range(0, len(self.eval_records), self.generation_batch_size):
            batch_records = self.eval_records[start : start + self.generation_batch_size]
            prompts = [
                self.processing_class.apply_chat_template(
                    build_messages(record["text"], None, self.task_config),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking
                )
                for record in batch_records
            ]
            encoded = self.processing_class(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
            )

            # Chỉ decode phần token mới để tránh parse lại prompt.
            prompt_length = encoded["input_ids"].shape[1]
            for row_index in range(len(batch_records)):
                output_ids = generated[row_index][prompt_length:]
                output_text = self.processing_class.decode(output_ids, skip_special_tokens=True)
                predictions.append(parse_model_output(output_text, self.task_config))
                references.append(batch_records[row_index]["labels"])
            
        random_idx = random.randint(0, len(batch_records) - 1)

        print("==============================Text==============================\n\n")
        print(self.eval_records[random_idx]["text"])

        print("==============================Predictions==============================\n\n")
        print(predictions[random_idx])

        print("==============================References==============================\n\n")
        print(references[random_idx])

        if was_training:
            self.model.train()
        return compute_micro_f1(predictions, references, self.task_config)



def load_extraction_records(path: str, task_config: TaskConfig) -> list[dict[str, Any]]:
    """Đọc JSONL và chuẩn hóa về text/labels/messages theo task_config.

    Args:
        path: Đường dẫn JSONL.
        task_config: Cấu hình prompt/schema của task.

    Returns:
        Danh sách record đã chuẩn hóa.
    """
    return [convert_raw_record(record, task_config) for record in read_records(path)]


def tokenize_record(record: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, list[int]]:
    """Tokenize một record và mask loss chỉ trên câu trả lời assistant.

    Args:
        record: Chat record có messages.
        tokenizer: Tokenizer có chat template.
        max_seq_length: Độ dài tối đa.

    Returns:
        Dict input_ids/attention_mask/labels cho Trainer.
    """
    prompt_messages = record["messages"][:-1]

    full_text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=args.enable_thinking)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_length:
        # Giữ phần cuối vì target assistant nằm cuối sample.
        overflow = len(full_ids) - max_seq_length
        full_ids = full_ids[overflow:]
        prompt_cut = max(0, len(prompt_ids) - overflow)
    else:
        prompt_cut = len(prompt_ids)

    labels = [IGNORE_INDEX] * min(prompt_cut, len(full_ids)) + full_ids[min(prompt_cut, len(full_ids)) :]
    attention_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def build_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load model/tokenizer và gắn LoRA adapter.

    Args:
        args: Cấu hình command line.

    Returns:
        Tuple model, tokenizer đã sẵn sàng train.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def save_training_config(args: argparse.Namespace, task_config: TaskConfig) -> None:
    """Lưu cấu hình train và task config để dễ tái lập.

    Args:
        args: Namespace command line.
        task_config: Cấu hình prompt/schema của task.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
    with (output_dir / "task_config.json").open("w", encoding="utf-8") as file:
        json.dump(task_config.to_dict(), file, ensure_ascii=False, indent=2)


def main() -> None:
    """Entry point finetune LoRA."""
    # args = parse_args()
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save_steps phải bằng hoặc là bội số của --eval_steps để chọn best model theo eval_f1.")

    task_config = load_task_config(args.task_config)
    train_records = load_extraction_records(args.train_file, task_config)
    eval_records = load_extraction_records(args.validation_file, task_config)
    model, tokenizer = build_model_and_tokenizer(args)

    train_dataset = Dataset.from_list(
        [tokenize_record(record, tokenizer, args.max_seq_length) for record in train_records]
    )
    eval_dataset = Dataset.from_list(
        [tokenize_record(record, tokenizer, args.max_seq_length) for record in eval_records]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = ExtractionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForCausalLM(tokenizer),
        eval_records=eval_records,
        task_config=task_config,
        generation_max_new_tokens=args.generation_max_new_tokens,
        generation_batch_size=args.generation_batch_size,
    )
    save_training_config(args, task_config)
    trainer.train()

    # Sau train, Trainer đã load checkpoint tốt nhất theo eval_f1.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_metrics = trainer.evaluate()
    with Path(args.output_dir, "final_eval_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, ensure_ascii=False, indent=2)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
