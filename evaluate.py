"""Evaluate model/LoRA extraction bằng generation và micro-F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_extraction.utils import (
    TaskConfig,
    build_messages,
    compute_micro_f1,
    convert_raw_record,
    load_task_config,
    parse_model_output,
    positive_int,
    read_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse tham số command line cho evaluate.

    Returns:
        Namespace cấu hình evaluate.
    """
    parser = argparse.ArgumentParser(description="Evaluate extraction model bằng micro-F1.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Đường dẫn LoRA adapter. Bỏ trống nếu model đã merge.")
    parser.add_argument("--validation_file", required=True, help="Dev/test JSONL có text/labels hoặc messages format.")
    parser.add_argument("--task_config", default="configs/asqp.json", help="File JSON config chứa prompt và schema label.")
    parser.add_argument("--output_predictions", default=None, help="File JSONL lưu prediction chi tiết.")
    parser.add_argument("--batch_size", type=positive_int, default=4)
    parser.add_argument("--max_new_tokens", type=positive_int, default=256)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load base model, tokenizer và adapter LoRA nếu có.

    Args:
        args: Cấu hình command line.

    Returns:
        Tuple model, tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def load_extraction_records(path: str, task_config: TaskConfig) -> list[dict[str, Any]]:
    """Đọc JSONL và chuẩn hóa về text/labels/messages theo task_config.

    Args:
        path: Đường dẫn JSONL.
        task_config: Cấu hình prompt/schema của task.

    Returns:
        Danh sách record đã chuẩn hóa.
    """
    return [convert_raw_record(record, task_config) for record in read_records(path)]


@torch.no_grad()
def generate_predictions(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    task_config: TaskConfig,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate prediction cho từng text.

    Args:
        model: Causal LM đã load.
        tokenizer: Tokenizer tương ứng.
        records: Dữ liệu dev/test.
        task_config: Cấu hình prompt/schema của task.
        batch_size: Batch size generate.
        max_new_tokens: Số token mới tối đa.

    Returns:
        Danh sách record prediction chi tiết.
    """
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(record["text"], None, task_config),
                tokenize=False,
                add_generation_prompt=True,
            )
            for record in batch_records
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = encoded["input_ids"].shape[1]
        for row_index in range(len(batch_records)):
            output_ids = generated[row_index][prompt_length:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            record = batch_records[row_index]
            outputs.append(
                {
                    "id": record.get("id"),
                    "text": record["text"],
                    "gold": record["labels"],
                    "raw_output": output_text,
                    "prediction": parse_model_output(output_text, task_config),
                }
            )
    return outputs


def main() -> None:
    """Entry point evaluate model."""
    args = parse_args()
    task_config = load_task_config(args.task_config)
    records = load_extraction_records(args.validation_file, task_config)
    model, tokenizer = load_model(args)
    predictions = generate_predictions(model, tokenizer, records, task_config, args.batch_size, args.max_new_tokens)
    metrics = compute_micro_f1(
        [item["prediction"] for item in predictions],
        [item["gold"] for item in predictions],
        task_config,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_predictions:
        write_jsonl(args.output_predictions, predictions)
        metrics_path = Path(args.output_predictions).with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
