"""Chuyển dữ liệu extraction raw sang chat JSONL để finetune LLM."""

from __future__ import annotations

import argparse

from llm_extraction.utils import convert_raw_record, load_task_config, read_records, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse tham số command line cho bước convert dữ liệu.

    Returns:
        Namespace chứa input/output/config path.
    """
    parser = argparse.ArgumentParser(description="Convert raw extraction JSONL sang chat-SFT JSONL.")
    parser.add_argument("--input", required=True, help="File JSONL hoặc JSON raw có text và labels.")
    parser.add_argument("--output", required=True, help="File JSONL output theo messages format.")
    parser.add_argument("--task_config", default="configs/asqp.json", help="File JSON config chứa prompt và schema label.")
    return parser.parse_args()


def main() -> None:
    """Entry point convert dữ liệu."""
    args = parse_args()
    task_config = load_task_config(args.task_config)
    converted = [convert_raw_record(record, task_config) for record in read_records(args.input)]
    write_jsonl(args.output, converted)
    print(f"Converted {len(converted)} records -> {args.output}")


if __name__ == "__main__":
    main()
