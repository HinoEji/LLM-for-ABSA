"""Tiện ích chung cho các bài toán trích xuất label bằng LLM."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TaskConfig:
    """Cấu hình prompt và schema label cho một task extraction."""

    system_prompt: str
    user_template: str
    label_fields: tuple[str, ...] = ()
    allowed_values: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    null_token: str = "NULL"

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "TaskConfig":
        """Tạo TaskConfig từ object JSON.

        Args:
            raw_config: Object đọc từ file JSON config.

        Returns:
            TaskConfig đã validate tối thiểu.
        """
        system_prompt = raw_config.get("system_prompt")
        user_template = raw_config.get("user_template")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("task_config cần trường system_prompt là chuỗi không rỗng.")
        if not isinstance(user_template, str) or "{text}" not in user_template:
            raise ValueError("task_config cần trường user_template chứa placeholder {text}.")

        label_fields = raw_config.get("label_fields", [])
        if label_fields is None:
            label_fields = []
        if not isinstance(label_fields, list) or not all(isinstance(item, str) for item in label_fields):
            raise ValueError("label_fields phải là list[str] nếu được khai báo.")

        allowed_values = raw_config.get("allowed_values", {})
        if allowed_values is None:
            allowed_values = {}
        if not isinstance(allowed_values, dict):
            raise ValueError("allowed_values phải là object nếu được khai báo.")

        normalized_allowed_values: dict[str, tuple[Any, ...]] = {}
        for field_name, values in allowed_values.items():
            if not isinstance(field_name, str) or not isinstance(values, list):
                raise ValueError("allowed_values phải có dạng {field_name: list}.")
            normalized_allowed_values[field_name] = tuple(values)

        null_token = raw_config.get("null_token", "NULL")
        if not isinstance(null_token, str) or not null_token:
            raise ValueError("null_token phải là chuỗi không rỗng.")

        return cls(
            system_prompt=system_prompt.strip(),
            user_template=user_template,
            label_fields=tuple(label_fields),
            allowed_values=normalized_allowed_values,
            null_token=null_token,
        )

    def to_dict(self) -> dict[str, Any]:
        """Chuyển TaskConfig về object JSON-serializable.

        Returns:
            Dict có thể ghi ra file JSON.
        """
        return {
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "label_fields": list(self.label_fields),
            "allowed_values": {key: list(value) for key, value in self.allowed_values.items()},
            "null_token": self.null_token,
        }


def load_task_config(path: str | Path) -> TaskConfig:
    """Đọc task config từ file JSON.

    Args:
        path: Đường dẫn file JSON chứa prompt/schema.

    Returns:
        TaskConfig dùng cho convert/train/evaluate.
    """
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        project_root_config = Path(__file__).resolve().parent.parent / config_path
        if project_root_config.exists():
            config_path = project_root_config

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)
    if not isinstance(raw_config, dict):
        raise ValueError("task_config phải là một JSON object.")
    return TaskConfig.from_dict(raw_config)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Đọc file JSONL và trả về danh sách bản ghi.

    Args:
        path: Đường dẫn file JSONL.

    Returns:
        Danh sách object JSON theo từng dòng.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi ở dòng {line_number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL dòng {line_number} phải là object.")
            records.append(parsed)
    return records


def read_json(path: str | Path) -> list[dict[str, Any]]:
    """Đọc file JSON chứa list record hoặc object có trường data/records.

    Args:
        path: Đường dẫn file JSON.

    Returns:
        Danh sách object record.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        parsed = json.load(file)
    if isinstance(parsed, dict):
        for field_name in ("data", "records", "examples"):
            if field_name in parsed:
                parsed = parsed[field_name]
                break
    if not isinstance(parsed, list):
        raise ValueError("File JSON phải là list record hoặc object có field data/records/examples.")
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Mỗi phần tử trong file JSON phải là object.")
    return parsed


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Đọc dữ liệu input từ JSONL hoặc JSON array.

    Args:
        path: Đường dẫn file .jsonl hoặc .json.

    Returns:
        Danh sách object record.
    """
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        return read_json(input_path)
    return read_jsonl(input_path)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Ghi danh sách bản ghi ra JSONL UTF-8.

    Args:
        path: Đường dẫn file output.
        records: Iterable chứa các object JSON-serializable.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def normalize_text(value: Any, null_token: str = "NULL") -> Any:
    """Strip chuỗi đầu/cuối và không làm sạch thêm.

    Args:
        value: Giá trị cần chuẩn hóa.
        null_token: Giá trị thay thế cho None hoặc chuỗi rỗng.

    Returns:
        Chuỗi đã strip, null_token, hoặc giá trị không phải chuỗi được giữ nguyên.
    """
    if value is None:
        return null_token
    if not isinstance(value, str):
        return value
    text = value.strip()
    return text if text else null_token


def normalize_input_text(value: Any) -> str:
    """Chuẩn hóa text đầu vào của mẫu bằng strip đầu/cuối.

    Args:
        value: Giá trị field text/review/sentence.

    Returns:
        Text đã strip đầu/cuối.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Mỗi record cần trường text/review/sentence là chuỗi không rỗng.")
    return value.strip()


def normalize_label(raw_label: dict[str, Any], task_config: TaskConfig) -> dict[str, Any]:
    """Chuẩn hóa một label theo schema task.

    Args:
        raw_label: Label raw, có thể là NER label hoặc quadruplet.
        task_config: Cấu hình schema/prompt của task.

    Returns:
        Label đã strip các field string và validate allowed_values.
    """
    if task_config.label_fields:
        field_names = task_config.label_fields
    else:
        field_names = tuple(raw_label.keys())

    normalized = {
        field_name: normalize_text(raw_label.get(field_name), task_config.null_token)
        for field_name in field_names
    }
    for field_name, valid_values in task_config.allowed_values.items():
        if field_name in normalized and normalized[field_name] not in valid_values:
            raise ValueError(
                f"{field_name} phải thuộc {list(valid_values)}, nhận được: {normalized[field_name]!r}"
            )
    return normalized


def canonical_label_key(label: dict[str, Any]) -> str:
    """Tạo khóa canonical để exact-match label trong metric.

    Args:
        label: Label đã chuẩn hóa.

    Returns:
        Chuỗi JSON canonical, độc lập với thứ tự key trong dict.
    """
    return json.dumps(label, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_labels(
    raw_labels: Any,
    task_config: TaskConfig,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Chuẩn hóa list label và loại duplicate trong cùng một text.

    Args:
        raw_labels: Giá trị thô kỳ vọng là list[dict].
        task_config: Cấu hình schema/prompt của task.
        strict: True thì raise lỗi với label sai; False thì bỏ qua label sai.

    Returns:
        Danh sách label hợp lệ đã chuẩn hóa.
    """
    if raw_labels is None:
        return []
    if not isinstance(raw_labels, list):
        if strict:
            raise ValueError("Trường labels phải là list.")
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_labels:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("Mỗi phần tử trong labels phải là object.")
            continue
        try:
            label = normalize_label(item, task_config)
        except ValueError:
            if strict:
                raise
            continue
        key = canonical_label_key(label)
        if key not in seen:
            normalized.append(label)
            seen.add(key)
    return normalized


def extract_record_text(record: dict[str, Any]) -> str:
    """Lấy text từ record, hỗ trợ cả tên field cũ.

    Args:
        record: Record raw hoặc đã convert.

    Returns:
        Text đã strip đầu/cuối.
    """
    for field_name in ("text", "review", "sentence"):
        if field_name in record:
            return normalize_input_text(record[field_name])
    raise ValueError("Mỗi record cần một trong các trường text/review/sentence.")


def extract_record_labels(record: dict[str, Any]) -> Any:
    """Lấy labels từ record, hỗ trợ cả tên field ASQP cũ.

    Args:
        record: Record raw hoặc đã convert.

    Returns:
        Giá trị labels/quads/quadruplets thô.
    """
    if "labels" in record:
        return record["labels"]
    if "quadruplets" in record:
        return record["quadruplets"]
    if "quads" in record:
        return record["quads"]
    return []


def format_assistant_answer(labels: list[dict[str, Any]], task_config: TaskConfig) -> str:
    """Format target assistant thành JSON array một dòng.

    Args:
        labels: Danh sách label đã hoặc chưa chuẩn hóa.
        task_config: Cấu hình schema/prompt của task.

    Returns:
        Chuỗi JSON array dùng làm target khi finetune.
    """
    normalized = normalize_labels(labels, task_config)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    text: str,
    labels: list[dict[str, Any]] | None,
    task_config: TaskConfig,
) -> list[dict[str, str]]:
    """Tạo messages theo chat format cho model Instruct.

    Args:
        text: Nội dung text/review.
        labels: Gold labels. Nếu None thì chỉ tạo prompt để generate.
        task_config: Cấu hình prompt/schema của task.

    Returns:
        Danh sách message role/content.
    """
    messages = [
        {"role": "system", "content": task_config.system_prompt},
        {"role": "user", "content": task_config.user_template.replace("{text}", text)},
    ]
    if labels is not None:
        messages.append({"role": "assistant", "content": format_assistant_answer(labels, task_config)})
    return messages


def convert_raw_record(record: dict[str, Any], task_config: TaskConfig) -> dict[str, Any]:
    """Chuyển một record raw sang chat-SFT record generic.

    Args:
        record: Record có text và labels; vẫn hỗ trợ review/quadruplets cũ.
        task_config: Cấu hình prompt/schema của task.

    Returns:
        Record gồm id tùy chọn, text, labels chuẩn hóa và messages.
    """
    text = extract_record_text(record)
    labels = normalize_labels(extract_record_labels(record), task_config)
    converted = {
        "text": text,
        "labels": labels,
        "messages": build_messages(text, labels, task_config),
    }
    if "id" in record:
        converted["id"] = record["id"]
    return converted


def parse_model_output(text: str, task_config: TaskConfig) -> list[dict[str, Any]]:
    """Parse JSON array label từ output model.

    Args:
        text: Chuỗi model generate.
        task_config: Cấu hình schema/prompt của task.

    Returns:
        Danh sách label chuẩn hóa; trả [] nếu không parse được.
    """
    stripped = text.strip()
    candidates = [stripped]

    # Một số model sinh kèm text ngoài JSON hoặc markdown fence; thử cắt array đầu tiên.
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "labels" in parsed:
                parsed = parsed["labels"]
            return normalize_labels(parsed, task_config, strict=False)
        except (json.JSONDecodeError, TypeError):
            continue
    return []


def labels_to_key_set(labels: list[dict[str, Any]], task_config: TaskConfig) -> set[str]:
    """Chuyển list label thành set khóa canonical cho metric.

    Args:
        labels: Danh sách label raw hoặc đã chuẩn hóa.
        task_config: Cấu hình schema/prompt của task.

    Returns:
        Set khóa canonical.
    """
    return {canonical_label_key(label) for label in normalize_labels(labels, task_config)}


def compute_micro_f1(
    predictions: Iterable[list[dict[str, Any]]],
    references: Iterable[list[dict[str, Any]]],
    task_config: TaskConfig,
) -> dict[str, float]:
    """Tính micro precision/recall/F1 theo exact label match.

    Args:
        predictions: Danh sách dự đoán theo từng text.
        references: Danh sách gold label theo từng text.
        task_config: Cấu hình schema/prompt của task.

    Returns:
        Dict chứa precision, recall, f1, tp, fp, fn.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for pred_labels, gold_labels in zip(predictions, references, strict=True):
        pred_set = labels_to_key_set(pred_labels, task_config)
        gold_set = labels_to_key_set(gold_labels, task_config)
        true_positive += len(pred_set & gold_set)
        false_positive += len(pred_set - gold_set)
        false_negative += len(gold_set - pred_set)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def positive_int(value: str) -> int:
    """Validate argparse int dương.

    Args:
        value: Chuỗi từ command line.

    Returns:
        Số nguyên dương.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Giá trị phải là số nguyên dương.")
    return parsed
