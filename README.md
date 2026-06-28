# LLM Extraction LoRA Fine-tuning

Repo này dùng để fine-tune LoRA cho các task trích xuất có output dạng danh sách label JSON, ví dụ:

- ASQP: `category`, `aspect`, `opinion`, `polarity`
- NER: `entity`, `type`

Pipeline chính dùng cùng một format dữ liệu `text` + `labels`, prompt được load từ file JSON trong `configs/`.

## Cài đặt

```powershell
pip install -r requirements.txt
```

Nếu dùng QLoRA 4-bit trên Linux, `bitsandbytes` sẽ được cài theo `requirements.txt`. Trên Windows, nên chạy LoRA thường trước, hoặc dùng môi trường Linux/WSL nếu cần 4-bit.

## Format Dữ Liệu

Input có thể là `.jsonl` hoặc `.json`.

JSONL, mỗi dòng là một sample:

```json
{"id":"1","text":"Barack Obama visited Hanoi.","labels":[{"entity":"Barack Obama","type":"PER"},{"entity":"Hanoi","type":"LOC"}]}
{"id":"2","text":"Apple released a new iPhone.","labels":[{"entity":"Apple","type":"ORG"},{"entity":"iPhone","type":"PRODUCT"}]}
```

JSON dạng list:

```json
[
  {
    "id": "1",
    "text": "The pizza was great.",
    "labels": [
      {"category": "FOOD#QUALITY", "aspect": "pizza", "opinion": "great", "polarity": "POS"}
    ]
  }
]
```

JSON dạng object cũng được nếu dữ liệu nằm trong `data`, `records`, hoặc `examples`:

```json
{
  "data": [
    {
      "id": "1",
      "text": "Barack Obama visited Hanoi.",
      "labels": [
        {"entity": "Barack Obama", "type": "PER"},
        {"entity": "Hanoi", "type": "LOC"}
      ]
    }
  ]
}
```

Quy ước xử lý text:

- `text` chỉ được `strip()` khoảng trắng đầu/cuối.
- Các field string trong `labels` cũng chỉ được `strip()` đầu/cuối.
- Chuỗi rỗng hoặc `null` trong label được đổi thành `NULL`.
- Một `text` có thể có nhiều label trong cùng danh sách `labels`.

## Task Config

Prompt và schema label nằm trong file JSON:

- `configs/asqp.json`
- `configs/ner.json`

Ví dụ config NER:

```json
{
  "system_prompt": "Bạn là hệ thống trích xuất thực thể có tên. Chỉ trả về JSON hợp lệ, không giải thích thêm.",
  "user_template": "Text:\n{text}\n\nHãy trích xuất tất cả named entity trong text. Mỗi label có đúng 2 trường: entity và type. Nếu một text có nhiều entity, trả về tất cả trong cùng một JSON array. Nếu không có entity nào, trả về [].",
  "label_fields": ["entity", "type"],
  "allowed_values": {},
  "null_token": "NULL"
}
```

`user_template` bắt buộc có `{text}`. Nếu muốn NER có span, đổi schema thành:

```json
"label_fields": ["entity", "type", "start", "end"]
```

Nếu field nào chỉ được nhận một số giá trị cố định, khai báo trong `allowed_values`. Ví dụ ASQP:

```json
"allowed_values": {
  "polarity": ["POS", "NEG", "NEU"]
}
```

## Chuẩn Bị Dữ Liệu Chat

Bạn có thể train trực tiếp từ file raw `.json` hoặc `.jsonl`. Tuy vậy, nên convert trước để kiểm tra prompt và target:

```powershell
python prepare_data.py `
  --input data/train.json `
  --output data/train.chat.jsonl `
  --task_config configs/ner.json
```

Với ASQP:

```powershell
python prepare_data.py `
  --input data/asqp_train.json `
  --output data/asqp_train.chat.jsonl `
  --task_config configs/asqp.json
```

Output `.chat.jsonl` sẽ có dạng:

```json
{"text":"Barack Obama visited Hanoi.","labels":[{"entity":"Barack Obama","type":"PER"}],"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"[{\"entity\":\"Barack Obama\",\"type\":\"PER\"}]"}]}
```

## Fine-tune LoRA

NER:

```powershell
python train_lora.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --train_file data/ner_train.chat.jsonl `
  --validation_file data/ner_dev.chat.jsonl `
  --task_config configs/ner.json `
  --output_dir outputs/qwen-ner-lora `
  --eval_steps 200 `
  --save_steps 200
```

ASQP:

```powershell
python train_lora.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --train_file data/asqp_train.chat.jsonl `
  --validation_file data/asqp_dev.chat.jsonl `
  --task_config configs/asqp.json `
  --output_dir outputs/qwen-asqp-lora `
  --eval_steps 200 `
  --save_steps 200
```

Nếu muốn dùng QLoRA 4-bit:

```powershell
python train_lora.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --train_file data/ner_train.chat.jsonl `
  --validation_file data/ner_dev.chat.jsonl `
  --task_config configs/ner.json `
  --output_dir outputs/qwen-ner-lora `
  --use_4bit
```

`save_steps` phải bằng hoặc là bội số của `eval_steps`, vì script chọn best checkpoint theo `eval_f1`.

## Evaluate

```powershell
python evaluate.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --adapter_path outputs/qwen-ner-lora `
  --validation_file data/ner_dev.chat.jsonl `
  --task_config configs/ner.json `
  --output_predictions outputs/ner_dev_predictions.jsonl
```

Metric là micro precision/recall/F1 theo exact match trên từng object trong `labels`.

File prediction có dạng:

```json
{"id":"1","text":"Barack Obama visited Hanoi.","gold":[...],"raw_output":"[...]","prediction":[...]}
```

## Output Model

Output LoRA là adapter, không phải toàn bộ Qwen model. Thư mục output thường có:

```text
adapter_model.safetensors
adapter_config.json
tokenizer.json
tokenizer_config.json
training_config.json
task_config.json
final_eval_metrics.json
```

Khi inference hoặc evaluate, cần load cả base model và adapter:

```powershell
python evaluate.py `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --adapter_path outputs/qwen-ner-lora `
  --validation_file data/ner_dev.json `
  --task_config configs/ner.json
```

## Cấu Trúc Chính

Các file chính của repo:

- `prepare_data.py`: convert dữ liệu raw sang chat JSONL.
- `train_lora.py`: fine-tune LoRA và chọn best checkpoint theo `eval_f1`.
- `evaluate.py`: evaluate model hoặc adapter bằng generation F1.
- `llm_extraction/utils.py`: utility đọc dữ liệu, load config, build prompt, parse output và tính F1.
- `configs/asqp.json`: prompt/schema mẫu cho ASQP.
- `configs/ner.json`: prompt/schema mẫu cho NER.

## Kiểm Tra

Chạy unit test:

```powershell
python -m unittest discover -s tests
```
