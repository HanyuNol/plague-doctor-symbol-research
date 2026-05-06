import json
import os
import re
from typing import Any, Dict


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(text: str) -> str:
    if text is None:
        text = "image"
    text = str(text)
    text = re.sub(r"[^\w\-_. ]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120].strip("_") or "image"


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取 JSON。
    当模型前后夹带说明文本时，截取最外层的大括号内容。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到有效 JSON。原始输出: {text}")

    raw = text[start:end + 1]
    return json.loads(raw)