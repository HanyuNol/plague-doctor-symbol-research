import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.config import MODEL_NAME, OUTPUT_DIR, META_DIR, TEMPERATURE, MAX_NEW_TOKENS
from src.utils import ensure_dir, load_json, extract_json_block

PROMPT = """你是一个严谨的公共卫生图像分析助手。
请分析这张图片，并严格输出 JSON，不要输出任何额外说明。

任务1：判断是否有人
- 如果图片中存在至少一个人（不论是否医务人员），则 "has_people": 1
- 否则 "has_people": 0

任务2：判断是否有穿防护服的医务人员（仅当 has_people=1 时认真判断）
- 如果画面中出现医护人员、医院工作人员、防疫人员，并且其穿着明显防护服、隔离服、防护面罩、面屏、全套医用防护装备(PPE)，则 "has_protective_medical_staff": 1
- 否则为 0
- 若 has_people=0，则此字段可设为 0

任务3：标签提取
从图像中提取若干标签。优先从以下集合中选择，允许少量补充：
["doctor","nurse","medical_staff","ppe","protective_suit","mask","face_shield","gloves",
 "hospital","clinic","patient","ambulance","testing","vaccination","microscope",
 "laboratory","virus","covid_sign","crowd","indoor","outdoor"]

输出格式必须是：
{
  "has_people": 0 或 1,
  "has_protective_medical_staff": 0 或 1,
  "confidence": 0.95,
  "tags": ["tag1","tag2",...],
  "reason": "简短判断理由"
}

注意：
1. 只能输出 JSON
2. 不要输出 markdown
3. confidence 取值范围 0-1
"""


def load_model():
    """加载模型：4-bit 量化 + 低分辨率，强制 GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 驱动和 PyTorch 安装。")

    print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download('qwen/Qwen2.5-VL-3B-Instruct', cache_dir='./models')
    except Exception as e:
        print(f"modelscope 下载失败，回退到 HuggingFace: {e}")
        model_dir = MODEL_NAME

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("✅ 使用 4-bit 量化加载模型")

    processor = AutoProcessor.from_pretrained(
        model_dir,
        min_pixels=64 * 28 * 28,
        max_pixels=320 * 28 * 28
    )

    device = next(model.parameters()).device
    print(f"✅ 模型运行设备: {device}")
    print(f"   GPU 显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    return model, processor


def infer_batch(model, processor, image_paths, prompt, batch_size=3):
    """批量推理多张图片"""
    if processor.tokenizer.padding_side != 'left':
        processor.tokenizer.padding_side = 'left'
    all_outputs = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start+batch_size]
        messages_list = []
        for img_path in batch_paths:
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            }])
        texts = []
        all_images = []
        for messages in messages_list:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            all_images.append(image_inputs[0] if image_inputs else None)

        inputs = processor(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        for i, gen_ids in enumerate(generated_ids):
            input_len = len(inputs.input_ids[i])
            trimmed = gen_ids[input_len:]
            output_text = processor.decode(trimmed, skip_special_tokens=True)
            all_outputs.append(output_text)
    return all_outputs


def classify_items_batch(model, processor, items, batch_size=3):
    """批量分类图片（每批 batch_size 张）"""
    rows = []
    # 过滤有效图片
    valid_items = []
    for item in items:
        img_path = item.get("save_path")
        if img_path and os.path.exists(img_path) and item.get("download_status") == "success":
            valid_items.append(item)
        else:
            row = dict(item)
            row.update({
                "has_people": None,
                "has_protective_medical_staff": None,
                "confidence": None,
                "tags": [],
                "people_count_range": "",
                "reason": "",
                "model_raw_output": "",
                "error": "图片文件不存在或下载失败"
            })
            rows.append(row)

    # 分批处理有效图片
    for start in tqdm(range(0, len(valid_items), batch_size), desc="批量分类"):
        batch_items = valid_items[start:start+batch_size]
        batch_paths = [it["save_path"] for it in batch_items]
        try:
            outputs = infer_batch(model, processor, batch_paths, PROMPT, batch_size)
            for item, output_text in zip(batch_items, outputs):
                row = dict(item)
                try:
                    result_json = extract_json_block(output_text)
                    row["has_people"] = int(result_json.get("has_people", 0))
                    row["has_protective_medical_staff"] = int(result_json.get("has_protective_medical_staff", 0))
                    row["confidence"] = float(result_json.get("confidence", 0.0))
                    row["tags"] = result_json.get("tags", [])
                    row["people_count_range"] = result_json.get("people_count_range", "unknown")
                    row["reason"] = result_json.get("reason", "")
                    row["model_raw_output"] = output_text
                    row["error"] = ""
                except Exception as e:
                    row.update({
                        "has_people": None,
                        "has_protective_medical_staff": None,
                        "confidence": None,
                        "tags": [],
                        "people_count_range": "",
                        "reason": "",
                        "model_raw_output": output_text,
                        "error": str(e)
                    })
                rows.append(row)
        except Exception as e:
            for item in batch_items:
                row = dict(item)
                row.update({
                    "has_people": None,
                    "has_protective_medical_staff": None,
                    "confidence": None,
                    "tags": [],
                    "people_count_range": "",
                    "reason": "",
                    "model_raw_output": "",
                    "error": f"批量推理失败: {e}"
                })
                rows.append(row)
    return rows


def main():
    ensure_dir(OUTPUT_DIR)
    meta_path = os.path.join(META_DIR, "unsplash_search_results.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("未找到抓取结果，请先运行 src/fetch_unsplash.py")

    records = load_json(meta_path)
    model, processor = load_model()

    # 批量分类，batch_size=3
    rows = classify_items_batch(model, processor, records, batch_size=3)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "image_classification_results.csv")
    json_path = os.path.join(OUTPUT_DIR, "image_classification_results.json")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("分类完成。")
    print(f"CSV 输出：{csv_path}")
    print(f"JSON 输出：{json_path}")


if __name__ == "__main__":
    main()