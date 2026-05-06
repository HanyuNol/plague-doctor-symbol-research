import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.classify_images import load_model
from src.config import TEMPERATURE, MAX_NEW_TOKENS
from src.utils import extract_json_block

PROMPT_PINTEREST = """你是一个严谨的图像分析专家。
请分析这张图片，并严格输出 JSON，不要输出任何额外说明。

任务1：判断是否有人物
- 如果图片中存在至少一个人（不论古今），则 "has_people": 1
- 否则 "has_people": 0

任务2：判断是否有“鸟嘴医生”形象（仅当 has_people=1 时认真判断）
鸟嘴医生特征：穿着黑色或深色长袍，戴着鸟嘴形状的面具（喙状突起），可能配有宽檐帽、手套、手杖。
如果出现符合上述特征的医生形象，则 "has_plague_doctor": 1，否则为 0。若 has_people=0，此字段为 0。

任务3：判断是否有“死神”形象（仅当 has_people=1 时认真判断）
死神特征：骷髅骨架、身披黑袍、手持镰刀，或明显的死亡象征（如骷髅头、骨镰等）。
如果出现上述特征，则 "has_grim_reaper": 1，否则为 0。若 has_people=0，此字段为 0。

任务4：标签提取
从以下集合中选取相关标签，可补充少量其他标签：
["plague_doctor","grim_reaper","skeleton","scythe","robe","mask","beak_mask","medieval","historical","crowd","indoor","outdoor"]

输出格式：
{
  "has_people": 0/1,
  "has_plague_doctor": 0/1,
  "has_grim_reaper": 0/1,
  "confidence": 0.95,
  "tags": ["tag1","tag2",...],
  "reason": "简短判断理由"
}
注意：只能输出 JSON，不要 markdown。
"""


def infer_single(model, processor, image_path: str, prompt: str) -> str:
    """单张图片推理（优化版）"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # 确定性生成，关闭采样，速度更快
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 关闭采样，加速
            temperature=None,  # do_sample=False时temperature无效
        )

    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
    output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
    return output_text


def infer_batch(model, processor, image_paths, prompt, batch_size=3):
    """批量推理多张图片（每批最多 batch_size 张）"""
    if processor.tokenizer.padding_side != 'left':
        processor.tokenizer.padding_side = 'left'
    all_outputs = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
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
    valid_items = [item for item in items if
                   item.get("save_path") and os.path.exists(item.get("save_path")) and item.get(
                       "download_status") == "success"]
    invalid_items = [item for item in items if item not in valid_items]
    for item in invalid_items:
        row = dict(item)
        row.update(
            {"has_people": None, "has_plague_doctor": None, "has_grim_reaper": None, "error": "文件不存在或下载失败"})
        rows.append(row)

    # 分批处理有效图片
    for start in tqdm(range(0, len(valid_items), batch_size), desc="批量分类"):
        batch_items = valid_items[start:start + batch_size]
        batch_paths = [it["save_path"] for it in batch_items]
        try:
            outputs = infer_batch(model, processor, batch_paths, PROMPT_PINTEREST, batch_size)
            for item, output_text in zip(batch_items, outputs):
                row = dict(item)
                try:
                    result_json = extract_json_block(output_text)
                    row["has_people"] = int(result_json.get("has_people", 0))
                    row["has_plague_doctor"] = int(result_json.get("has_plague_doctor", 0))
                    row["has_grim_reaper"] = int(result_json.get("has_grim_reaper", 0))
                    row["confidence"] = float(result_json.get("confidence", 0.0))
                    row["tags"] = result_json.get("tags", [])
                    row["reason"] = result_json.get("reason", "")
                    row["error"] = ""
                except Exception as e:
                    row.update(
                        {"has_people": None, "has_plague_doctor": None, "has_grim_reaper": None, "error": str(e)})
                rows.append(row)
        except Exception as e:
            for item in batch_items:
                row = dict(item)
                row.update({"has_people": None, "has_plague_doctor": None, "has_grim_reaper": None,
                            "error": f"批量推理失败: {e}"})
                rows.append(row)
    return rows


def update_classification_results(csv_path, json_path, new_classified_rows):
    """合并分类结果到 CSV 和 JSON"""
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        existing_rows = existing_df.to_dict('records')
    else:
        existing_rows = []
    url_map = {row["image_url"]: row for row in existing_rows}
    for new_row in new_classified_rows:
        url_map[new_row["image_url"]] = new_row
    merged_rows = list(url_map.values())
    df = pd.DataFrame(merged_rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_rows, f, ensure_ascii=False, indent=2)
    print(f"分类结果已更新：{csv_path}, {json_path}")
    return merged_rows


if __name__ == "__main__":
    # 测试
    model, processor = load_model()
    test_items = [{"save_path": "some_path.jpg", "download_status": "success", "image_url": "test"}]
    classified = classify_items_batch(model, processor, test_items)
    print(classified)