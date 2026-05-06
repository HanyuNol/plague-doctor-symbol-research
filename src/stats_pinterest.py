import os
import json
import pandas as pd
from src.fetch_pinterest import search_pinterest_incremental, save_metadata
from src.classify_pinterest import load_model, classify_items_batch, update_classification_results
from src.config import (
    PINTEREST_META_DIR, PINTEREST_OUTPUT_DIR, PINTEREST_MAX_SCROLLS
)
from src.utils import ensure_dir

BD_QUERY = "black death"
PD_QUERY = "plague doctor"

def count_people_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if "has_people" not in df.columns:
        return 0
    return len(df[df["has_people"] == 1])

def count_plague_doctor_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if "has_people" not in df.columns or "has_plague_doctor" not in df.columns:
        return 0
    return len(df[(df["has_people"] == 1) & (df["has_plague_doctor"] == 1)])

def main():
    ensure_dir(PINTEREST_META_DIR)
    ensure_dir(PINTEREST_OUTPUT_DIR)

    # ---------- 1. 爬取并分类 black death ----------
    print("\n========== 阶段1：处理 'black death' ==========")
    bd_meta_path = os.path.join(PINTEREST_META_DIR, "black_death_meta.json")
    bd_csv_path = os.path.join(PINTEREST_OUTPUT_DIR, "black_death_classify.csv")
    bd_json_path = os.path.join(PINTEREST_OUTPUT_DIR, "black_death_classify.json")

    existing_urls = set()
    if os.path.exists(bd_meta_path):
        with open(bd_meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
            existing_urls = {item["image_url"] for item in existing_meta}

    print("开始爬取 black death 新图片...")
    new_bd_items = search_pinterest_incremental(BD_QUERY, existing_urls, max_scrolls=PINTEREST_MAX_SCROLLS)
    if new_bd_items:
        save_metadata(bd_meta_path, new_bd_items)
        model, processor = load_model()
        classified_new = classify_items_batch(model, processor, new_bd_items, batch_size=3)
        update_classification_results(bd_csv_path, bd_json_path, classified_new)
    else:
        print("无新图片，直接使用已有分类结果")

    N_bd_people = count_people_from_csv(bd_csv_path)
    bd_df = pd.read_csv(bd_csv_path, encoding='utf-8-sig')
    bd_people = bd_df[bd_df["has_people"] == 1]
    if len(bd_people) > 0:
        plague_in_bd = len(bd_people[bd_people["has_plague_doctor"] == 1])
        reaper_in_bd = len(bd_people[bd_people["has_grim_reaper"] == 1])
        print(f"Black Death 含人物图片总数: {N_bd_people}")
        print(f"其中鸟嘴医生出现次数: {plague_in_bd} ({plague_in_bd/len(bd_people):.2%})")
        print(f"死神形象出现次数: {reaper_in_bd} ({reaper_in_bd/len(bd_people):.2%})")
    else:
        print("Black Death 中未检测到含人物图片")
        return

    # ---------- 2. 爬取所有 plague doctor 图片（自然结束） ----------
    print("\n========== 阶段2：爬取所有 plague doctor 图片 ==========")
    pd_meta_path = os.path.join(PINTEREST_META_DIR, "plague_doctor_meta.json")
    pd_csv_path = os.path.join(PINTEREST_OUTPUT_DIR, "plague_doctor_classify.csv")
    pd_json_path = os.path.join(PINTEREST_OUTPUT_DIR, "plague_doctor_classify.json")

    existing_pd_urls = set()
    if os.path.exists(pd_meta_path):
        with open(pd_meta_path, "r", encoding="utf-8") as f:
            existing_pd_meta = json.load(f)
            existing_pd_urls = {item["image_url"] for item in existing_pd_meta}

    current_valid_pd = count_plague_doctor_from_csv(pd_csv_path)
    print(f"当前 plague doctor 有效鸟嘴医生数: {current_valid_pd}")

    model, processor = load_model()   # 复用模型

    scroll_round = 0
    # 爬取直到没有新图片
    while True:
        scroll_round += 1
        print(f"\n--- 第 {scroll_round} 轮爬取 plague doctor，当前有效 {current_valid_pd} 张 ---")
        # 每轮滚动 20 次（可根据需要调整）
        new_pd_items = search_pinterest_incremental(PD_QUERY, existing_pd_urls, max_scrolls=20)
        if not new_pd_items:
            print("未获取到新图片，停止爬取")
            break
        save_metadata(pd_meta_path, new_pd_items)
        classified_new = classify_items_batch(model, processor, new_pd_items, batch_size=3)
        update_classification_results(pd_csv_path, pd_json_path, classified_new)
        current_valid_pd = count_plague_doctor_from_csv(pd_csv_path)
        print(f"更新后有效鸟嘴医生数: {current_valid_pd}")

        # 安全上限：总成功下载超过 500 张停止
        with open(pd_meta_path, "r", encoding="utf-8") as f:
            pd_meta = json.load(f)
        total_pd_success = sum(1 for it in pd_meta if it.get("download_status") == "success")
        if total_pd_success > 500:
            print("已达到安全上限 500 张，停止爬取")
            break

    # ---------- 3. 最终统计对比 ----------
    print("\n========== 最终统计报告 ==========")
    print(f"Black Death 含人物图片总数: {N_bd_people}")
    print(f"其中鸟嘴医生出现次数: {plague_in_bd} (占有人物图片的 {plague_in_bd/len(bd_people):.2%})")
    print(f"Plague Doctor 有效鸟嘴医生图片数（直接搜索）: {current_valid_pd}")
    if plague_in_bd > 0:
        ratio = current_valid_pd / plague_in_bd
        print(f"后者是前者的 {ratio:.1f} 倍")
    else:
        print("Black Death 中无鸟嘴医生，无法计算倍数")

    # 保存最终统计 JSON
    final_stats = {
        "black_death": {
            "total_people_images": N_bd_people,
            "plague_doctor_count": int(plague_in_bd),
            "grim_reaper_count": int(reaper_in_bd),
        },
        "plague_doctor": {
            "valid_plague_doctor_images": int(current_valid_pd),
            "ratio_vs_blackdeath_plague": current_valid_pd / plague_in_bd if plague_in_bd else 0,
        }
    }
    stats_path = os.path.join(PINTEREST_OUTPUT_DIR, "pinterest_comparison_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存至: {stats_path}")

if __name__ == "__main__":
    main()