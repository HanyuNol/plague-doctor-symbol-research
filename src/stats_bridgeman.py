import os
import json
import pandas as pd
import numpy as np

# Black Death 专用输出目录
BD_OUTPUT_DIR = "blackdeath_data/outputs"

from src.utils import ensure_dir

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

def safe_convert(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_convert(i) for i in obj]
    return obj

def main():
    ensure_dir(BD_OUTPUT_DIR)
    csv_path = os.path.join(BD_OUTPUT_DIR, "bridgeman_classification.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到分类结果文件 {csv_path}，请先运行 src.classify_bridgeman")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if "has_people" not in df.columns:
        print("⚠️ 缺少 has_people 列，无法统计。")
        return

    df_people = df[df["has_people"] == 1].copy()
    total_people = len(df_people)
    if total_people == 0:
        print("没有检测到任何有人的图片，统计结束。")
        stats = {"total_people_images": 0, "has_plague_doctor": {"count": 0, "ratio": 0.0}, "top_tags": {}}
        with open(os.path.join(BD_OUTPUT_DIR, "bridgeman_stats.json"), "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return

    df_people['has_plague_doctor'] = pd.to_numeric(df_people['has_plague_doctor'], errors='coerce')
    valid = df_people[df_people['has_plague_doctor'].notna()]
    valid_count = len(valid)
    if valid_count > 0:
        plague_count = valid["has_plague_doctor"].sum()
        plague_ratio = plague_count / valid_count
    else:
        plague_count = 0
        plague_ratio = 0.0

    # 标签统计
    all_tags = []
    if "tags" in df_people.columns:
        for tags_str in df_people["tags"].dropna():
            try:
                if isinstance(tags_str, str):
                    tags_str = tags_str.strip()
                    if tags_str.startswith('[') and tags_str.endswith(']'):
                        import ast
                        tags = ast.literal_eval(tags_str)
                    else:
                        tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                else:
                    tags = tags_str
                if isinstance(tags, list):
                    all_tags.extend(tags)
            except Exception:
                pass
    tag_counts = pd.Series(all_tags).value_counts().head(20) if all_tags else pd.Series(dtype=int)

    # 打印报告
    print("=" * 60)
    print("Black Death 图片分析统计报告（仅统计有人的图片）")
    print("=" * 60)
    print(f"有人的图片总数: {total_people}")
    print(f"其中有效分类（鸟嘴医生字段有效）的图片数: {valid_count}")
    print(f"\n【核心结果】包含鸟嘴医生形象的图片数: {plague_count}")
    print(f"占有人的图片比例: {plague_ratio:.2%}")
    print(f"\n高频标签 Top 20:")
    if not tag_counts.empty:
        for tag, cnt in tag_counts.items():
            print(f"  {tag}: {cnt}")
    else:
        print("  无标签数据")

    # 生成饼图
    if HAS_PLT and valid_count > 0:
        try:
            plt.figure(figsize=(6, 6))
            plt.pie([plague_count, valid_count - plague_count],
                    labels=['有鸟嘴医生', '无鸟嘴医生'],
                    autopct='%1.1f%%', startangle=90)
            plt.title('有人图片中鸟嘴医生形象比例')
            chart_path = os.path.join(BD_OUTPUT_DIR, "bridgeman_plague_doctor_ratio.png")
            plt.savefig(chart_path, dpi=150)
            print(f"\n统计图表已保存: {chart_path}")
            plt.close()
        except Exception as e:
            print(f"生成图表失败: {e}")
    elif not HAS_PLT:
        print("\n未安装 matplotlib，跳过图表生成")

    # 保存统计JSON
    stats = {
        "total_people_images": int(total_people),
        "valid_classifications_among_people": int(valid_count),
        "has_plague_doctor": {
            "count": int(plague_count),
            "ratio": float(plague_ratio)
        },
        "top_tags_among_people": safe_convert(tag_counts.to_dict()) if not tag_counts.empty else {}
    }
    stats_path = os.path.join(BD_OUTPUT_DIR, "bridgeman_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n详细统计已保存: {stats_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()