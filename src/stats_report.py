import os
import json
import pandas as pd
import numpy as np
from src.config import OUTPUT_DIR
from src.utils import ensure_dir

try:
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def safe_convert_to_int(obj):
    """将 pandas 整数类型转换为 Python int，便于 JSON 序列化"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: safe_convert_to_int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_convert_to_int(i) for i in obj]
    return obj


def main():
    ensure_dir(OUTPUT_DIR)

    # 读取分类结果
    csv_path = os.path.join(OUTPUT_DIR, "image_classification_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到分类结果文件 {csv_path}，请先运行 src.classify_images")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 确保 has_people 列存在，若没有则尝试从 people_count_range 推断（兼容旧数据）
    if "has_people" not in df.columns:
        if "people_count_range" in df.columns:
            df["has_people"] = df["people_count_range"].apply(
                lambda x: 0 if pd.isna(x) or str(x).strip() in ["", "unknown"] else 1
            )
        else:
            # 无法判断，默认所有图片都有人（保守处理）
            df["has_people"] = 1

    # 只保留有人的图片
    df_people = df[df["has_people"] == 1].copy()
    total_people_images = len(df_people)

    if total_people_images == 0:
        print("=" * 60)
        print("图片分类统计报告")
        print("=" * 60)
        print("警告：没有检测到任何有人的图片，无法计算防护服占比。")
        print("请检查分类结果或原始数据。")
        print("=" * 60)
        # 保存空的统计结果
        stats = {
            "total_people_images": 0,
            "has_protective_medical_staff": {"count": 0, "ratio": 0.0},
            "top_tags": {}
        }
        stats_path = os.path.join(OUTPUT_DIR, "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return

    # 转换防护服字段为数值
    df_people['has_protective_medical_staff'] = pd.to_numeric(
        df_people['has_protective_medical_staff'], errors='coerce'
    )
    valid_people = df_people[df_people["has_protective_medical_staff"].notna()]
    valid_count = len(valid_people)

    # 错误计数（仅针对有人图片中的错误）
    error_count = 0
    if "error" in df_people.columns:
        error_mask = df_people["error"].notna() & (df_people["error"].astype(str).str.strip() != "")
        error_count = error_mask.sum()

    # 防护服统计
    if valid_count > 0:
        has_ppe = valid_people["has_protective_medical_staff"].sum()
        ppe_ratio = has_ppe / valid_count
    else:
        has_ppe = 0
        ppe_ratio = 0.0

    # 标签统计（仅基于有人图片）
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
    print("图片分类统计报告（仅统计有人的图片）")
    print("=" * 60)
    print(f"有人的图片总数: {total_people_images}")
    print(f"其中成功分类（防护服字段有效）的图片数: {valid_count}")
    print(f"分类出错的图片数: {error_count}")
    print(f"\n【核心结果】包含穿防护服医务人员的图片数: {int(has_ppe)}")
    print(f"占有人图片的比例: {ppe_ratio:.2%}")

    print(f"\n高频标签 Top 20（仅限有人图片）:")
    if not tag_counts.empty:
        for tag, cnt in tag_counts.items():
            print(f"  {tag}: {cnt}")
    else:
        print("  无标签数据")

    # 生成图表（仅防护服占比饼图）
    if HAS_PLT and valid_count > 0 and has_ppe is not None and not np.isnan(has_ppe):
        try:
            plt.figure(figsize=(6, 6))
            ppe_counts = [float(has_ppe), float(valid_count - has_ppe)]
            plt.pie(ppe_counts, labels=['有防护服', '无防护服'], autopct='%1.1f%%', startangle=90)
            plt.title('有人图片中穿防护服医务人员比例')
            chart_path = os.path.join(OUTPUT_DIR, "ppe_ratio_chart.png")
            plt.savefig(chart_path, dpi=150)
            print(f"\n统计图表已保存: {chart_path}")
            plt.close()
        except Exception as e:
            print(f"生成图表失败: {e}")
    elif not HAS_PLT:
        print("\n未安装 matplotlib，跳过图表生成")

    # 保存统计 JSON（只保存核心信息）
    stats = {
        "total_people_images": int(total_people_images),
        "successful_classifications_among_people": int(valid_count),
        "error_count_among_people": int(error_count),
        "has_protective_medical_staff_among_people": {
            "count": int(has_ppe) if not np.isnan(has_ppe) else 0,
            "ratio": float(ppe_ratio) if not np.isnan(ppe_ratio) else 0.0
        },
        "top_tags_among_people": safe_convert_to_int(tag_counts.to_dict()) if not tag_counts.empty else {}
    }

    stats_path = os.path.join(OUTPUT_DIR, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n详细统计已保存: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()