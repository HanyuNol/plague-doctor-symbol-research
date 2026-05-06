import subprocess
import sys
import os

# 创建 Black Death 专用数据目录（如果不存在）
os.makedirs("blackdeath_data/raw", exist_ok=True)
os.makedirs("blackdeath_data/meta", exist_ok=True)
os.makedirs("blackdeath_data/outputs", exist_ok=True)

def run_step(module_name: str):
    print(f"\n===== Running: {module_name} =====")
    subprocess.run([sys.executable, "-m", module_name], check=True)

def main():
    run_step("src.fetch_bridgeman")
    run_step("src.classify_bridgeman")
    run_step("src.stats_bridgeman")
    print("\nBlack Death 图片抓取与分析全部完成！结果保存在 blackdeath_data/ 目录下。")

if __name__ == "__main__":
    main()