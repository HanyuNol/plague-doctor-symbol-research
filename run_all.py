import subprocess
import sys


def run_step(module_name: str):
    print(f"\n===== Running: {module_name} =====")
    subprocess.run([sys.executable, "-m", module_name], check=True)


def main():
    run_step("src.fetch_unsplash")
    run_step("src.classify_images")
    run_step("src.stats_report")
    print("\n全部流程执行完成。")


if __name__ == "__main__":
    main()