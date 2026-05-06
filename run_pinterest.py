import subprocess
import sys
import os

os.makedirs("pinterest_data/raw", exist_ok=True)
os.makedirs("pinterest_data/meta", exist_ok=True)
os.makedirs("pinterest_data/outputs", exist_ok=True)

def run_step(module_name: str):
    print(f"\n===== Running: {module_name} =====")
    subprocess.run([sys.executable, "-m", module_name], check=True)

def main():
    run_step("src.stats_pinterest")

if __name__ == "__main__":
    main()