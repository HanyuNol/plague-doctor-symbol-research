import os
from dotenv import load_dotenv

load_dotenv()

# Pexels API (复用原来的变量名)
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()  # 实际存放 Pexels Key
SEARCH_QUERY = os.getenv("SEARCH_QUERY", "covid 19").strip()
PER_PAGE = int(os.getenv("PER_PAGE", "30"))      # Pexels 最大支持 80
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))     # 测试用，可改大

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct").strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

RAW_DIR = "data/raw"
META_DIR = "data/meta"
OUTPUT_DIR = "data/outputs"
LOG_DIR = "data/logs"
# ========== Pinterest 爬虫配置 ==========
PINTEREST_DATA_DIR = "pinterest_data"
PINTEREST_RAW_DIR = os.path.join(PINTEREST_DATA_DIR, "raw")
PINTEREST_META_DIR = os.path.join(PINTEREST_DATA_DIR, "meta")
PINTEREST_OUTPUT_DIR = os.path.join(PINTEREST_DATA_DIR, "outputs")

# 爬虫参数
PINTEREST_SCROLL_PAUSE = 2.0          # 滚动等待时间（秒）
PINTEREST_DOWNLOAD_DELAY = 0.5        # 下载间隔
PINTEREST_MAX_SCROLLS = 100            # 每个关键词最大滚动次数（约600-1000张）