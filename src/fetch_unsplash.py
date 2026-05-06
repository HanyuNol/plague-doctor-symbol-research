import os
import time
import requests
from tqdm import tqdm

from src.config import (
    UNSPLASH_ACCESS_KEY,   # 实际存放 Pexels API Key
    SEARCH_QUERY,
    PER_PAGE,
    RAW_DIR,
    META_DIR,
)
from src.utils import ensure_dir, save_json, safe_filename

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# 目标爬取总数
TARGET_TOTAL = 500
# 每页最大数量（Pexels 限制 80）
PER_PAGE = min(PER_PAGE, 80)


def search_photos(query: str, page: int, per_page: int):
    headers = {"Authorization": UNSPLASH_ACCESS_KEY}
    params = {"query": query, "page": page, "per_page": per_page}
    while True:
        try:
            resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                print("\n⚠️ 触发 Pexels 速率限制，等待 60 秒后重试...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"\n请求失败：{e}")
            if resp.status_code == 403:
                print("请检查 Pexels API Key 是否正确")
                raise
            time.sleep(5)


def download_image(url: str, save_path: str):
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    if not UNSPLASH_ACCESS_KEY:
        raise ValueError("请在 .env 文件中配置 UNSPLASH_ACCESS_KEY（实际为 Pexels API Key）")

    ensure_dir(RAW_DIR)
    ensure_dir(META_DIR)

    all_items = []
    total_downloaded = 0
    page = 1

    while total_downloaded < TARGET_TOTAL:
        print(f"\n开始抓取第 {page} 页（每页 {PER_PAGE} 张，已下载 {total_downloaded}/{TARGET_TOTAL}）")
        data = search_photos(SEARCH_QUERY, page, PER_PAGE)

        if not data or "photos" not in data:
            print("未获取到图片数据，提前结束")
            break

        photos = data["photos"]
        if not photos:
            print("当前页无图片，抓取结束")
            break

        for item in tqdm(photos, desc=f"第 {page} 页下载"):
            if total_downloaded >= TARGET_TOTAL:
                break

            photo_id = item.get("id")
            alt_desc = item.get("alt") or f"pexels_{photo_id}"
            filename = f"{photo_id}_{safe_filename(alt_desc)}.jpg"
            save_path = os.path.join(RAW_DIR, filename)
            image_url = item.get("src", {}).get("original")

            meta = {
                "photo_id": photo_id,
                "query": SEARCH_QUERY,
                "filename": filename,
                "save_path": save_path,
                "alt_description": item.get("alt"),
                "description": None,
                "width": item.get("width"),
                "height": item.get("height"),
                "color": item.get("avg_color"),
                "blur_hash": None,
                "created_at": None,
                "updated_at": None,
                "promoted_at": None,
                "user_name": item.get("photographer"),
                "user_username": None,
                "user_portfolio_url": item.get("photographer_url"),
                "unsplash_page": item.get("url"),
                "download_location": None,
                "image_url_regular": item.get("src", {}).get("large"),
                "image_url_small": item.get("src", {}).get("small"),
                "image_url_full": image_url,
                "source": "pexels",
            }

            try:
                if not os.path.exists(save_path):
                    download_image(image_url, save_path)
                meta["download_status"] = "success"
                total_downloaded += 1
            except Exception as e:
                meta["download_status"] = "failed"
                meta["download_error"] = str(e)

            all_items.append(meta)
            time.sleep(0.5)

        # 检查是否还有下一页
        total_results = data.get("total_results", 0)
        if page * PER_PAGE >= total_results:
            print("已到达最后一页。")
            break

        page += 1
        time.sleep(1)

    out_path = os.path.join(META_DIR, "unsplash_search_results.json")
    save_json(out_path, all_items)
    successful = sum(1 for it in all_items if it.get("download_status") == "success")
    print(f"\n抓取完成，共成功下载 {successful} / {len(all_items)} 张图片。")
    print(f"元数据文件：{out_path}")


if __name__ == "__main__":
    main()