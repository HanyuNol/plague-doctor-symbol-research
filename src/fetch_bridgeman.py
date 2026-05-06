import os
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

# Black Death 专用目录
BD_BASE_DIR = "blackdeath_data"
BD_RAW_DIR = os.path.join(BD_BASE_DIR, "raw")
BD_META_DIR = os.path.join(BD_BASE_DIR, "meta")

# 配置参数
SEARCH_QUERY = "Black Death"
BASE_URL = "https://bridgemaneducation.com/en/search"
MAX_PAGES = 100              # 最大页数限制（防止无限循环）
MAX_TOTAL_IMAGES = 500       # 目标总下载图片数
REQUEST_DELAY = 1.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://bridgemaneducation.com/",
    "Origin": "https://bridgemaneducation.com",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, data):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_filename(text: str, max_len: int = 100) -> str:
    import re
    if not text:
        text = "image"
    text = str(text)
    text = re.sub(r"[^\w\-_. ]", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("_") or "image"


def download_image_with_headers(url: str, save_path: str, timeout=30):
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        ext_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif"
        }
        correct_ext = ""
        for ctype, ext in ext_map.items():
            if ctype in content_type:
                correct_ext = ext
                break
        if correct_ext:
            base, _ = os.path.splitext(save_path)
            if not save_path.lower().endswith(correct_ext):
                save_path = base + correct_ext

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(save_path) == 0:
            os.remove(save_path)
            raise Exception("下载的文件为空")

        try:
            from PIL import Image
            img = Image.open(save_path)
            img.verify()
        except ImportError:
            pass
        except Exception as e:
            os.remove(save_path)
            raise Exception(f"图片文件损坏: {e}")

        return True, save_path
    except Exception as e:
        return False, str(e)


def get_soup(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def extract_items_from_page(soup, page_url: str):
    items = []
    img_tags = soup.select("img.asset-image-thumbnail")
    for img in img_tags:
        img_url = img.get("src") or img.get("data-src")
        if not img_url:
            continue
        img_url = urljoin(page_url, img_url)
        title = img.get("alt", "").strip()
        if not title:
            title = "black_death_image"
        parent_a = img.find_parent("a")
        detail_url = urljoin(page_url, parent_a.get("href")) if parent_a and parent_a.get("href") else ""
        items.append({
            "image_url": img_url,
            "title": title,
            "detail_url": detail_url,
        })
    return items


def get_next_page_url(soup, current_url: str):
    next_link = soup.find("a", rel="next")
    if next_link and next_link.get("href"):
        return urljoin(current_url, next_link["href"])
    for a in soup.find_all("a", href=True):
        if "next" in a.get_text(strip=True).lower():
            return urljoin(current_url, a["href"])
    return None


def main():
    ensure_dir(BD_RAW_DIR)
    ensure_dir(BD_META_DIR)

    all_items = []
    page_num = 1
    search_url = f"{BASE_URL}?filter_text={SEARCH_QUERY.replace(' ', '+')}"
    next_url = search_url

    total_downloaded = 0

    while page_num <= MAX_PAGES and next_url and total_downloaded < MAX_TOTAL_IMAGES:
        print(f"\n抓取第 {page_num} 页: {next_url}")
        try:
            soup = get_soup(next_url)
        except Exception as e:
            print(f"请求失败: {e}")
            break

        items = extract_items_from_page(soup, next_url)
        if not items:
            print("当前页无图片，停止抓取")
            break

        print(f"本页发现 {len(items)} 张图片")
        for item in tqdm(items, desc=f"第 {page_num} 页下载"):
            if total_downloaded >= MAX_TOTAL_IMAGES:
                break

            img_url = item["image_url"]
            title = item["title"]
            safe_title = safe_filename(title)
            temp_filename = f"bd_{page_num}_{safe_title}.jpg"
            temp_save_path = os.path.join(BD_RAW_DIR, temp_filename)

            meta_record = {
                "source_page": next_url,
                "page_num": page_num,
                "title": title,
                "image_url": img_url,
                "detail_url": item["detail_url"],
                "filename": temp_filename,
                "save_path": temp_save_path,
                "download_status": "pending",
            }

            success, result = download_image_with_headers(img_url, temp_save_path)
            if success:
                final_path = result
                meta_record["save_path"] = final_path
                meta_record["filename"] = os.path.basename(final_path)
                meta_record["download_status"] = "success"
                total_downloaded += 1
            else:
                meta_record["download_status"] = "failed"
                meta_record["error"] = result
                print(f"下载失败: {img_url} - {result}")

            all_items.append(meta_record)
            time.sleep(REQUEST_DELAY)

        if total_downloaded >= MAX_TOTAL_IMAGES:
            print(f"已达到目标数量 {MAX_TOTAL_IMAGES} 张，停止爬取。")
            break

        next_url = get_next_page_url(soup, next_url)
        page_num += 1
        time.sleep(REQUEST_DELAY)

    out_path = os.path.join(BD_META_DIR, "bridgeman_results.json")
    save_json(out_path, all_items)
    successful = sum(1 for it in all_items if it["download_status"] == "success")
    print(f"\n抓取完成，成功下载 {successful} / {len(all_items)} 张图片。")
    print(f"元数据文件：{out_path}")


if __name__ == "__main__":
    main()