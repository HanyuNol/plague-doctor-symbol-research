import os
import time
import json
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

from src.config import (
    PINTEREST_RAW_DIR, PINTEREST_META_DIR,
    PINTEREST_SCROLL_PAUSE, PINTEREST_DOWNLOAD_DELAY, PINTEREST_MAX_SCROLLS
)
from src.utils import ensure_dir, safe_filename

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.pinterest.com/",
}


def init_driver():
    """初始化 Edge 浏览器（无头模式）"""
    edge_options = Options()
    edge_options.add_argument("--headless")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--window-size=1920,1080")
    edge_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    try:
        driver = webdriver.Edge(options=edge_options)
    except Exception as e:
        print(f"无法启动 EdgeDriver: {e}")
        print("请确保已安装 Edge 浏览器，并将 EdgeDriver 所在目录添加到 PATH 环境变量中。")
        print("下载地址: https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/")
        raise
    return driver


def extract_image_urls_from_page(driver):
    """提取当前页面所有图片 URL 和 alt"""
    images = driver.find_elements(By.CSS_SELECTOR, "img[src*='pinimg.com']")
    results = []
    for img in images:
        try:
            src = img.get_attribute("src")
            alt = img.get_attribute("alt") or ""
            if src and "pinimg.com" in src:
                if "/564x/" in src:
                    src = src.replace("/564x/", "/originals/")
                results.append({"url": src, "title": alt[:100]})
        except:
            continue
    return results


def scroll_to_load(driver, pause_time, max_attempts=5):
    """
    滚动到底部并等待新内容加载
    返回是否成功加载到新内容
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(pause_time)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height > last_height:
        return True
    else:
        # 尝试多次滚动（有时需要多次触发）
        for _ in range(max_attempts):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height > last_height:
                return True
        return False


def download_image(url, save_path):
    """下载图片（带重试）"""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=30, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"下载失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(2)
    return False


def search_pinterest_incremental(query, existing_urls_set, max_scrolls=None):
    """
    增量爬取：滚动加载直到 max_scrolls 次或无法加载新图片。
    """
    if max_scrolls is None:
        max_scrolls = PINTEREST_MAX_SCROLLS

    driver = init_driver()
    search_url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
    print(f"正在打开: {search_url}")
    driver.get(search_url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img[src*='pinimg.com']"))
        )
    except Exception as e:
        print(f"页面加载超时: {e}")
        driver.quit()
        return []

    all_new_items = []
    scroll_count = 0
    last_item_count = 0

    while scroll_count < max_scrolls:
        # 滚动并等待新内容
        has_new_content = scroll_to_load(driver, PINTEREST_SCROLL_PAUSE)
        scroll_count += 1
        print(f"  滚动第 {scroll_count} 次，已收集 {len(all_new_items)} 张新图片")

        # 提取当前页面的所有图片
        page_items = extract_image_urls_from_page(driver)
        new_items = []
        for item in page_items:
            if item["url"] not in existing_urls_set:
                new_items.append(item)
                existing_urls_set.add(item["url"])

        if not new_items:
            # 没有新图片，但如果之前有图片且滚动后没有新内容，可能到底了
            if not has_new_content and scroll_count > 2:
                print("  滚动后无新内容且无新图片，结束")
                break
            else:
                # 可能加载慢，再试一次
                time.sleep(2)
                continue

        # 下载新图片
        for item in tqdm(new_items, desc=f"下载 {query} 新图片"):
            img_url = item["url"]
            title = item["title"] or f"pinterest_{query}_{len(all_new_items)+1}"
            safe_title = safe_filename(title)
            filename = f"{query.replace(' ', '_')}_{int(time.time())}_{safe_title}.jpg"
            save_path = os.path.join(PINTEREST_RAW_DIR, filename)

            if download_image(img_url, save_path):
                all_new_items.append({
                    "search_query": query,
                    "title": title,
                    "image_url": img_url,
                    "filename": filename,
                    "save_path": save_path,
                    "download_status": "success"
                })
            else:
                all_new_items.append({
                    "search_query": query,
                    "title": title,
                    "image_url": img_url,
                    "filename": filename,
                    "save_path": save_path,
                    "download_status": "failed",
                    "error": "下载失败"
                })
            time.sleep(PINTEREST_DOWNLOAD_DELAY)

        # 如果本次新图片很少且已经滚动多次，可能到底
        if len(new_items) < 5 and scroll_count > 5:
            print("  新图片很少，可能已到底")
            # 再尝试滚动一次
            if not scroll_to_load(driver, PINTEREST_SCROLL_PAUSE):
                break

        # 检查图片总数是否不再增长
        current_count = len(driver.find_elements(By.CSS_SELECTOR, "img[src*='pinimg.com']"))
        if current_count == last_item_count and scroll_count > 3:
            print("  图片数量未增加，结束")
            break
        last_item_count = current_count

    driver.quit()
    print(f"关键词 '{query}' 本次新增 {len(all_new_items)} 张图片（成功下载 {sum(1 for i in all_new_items if i['download_status']=='success')}）")
    return all_new_items


def save_metadata(meta_path, all_items):
    """保存或追加元数据到 JSON 文件"""
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    url_map = {item["image_url"]: item for item in existing}
    for new_item in all_items:
        url_map[new_item["image_url"]] = new_item
    merged = list(url_map.values())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


if __name__ == "__main__":
    # 测试
    test_set = set()
    new = search_pinterest_incremental("black death", test_set, max_scrolls=5)
    save_metadata("test_meta.json", new)