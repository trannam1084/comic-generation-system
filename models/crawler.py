import time
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_soup(driver, wait, url):
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    return BeautifulSoup(driver.page_source, "lxml")

def scrape_story(driver, wait, story_url, max_retries=3):
    for attempt in range(max_retries):
        try:
            soup = get_soup(driver, wait, story_url)
            article = soup.find("article")

            for tag in article.select("ins.adsbygoogle, script, style, sup"):
                tag.decompose()

            title = article.find("h1").get_text(" ", strip=True)
            author_source = None
            content_lines = []

            for p in article.select("p"):
                text = p.get_text(" ", strip=True)
                if "bg-des" in p.get("class", []):
                    continue
                if "text-align: right" in p.get("style", ""):
                    author_source = text
                    break
                content_lines.append(text)

            return {
                "url": story_url,
                "title": title,
                "content": "\n".join(content_lines).strip(),
                "author_or_source": author_source
            }
        except Exception:
            time.sleep(1 + attempt)
    return None

def run_crawler():
    BASE_URL = "https://truyendangian.com"
    START_PAGE = 1
    END_PAGE = 16
    TIMEOUT = 15
    # Sua doi OUTPUT_CSV thanh path noi luu file CSV
    OUTPUT_CSV = "/media/nam/DATA/RAI_K63/HK9/NLP/comic-generation/data/full_story.csv"

    print("Đang khởi tạo trình duyệt Chrome trên Kaggle...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-extensions")

    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    wait = WebDriverWait(driver, TIMEOUT)

    story_links = []
    seen = set()

    print("Giai đoạn 1: Lấy URL truyện...")
    for page_idx in tqdm(range(START_PAGE, END_PAGE + 1), desc="Thu thập Links"):
        page_url = BASE_URL if page_idx == 1 else f"{BASE_URL}/page/{page_idx}/"
        soup = get_soup(driver, wait, page_url)

        for a in soup.select("h2 a[href]"):
            title = a.get_text(" ", strip=True)
            url = a["href"].strip().rstrip("/")
            if not title or not url or "bài thơ" in title.lower() or url in seen:
                continue
            seen.add(url)
            story_links.append({"title": title, "url": url})

    print("Giai đoạn 2: Trích xuất nội dung...")
    datasets = []
    for item in tqdm(story_links, desc="Cào dữ liệu chi tiết"):
        result = scrape_story(driver, wait, item["url"])
        if result:
            datasets.append(result)

    df = pd.DataFrame(datasets)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Hoàn tất! Đã lưu dữ liệu tại: {OUTPUT_CSV}")
    driver.quit()

if __name__ == "__main__":
    run_crawler()
