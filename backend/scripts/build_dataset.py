import time
import json
import os
from tqdm import tqdm
from crawler.url_discovery import get_sitemap_urls
from crawler.article_extractor import fetch_and_extract
from crawler.validate_and_save import to_record, append_jsonl

CRAWL_DELAY = 1.0  # polite delay

TOPIC_KEYWORDS = {
    "sports": ["sport", "match", "tournament", "goal", "cricket", "football", "বিশ্বকাপ", "ম্যাচ", "খেলা"],
    "politics": ["election", "minister", "parliament", "government", "নির্বাচন", "মন্ত্রী", "সরকার"],
    "business": ["bank", "market", "economy", "inflation", "টাকা", "বাজার", "অর্থনীতি"],
    "entertainment": ["film", "music", "celebrity", "নাটক", "সিনেমা", "গান"],
}

GENERIC_URL_TOPICS = {"news", "article", "story", "bangladesh", "national", "latest", "other"}


SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".avi", ".mkv",
    ".pdf", ".zip", ".rar",
)
SKIP_SUBSTRINGS = (
    "/photo/", "/photos/",
    "/video/", "/videos/",
    "/gallery/", "/galleries/",
    "/tag/", "/topic/",
    "/author/", "/authors/",
    "/search/",
)

def should_skip_url(url: str) -> bool:
    u = url.lower().split("?", 1)[0] 

    if "dailynewnation.com" in u and "bangla" in u:
        return True

    if u.endswith(SKIP_EXTENSIONS):
        return True
    for s in SKIP_SUBSTRINGS:
        if s in u:
            return True
    return False



def load_existing_urls(jsonl_path: str) -> set:
    """Load URLs already saved in a jsonl file."""
    urls = set()
    if not os.path.exists(jsonl_path):
        return urls

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                urls.add(json.loads(line)["url"])
            except Exception:
                continue
    return urls


def topic_from_url(url: str) -> str:
    parts = url.split("/")
    topic = parts[3].lower() if len(parts) > 3 else "other"

    
    if topic in ("news", "article", "story", "bangladesh"):
        if len(parts) > 4 and parts[4]:
            return parts[4].lower()

    return topic


def detect_topic(title: str, body: str, url: str) -> str:
    """
    Hybrid topic detection:
    1) Use URL-derived topic if it looks meaningful.
    2) Otherwise fallback to keyword matching on title+body.
    """
    url_topic = topic_from_url(url)
    if url_topic not in GENERIC_URL_TOPICS and len(url_topic) > 2:
        return url_topic

    text = f"{title} {body}".lower()

    best_topic = "other"
    best_score = 0
    for topic, kws in TOPIC_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw.lower() in text:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic

def to_record_force_en(doc):
    """
    Force an English record for Daily New Nation, bypassing language checks.
    Keeps a minimal sanity check to avoid empty/junk pages.
    """
    title = (doc.get("title") or "").strip()
    body = (doc.get("body") or "").strip()
    url = doc.get("url")

    if not url or not body or len(body) < 200:
        return None

    return {
        "title": title,
        "body": body,
        "language": "en",
        "url": url,
     }



def build_for_site(site_base: str, out_path: str, expected_language: str, limit: int):
    urls = get_sitemap_urls(site_base)

    kept = 0
    seen = set()
    existing_urls = load_existing_urls(out_path)
    seen.update(existing_urls)

    for url in tqdm(urls, desc=f"{site_base}"):
        if kept >= limit:
            break
        if url in seen:
            continue
        if should_skip_url(url):
            continue
        seen.add(url)

        try:
            doc = fetch_and_extract(url)
            if not doc:
                continue

            if "dailynewnation.com" in url:
                rec = to_record_force_en(doc)
            else:
                rec = to_record(doc, expected_language)

            if not rec:
                time.sleep(CRAWL_DELAY)
                continue

            append_jsonl(out_path, rec)
            kept += 1
            time.sleep(CRAWL_DELAY)
        except Exception:
            continue

    return kept

def main():
    # You should load these from yaml later; hardcoding is OK for first run.
    bn_sites = [
        "https://bangla.bdnews24.com",
        # "https://www.prothomalo.com",
        # "https://banglatribune.com",
        # "https://www.dhakapost.com",
    ]
    en_sites = [
        # "https://www.dhakatribune.com",
        # "https://www.thedailystar.net",
        "https://www.newagebd.net",
        "https://www.banglanews24.com",
        "https://www.dailynewnation.com",
        "https://www.daily-sun.com",
        "https://www.dhakatribune.com",
    ]

    bn_out = "data/processed/bn.jsonl"
    en_out = "data/processed/en.jsonl"

    bn_target = 2500
    en_target = 2500

    # Distribute quota across 5 sites (500 each). You can adjust dynamically.
    quota_per_site = 10

    total_bn = 0
    for s in bn_sites:
        total_bn += build_for_site(s, bn_out, "bn", quota_per_site)
        if total_bn >= bn_target:
            break

    total_en = 0
    for s in en_sites:
        total_en += build_for_site(s, en_out, "en", quota_per_site)
        if total_en >= en_target:
            break

    print("DONE")
    print("Bangla docs:", total_bn)
    print("English docs:", total_en)

if __name__ == "__main__":
    main()

