import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from joblib import Memory, Parallel, delayed
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

DATA_DIR = Path("./nyiso_data")
CACHE_DIR = DATA_DIR / "joblib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

memory = Memory(location=CACHE_DIR, verbose=0)

START_YM = "202001"
END_YM = datetime.now().strftime("%Y%m")
N_WORKERS = 4


def get_resilient_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        pool_connections=N_WORKERS, pool_maxsize=N_WORKERS, max_retries=retries
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Research Data Pipeline)"})
    return session


SESSION = get_resilient_session()


# Keep the exact name, signature, and body to preserve your 986 cached files
@memory.cache
def fetch_single_zip(url: str) -> bytes:
    """Downloads a single monthly zip archive and persists raw bytes to joblib cache."""
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    time.sleep(0.2)  # Polite pacing on fresh downloads
    return resp.content


def get_archive_urls(report_code: str) -> list[str]:
    index_url = f"https://mis.nyiso.com/public/{report_code}list.htm"
    resp = SESSION.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    zip_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Matches both YYYYMMDD*.zip and YYYYMM*.zip (used by P-27 DAM Bid archives)
        match = re.search(r"(\d{6})(?:\d{2})?.*\.zip$", href)
        if match and (START_YM <= match.group(1) <= END_YM):
            zip_links.append(urljoin(index_url, href))

    return sorted(set(zip_links))


def _download_worker(url: str) -> tuple[str, bool]:
    try:
        fetch_single_zip(url)
        return url, True
    except Exception as e:
        tqdm.write(f"[FAILED] {url} -> {e}")
        return url, False


def fetch_all_datasets(datasets: dict[str, str]) -> dict[str, list[str]]:
    failed_by_dataset: dict[str, list[str]] = {}
    for report_code, folder_name in datasets.items():
        urls = get_archive_urls(report_code)
        cached_count = sum(fetch_single_zip.check_call_in_cache(u) for u in urls)
        desc = f"{report_code} ({folder_name}) [{cached_count}/{len(urls)} cached]"

        parallel_gen = Parallel(
            n_jobs=N_WORKERS, prefer="threads", return_as="generator"
        )(delayed(_download_worker)(url) for url in urls)

        results = list(tqdm(parallel_gen, total=len(urls), desc=desc, unit="zip"))
        failed_by_dataset[report_code] = [url for url, ok in results if not ok]

    return failed_by_dataset


@memory.cache
def fetch_static_report(report_code: str) -> bytes:
    """Scrapes the CSV link from a static NYISO report page (e.g., P-19, P-23) and caches bytes."""
    index_url = f"https://mis.nyiso.com/public/{report_code}list.htm"
    resp = SESSION.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    csv_link = next(
        (urljoin(index_url, a["href"]) for a in soup.find_all("a", href=True) if a["href"].endswith(".csv")),
        None,
    )
    if csv_link is None:
        raise ValueError(f"No .csv link found on {index_url}")

    csv_resp = SESSION.get(csv_link, timeout=30)
    csv_resp.raise_for_status()
    return csv_resp.content
