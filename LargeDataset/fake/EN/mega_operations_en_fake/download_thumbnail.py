import json
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === Step 1: Load video IDs from yt-dlp output ===
file_path = r"C:\Users\farha\Downloads\en_fake_images\mega_operations_en_fake\output.json"  # update path to your JSON file

# Try UTF-8 first, fallback to UTF-16 if needed
data = []
try:
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Line {i}] JSON decode error: {e}")
except UnicodeError:
    with open(file_path, encoding="utf-16") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Line {i}] JSON decode error: {e}")

# Extract video IDs
video_ids = [item["id"] for item in data if "id" in item]
print(f"✅ Loaded {len(video_ids)} video IDs\n")


# === Step 2: Download thumbnails with retries & resume support ===
def download_thumbnails(video_ids, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Saving thumbnails to: {os.path.abspath(output_folder)}\n")

    # Create a requests session with retries
    session = requests.Session()
    retries = Retry(
        total=5,                # retry up to 5 times
        backoff_factor=1,       # wait 1s, 2s, 4s, 8s, 16s...
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {"User-Agent": "Mozilla/5.0"}

    for index, vid in enumerate(video_ids, start=1):
        filename = os.path.join(output_folder, f"{vid}.jpg")

        # Skip if already downloaded
        if os.path.exists(filename):
            print(f"[{index}/{len(video_ids)}] Skipped (exists): {vid}")
            continue

        # Try both maxres and HQ fallback
        for quality in ["maxresdefault", "hqdefault"]:
            url = f"https://img.youtube.com/vi/{vid}/{quality}.jpg"
            try:
                response = session.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"[{index}/{len(video_ids)}] ✅ Downloaded ({quality}): {vid}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"[{index}/{len(video_ids)}] ⚠️ Network error: {e}")
                time.sleep(2)  # brief pause before retry
        else:
            print(f"[{index}/{len(video_ids)}] ❌ Failed for: {vid}")

        # small delay to avoid being blocked by YouTube
        time.sleep(0.3)


# === Step 3: Run downloader ===
output_folder = r"C:\Users\farha\Downloads\en_fake_images\mega_operations_en_fake"  # update this path
download_thumbnails(video_ids, output_folder)
