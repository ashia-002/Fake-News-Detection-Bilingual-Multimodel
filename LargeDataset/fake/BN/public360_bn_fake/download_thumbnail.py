import json
import os
import requests

# === Step 1: Load video IDs from yt-dlp output ===
file_path = r"C:\Users\farha\Downloads\public360_bn_fake\output.json"  # update path to your JSON file

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


# === Step 2: Download thumbnails ===
def download_thumbnails(video_ids, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Saving thumbnails to: {os.path.abspath(output_folder)}\n")

    for index, vid in enumerate(video_ids, start=1):
        filename = os.path.join(output_folder, f"{vid}.jpg")

        # Skip if already downloaded
        if os.path.exists(filename):
            print(f"[{index}/{len(video_ids)}] Skipped (exists): {vid}")
            continue

        # Try max resolution first
        url = f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg"
        response = requests.get(url)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"[{index}/{len(video_ids)}] ✅ Downloaded (maxres): {vid}")
        else:
            # fallback to HQ thumbnail
            fallback_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
            response = requests.get(fallback_url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"[{index}/{len(video_ids)}] ⚠️ Downloaded (hq): {vid}")
            else:
                print(f"[{index}/{len(video_ids)}] ❌ Failed for: {vid}")


# === Step 3: Run downloader ===
output_folder = r"C:\Users\farha\Downloads\public360_bn_fake"  # update this path
download_thumbnails(video_ids, output_folder)
