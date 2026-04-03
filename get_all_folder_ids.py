import requests
import time
import re

# ID корневой папки
FOLDER_ID = "1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr"


def get_folder_contents(folder_id):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers)
    html = response.text

    # Ищем ID папок
    ids = re.findall(r'data-id="([^"]+)"', html)
    names = re.findall(r'data-title="([^"]+)"', html)

    result = []
    for i, fid in enumerate(ids):
        name = names[i] if i < len(names) else "unknown"
        if len(fid) > 20 and not fid.startswith("_"):
            result.append((name, fid))

    return result


print(f"Getting contents of folder {FOLDER_ID}...")
folders = get_folder_contents(FOLDER_ID)

print(f"Found {len(folders)} folders:")
for name, fid in folders:
    print(f"{name}: {fid}")
