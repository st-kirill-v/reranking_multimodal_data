import subprocess
import time
import os

# ID всех папок из предыдущего вывода (вставь сюда полный список)
folder_ids = [
    "1kg4MuBAQYpPcJh7qObtBZX9FIDGN40hB",
    "1irAP5cJJX2tq9rk4P13mGehvMq01Y13W",
    "1NSiBVn3PFj1ZC1JcZzfMEgc-EsA3tFti",
    "1DGadPV9hQX3W-0PLLrnhdmrpy3YIBZ67",
    "1ddjy6lXYJA4VJbP-cjkWwVkqAGGKA76i",
    "1jFXSBgS3ZRGTjwudXVVNIKvER7E9mDEU",
    "1XgSd9fqL3Uku9AtOsijnO0UL6n7Ta2rT",
    "1ShrzVxhXxeK0HE5uCgBOek7sF9oC5DZo",
    "1I-BHQtxkctMfwCuEqiGo2Qwu5ZAqjPMB",
    "1mqtJNg1vyaqjGwY_8pNxf9YoKPTcHauQ",
    "1hz9lnPyTn_gA-o3XsxPi9NLRdGadKPvg",
    "1uFkP7JcJZ3wVyb46hl_ii8-SVOFxCxGs",
    "17ycHLmZL_7VxNsFc7QHJw0jKnZpFMzui",
    "1TrjQiGNrJGeHQZbb2XwaAPN3ZT5N_Xst",
    "1XaFCAiHuoCmzrvHsi4Wn4hmdC0OKknQR",
    "1aCTp4l40sJ8beeMG9HJk0TAXz5cRFOIE",
    "1XAzgSe0ERw0kKf4HMbIuLwGiVZax2Yjg",
    "1TJ7c_TcDJcUnb1yOMUjWIrcNI4unQsxV",
    "1DT8IQKVrpWhrXpxtMrM4FliB-ge1eNZy",
    "1-hD6x8OE-FGRwJZktWPNE5fvVQHXZ97j",
    "1owGgGxRicwq-f0GHr2nIVXE_kTqw0jVN",
    "1sbI1U-0lBEYBZBmqzL0o8hWKbPEPIxYO",
    "1mP9bT8hv2fzkTOyIr34yrCA2jkXmoQc2",
    "1tyfrCbvx6y1ddiKLAMYx7uCDoeXW0gWT",
    "1rPBVYkMSYuA-WnyzSg0m4m1IOLhRpZQD",
    "1ywxQo8k-tiKcMT1JFsYSgJhKVrqrD_Sx",
    "14QDQenJmin-mwyIgLj3qa6B3-kdQd17F",
    "1QWLMqasqvawyXH4FznTqVHjFgS293hph",
    "1EBC7sfK54TnZOh3Fpes__e5gN1o-qcds",
    "1ybS2oDMBrjF5XOGFD43nH_Z7ANl8wJ7B",
    "1oywb4Ff2hsn04yNZtLHf1D0_Ig-yNz63",
    "1BwIHnMZ8Hzx8IGtwDAVgaVYyldH07I7r",
    "1SlMp54GMOa21IYSC0xXTj_lsOZ0knNIw",
    "16lleRWBFG9ISpRLf9aE2glsPntkmkq55",
    "1bzO0Trs2RHPdZTCY9Yqvq5A0J9jJ5y_N",
    "1AZU1IH6uSUTPeGCD1Pyq2D3cJlUBRwKx",
    "1H7UJkr0fEHiwSejfypBufehUY1-2W-6q",
    "1na6LMWrv8NcMbJQRimNwNW0CffaQMzjF",
    "1XHMov-nj9eJBg0-UkMxJSnKkfUMjGRfY",
    "1NMfmRFSebueOFhoQhdRV83GPwDYVprbo",
    "1NUOn0rgFCICiWonQC5qFjZaAUfATzxI2",
    "18gP3CefatxAkLX_fKwI_ng1OD59S_uMq",
    "1-FUsHUorJzLIowCIUTdA1YvUgp1gT9UU",
    "1gi72ScZXe96mC3z5IUdh_AY1ViRebJga",
    "1882Q3QyDnE04b39bzGwP3r7dv_H9rtYN",
    "1Uk22tFR3tRy02TatbQfNpA0SOouMCLEa",
    "18_3B4CvA8TcilHapxYDSG1uQDdjK3Aj9",
    "1jOxBhvhTVnXNNH7X5qLwqIfWP6XsvE4G",
    "1qaibz-v3bZ5T_A1-_QrLeAThQwlz2B6D",
    "16I1xzLTD-xQYbPbsXwhLPA6N1taA-UI7",
]

# Создаём папку если нет
os.makedirs("data/datasets/docbench/data", exist_ok=True)

# Скачиваем по 50 папок с паузой
batch_size = 50
for i in range(0, len(folder_ids), batch_size):
    batch = folder_ids[i : i + batch_size]
    print(f"\n=== Downloading batch {i//batch_size + 1} (folders {i}-{i+len(batch)-1}) ===")

    for fid in batch:
        print(f"Downloading {fid}...")
        subprocess.run(
            [
                "gdown",
                "--folder",
                f"https://drive.google.com/drive/folders/{fid}",
                "-O",
                "data/datasets/docbench/data/",
            ],
            capture_output=True,
        )
        time.sleep(2)

    print(f"Batch {i//batch_size + 1} completed. Waiting 10 seconds...")
    time.sleep(10)
