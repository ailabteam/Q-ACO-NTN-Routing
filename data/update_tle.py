# File: data/update_tle.py (Đã cập nhật để dùng API QUERY)

import requests
import os
import sys

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.network_model import TLE_PATH

# URL mới sử dụng GP Query cho nhóm STARLINK
STARLINK_QUERY_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=STARLINK&FORMAT=TLE"
OUTPUT_PATH = 'data/starlink.txt' 

def download_latest_tle():
    print(f"Đang tải TLE Starlink mới nhất từ API: {STARLINK_QUERY_URL}")
    try:
        response = requests.get(STARLINK_QUERY_URL, timeout=30)
        response.raise_for_status() # Bắn lỗi cho status code 4xx/5xx

        # Nội dung trả về là file TLE
        tle_data = response.text

        # Đếm số lượng vệ tinh (mỗi vệ tinh 3 dòng: Tên, Dòng 1, Dòng 2)
        # Loại bỏ các dòng trống nếu có
        tle_lines = [line for line in tle_data.strip().split('\n') if line.strip()]
        num_sats = len(tle_lines) // 3

        if num_sats == 0:
            print("Cảnh báo: Tải thành công nhưng không tìm thấy TLE nào trong dữ liệu.")
            return 0

        with open(OUTPUT_PATH, 'w') as f:
            f.write(tle_data)

        print(f"Tải thành công! Đã lưu {num_sats} vệ tinh vào {OUTPUT_PATH}")
        return num_sats

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải file TLE: {e}")
        return 0

if __name__ == "__main__":

    if not os.path.exists('data'):
        os.makedirs('data')

    download_latest_tle()
