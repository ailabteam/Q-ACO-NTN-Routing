# File: analysis/tle_analyzer.py

import numpy as np
from skyfield.api import load
import os
import sys

# FIX: Thêm thư mục gốc vào PYTHONPATH (Đã sửa lỗi import)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.network_model import TLE_PATH, NUM_SATS 

def analyze_tle_file():
    
    # ... (Kiểm tra file và tải TLE giữ nguyên) ...
    print(f"--- Phân tích File TLE: {TLE_PATH} ---")
    if not os.path.exists(TLE_PATH):
        print(f"Lỗi: File TLE không tìm thấy tại {TLE_PATH}")
        return

    try:
        ts = load.timescale()
        full_sats = load.tle_file(TLE_PATH)
        sats = full_sats[:NUM_SATS]
    except Exception as e:
        print(f"Lỗi khi tải TLE: {e}")
        return

    if not sats:
        print("Không có vệ tinh nào được load.")
        return

    print(f"Tổng số vệ tinh được load (Top {NUM_SATS}): {len(sats)}")
    
    # Danh sách để thu thập dữ liệu
    inclinations = []
    mean_motions = []
    rascensions = [] # Thêm rascensions
    
    for sat in sats:
        model = sat.model 
        
        # 1. Inclination (Độ nghiêng) - nằm trong model.inclo (radians)
        # Sử dụng numpy.degrees để chuyển đổi
        inclinations.append(np.degrees(model.inclo))
        
        # 2. Mean Motion (Số vòng/ngày)
        mean_motions.append(model.no_kozai)
        
        # 3. Right Ascension of Ascending Node (RAAN)
        rascensions.append(np.degrees(model.nodeo))

    # Chuyển sang mảng NumPy để tính toán thống kê
    inclinations_np = np.array(inclinations)
    mean_motions_np = np.array(mean_motions)
    rascensions_np = np.array(rascensions) # Thêm rascensions_np

    # --- Thống kê Độ nghiêng ---
    print("\n[Thống kê Độ Nghiêng Quỹ Đạo (Inclination)]")
    # ... (giữ nguyên logic thống kê) ...
    print(f"  Trung bình: {np.mean(inclinations_np):.2f} degrees")
    print(f"  Độ lệch chuẩn: {np.std(inclinations_np):.2f} degrees")
    print(f"  Phạm vi (Min/Max): {np.min(inclinations_np):.2f} / {np.max(inclinations_np):.2f} degrees")
    
    # Thống kê phân bố độ nghiêng
    unique_inclinations, counts = np.unique(np.round(inclinations_np, 1), return_counts=True)
    print("\n  Phân bố Độ nghiêng chính:")
    for inc, count in zip(unique_inclinations, counts):
        if count > 5: # Chỉ liệt kê các nhóm lớn
             print(f"    - {inc:.1f} degrees: {count} vệ tinh")

    # --- Thống kê Mean Motion (Độ cao) ---
    print("\n[Thống kê Mean Motion (Số vòng/ngày)]")
    print(f"  Trung bình: {np.mean(mean_motions_np):.2f} revs/day")
    if np.mean(mean_motions_np) < 16.0:
        print("  => Độ cao LEO (phần lớn ~550km - 1000km).")
    
    # --- Phân tích Mặt phẳng Quỹ đạo (Right Ascension) ---
    print("\n[Phân tích Mặt phẳng Quỹ đạo (Right Ascension - RAAN)]")
    print(f"  Trung bình: {np.mean(rascensions_np):.2f} degrees")
    print(f"  Phạm vi (Min/Max): {np.min(rascensions_np):.2f} / {np.max(rascensions_np):.2f} degrees")
    print(f"  Độ lệch chuẩn RAAN: {np.std(rascensions_np):.2f} degrees")
    
    if np.std(rascensions_np) < 10.0:
        print("  => CẢNH BÁO: Các vệ tinh có xu hướng tập trung ở một vài mặt phẳng quỹ đạo hẹp.")
    elif np.std(rascensions_np) > 100.0:
        print("  => TỐT: Các vệ tinh có độ phân bố RAAN rộng, khả năng tạo lưới cao.")

if __name__ == "__main__":
    
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
        
    analyze_tle_file()
