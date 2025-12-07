# File: analysis/network_analyzer.py (ĐÃ SỬA LỖI IMPORT)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import sys

# FIX LỖI IMPORT: Đảm bảo thư mục gốc nằm trong PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các thành phần từ mô hình Synthetic mới
from simulation.network_model import (
    DynamicLEONetwork, 
    SIM_DURATION_SECONDS, 
    TIME_SLOT_SIZE,
    NUM_SATS,
    NUM_GROUND_STATIONS
)

def analyze_network_dynamics():
    """Chạy mô phỏng dài hơi để phân tích các chỉ số mạng."""
    
    print(f"Bắt đầu phân tích động lực học mạng Synthetic...")
    
    # Khởi tạo network không cần tham số
    network = DynamicLEONetwork()

    time_slots = np.arange(0, SIM_DURATION_SECONDS, TIME_SLOT_SIZE)
    
    link_counts = []
    path_delays = []
    path_hops = []
    connectivity_status = []
    
    source = network.gs_names[2] # London (GS_2)
    destination = network.gs_names[3] # Tokyo (GS_3)

    print(f"\n--- Phân tích Động lực học Mạng Synthetic ({SIM_DURATION_SECONDS}s) ---")
    
    for t_sec in time_slots:
        G_t = network.get_network_graph(t_sec)
        
        # A. Thu thập số lượng liên kết
        link_counts.append(G_t.number_of_edges())
        
        # B. Thu thập trạng thái kết nối
        try:
             is_connected = nx.is_connected(G_t)
        except nx.NetworkXPointlessConcept:
             # Xử lý trường hợp đồ thị quá thưa
             is_connected = False
             
        connectivity_status.append(is_connected)

        # C. Thu thập thông số đường đi ngắn nhất
        if nx.has_path(G_t, source, destination):
            shortest_path = nx.shortest_path(G_t, source=source, target=destination, weight='weight')
            delay = nx.shortest_path_length(G_t, source=source, target=destination, weight='weight')
            path_delays.append(delay)
            path_hops.append(len(shortest_path) - 1)
        else:
            path_delays.append(np.nan)
            path_hops.append(np.nan)
            
        if t_sec % 60 == 0:
            print(f"Time t={t_sec:.0f}s: Links={G_t.number_of_edges()}, Connected={is_connected}")

    # --- Kết quả Phân tích ---
    
    avg_links = np.mean(link_counts)
    conn_perc = np.mean(connectivity_status) * 100
    
    print("\n--- Tóm tắt Thống kê ---")
    print(f"Số lượng vệ tinh mô phỏng: {NUM_SATS}")
    print(f"Thời gian mô phỏng: {SIM_DURATION_SECONDS} giây")
    print(f"Số liên kết trung bình: {avg_links:.2f}")
    print(f"Tỉ lệ mạng được kết nối: {conn_perc:.2f}%")
    
    valid_delays = np.array([d for d in path_delays if not np.isnan(d)])
    
    print(f"\n--- Phân tích Đường đi ({source} -> {destination}) ---")
    if len(valid_delays) > 0:
        print(f"Tỉ lệ thời gian tìm thấy đường đi: {len(valid_delays) / len(time_slots) * 100:.2f}%")
        print(f"Độ trễ trung bình (SP): {np.mean(valid_delays):.3f}ms")
        print(f"Độ trễ tối thiểu (SP): {np.min(valid_delays):.3f}ms")
        print(f"Số hop trung bình: {np.mean([h for h in path_hops if not np.isnan(h)]):.2f}")
    else:
        print("Không tìm thấy đường đi nào trong suốt thời gian mô phỏng.")

    # 3. Vẽ biểu đồ (Đảm bảo không lỗi)
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ 1: Số lượng Liên kết
    plt.subplot(1, 2, 1)
    plt.plot(time_slots, link_counts)
    plt.title('Số lượng Liên kết (Links) theo Thời gian')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Số lượng Liên kết')
    
    # Biểu đồ 2: Độ trễ Đường đi ngắn nhất
    plt.subplot(1, 2, 2)
    
    if len(valid_delays) > 0:
        delays_to_plot = np.array(path_delays)
        plt.plot(time_slots, delays_to_plot, label='Độ trễ SP')
        plt.ylim(bottom=0)
    else:
        plt.text(0.5, 0.5, 'Không có đường đi nào được tìm thấy.', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, color='red')
                 
    plt.title(f'Độ trễ Đường đi ngắn nhất {source} -> {destination}')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Độ trễ (ms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/network_dynamics_analysis.png')


if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    analyze_network_dynamics()
