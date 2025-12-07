# File: simulation/network_model.py (FINAL SYNTHETIC MODEL)

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import time

# --- Cấu hình Mạng Synthetic ---
NUM_SATS = 100
SIM_DURATION_SECONDS = 300
TIME_SLOT_SIZE = 1.0
NUM_GROUND_STATIONS = 5

# Tham số Delay
BASE_DELAY_ISL_MS = 20.0 
BASE_DELAY_GSL_MS = 15.0 

# Các trạm mặt đất
GS_COORDS = [
    (21.0285, 105.8542),   # GS_0: Hà Nội
    (40.7128, -74.0060),   # GS_1: New York
    (51.5074, 0.1278),     # GS_2: London
    (35.6895, 139.6917),   # GS_3: Tokyo (Chuyển Tokyo lên 3 để dễ chọn)
    (-33.8688, 151.2093)   # GS_4: Sydney (Chuyển Sydney xuống 4)
]

# TLE_PATH và các hằng số Skyfield không còn cần thiết

class DynamicLEONetwork:
    """Mô hình tạo ra đồ thị mạng động Synthetic (đảm bảo kết nối và tính động)."""
    
    def __init__(self):
        self.sat_names = [f'Sat_{i}' for i in range(NUM_SATS)]
        self.gs_names = [f'GS_{i}' for i in range(NUM_GROUND_STATIONS)]
        
        self.all_nodes = self.sat_names + self.gs_names
        self.num_nodes = len(self.all_nodes)
        
        self.node_to_id = {name: i for i, name in enumerate(self.all_nodes)}
        self.id_to_node = {i: name for name, i in self.node_to_id.items()}
        print(f"Tổng số nút trong mạng: {self.num_nodes}")
        print(f"Sử dụng mô hình Dynamic Synthetic Graph (100 Sat + 5 GS).")

    def get_network_graph(self, time_sec: float) -> nx.Graph:
        """Tạo đồ thị động tại thời điểm t."""
        G = nx.Graph()
        G.add_nodes_from(self.all_nodes)
        
        # 1. Liên kết ISL (giữa các vệ tinh)
        for i in range(NUM_SATS):
            sat_u = self.sat_names[i]
            for j in range(i + 1, NUM_SATS):
                sat_v = self.sat_names[j]
                
                # Xác suất kết nối: 1% để tạo mạng thưa và thay đổi (khó khăn cho routing)
                P_connect = 0.01 
                
                if np.random.rand() < P_connect:
                    # Độ trễ động (Base + Noise + Dynamic Component)
                    dynamic_delay_ms = BASE_DELAY_ISL_MS + np.random.uniform(-5, 5) + np.sin(time_sec / 30) * 5 
                    if dynamic_delay_ms > 1:
                        G.add_edge(sat_u, sat_v, 
                                   weight=dynamic_delay_ms, type='ISL',
                                   bandwidth_Mbps=1000)
                               
        # 2. Liên kết GSL (GS <-> Sat)
        for gs_name in self.gs_names:
            # Mỗi GS kết nối với 10 vệ tinh (đảm bảo GSL tồn tại)
            k_closest_sats = np.random.choice(self.sat_names, 10, replace=False)
            for sat_name in k_closest_sats:
                gsl_delay_ms = BASE_DELAY_GSL_MS + np.random.uniform(-5, 2)
                if gsl_delay_ms > 1:
                    G.add_edge(gs_name, sat_name,
                               weight=gsl_delay_ms, type='GSL',
                               bandwidth_Mbps=500)

        return G

    def generate_traffic_demands(self, num_demands: int, max_rate_Mbps: float = 100) -> List[Dict]:
        demands = []
        gs_names = self.gs_names
        if len(gs_names) < 2: return demands

        for req_id in range(num_demands):
            s, d = np.random.choice(gs_names, 2, replace=False)
            rate = np.random.uniform(5, max_rate_Mbps)
            
            demands.append({
                'id': f'Req_{req_id}',
                'source': s,
                'destination': d,
                'rate_Mbps': rate
            })
        return demands


if __name__ == "__main__":
    
    network = DynamicLEONetwork() 
    
    print("-" * 50)
    print(f"Bắt đầu mô phỏng {network.num_nodes} nút trong {SIM_DURATION_SECONDS} giây...")
    
    time_slots = np.arange(0, SIM_DURATION_SECONDS, TIME_SLOT_SIZE)
    source = network.gs_names[2] # London
    destination = network.gs_names[3] # Tokyo
    
    for t_sec in time_slots:
        t_start_topo = time.time()
        G_t = network.get_network_graph(t_sec)
        t_end_topo = time.time()
        
        num_links = G_t.number_of_edges()
        is_connected = nx.is_connected(G_t)

        log_msg = f"Time t={t_sec:.1f}s (Build: {t_end_topo - t_start_topo:.4f}s): Links={num_links}, Connected={is_connected}"

        if nx.has_path(G_t, source, destination):
            shortest_path_delay = nx.shortest_path_length(G_t, source=source, target=destination, weight='weight')
            log_msg += f", SP Delay={shortest_path_delay:.3f}ms"
        else:
            log_msg += f", NO PATH."
        
        if t_sec % 30 == 0: 
             print(log_msg)
            
    print("-" * 50)
    print(f"Tổng thời gian mô phỏng cơ bản: {time.time() - start_time:.2f} giây.")
