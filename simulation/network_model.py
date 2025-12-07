import numpy as np
import networkx as nx
from skyfield.api import load, EarthSatellite, Topos
from typing import List, Tuple, Dict
import time

# --- Cấu hình Mạng ---
NUM_SATS = 100              # Số lượng vệ tinh được sử dụng (lấy từ đầu file TLE)
SIM_DURATION_SECONDS = 300  # 5 phút mô phỏng
TIME_SLOT_SIZE = 1.0        # Mỗi 1.0 giây cập nhật topo
MAX_ISL_RANGE_KM = 3000.0   # Phạm vi liên kết giữa các vệ tinh
SPEED_OF_LIGHT_KM_PER_S = 299792.458
MIN_ELEVATION_DEG = 10.0    # Góc tối thiểu cho liên kết GSL

class DynamicLEONetwork:
    def __init__(self, tle_file_path: str, gs_coords: List[Tuple[float, float]]):
        self.ts = load.timescale()
        
        # 1. Tải Vệ tinh TLE từ file local
        print(f"Đang tải TLE từ file local: {tle_file_path}...")
        try:
            # Tải toàn bộ và chỉ lấy NUM_SATS đầu tiên
            full_sats = load.tle_file(tle_file_path, lazy=False)
            self.sats = full_sats[:NUM_SATS]
            
            if len(self.sats) == 0:
                 raise ValueError("File TLE trống hoặc định dạng không đúng.")
            if len(self.sats) < NUM_SATS:
                 print(f"Cảnh báo: Chỉ tìm thấy {len(self.sats)} vệ tinh, sử dụng tất cả.")
            else:
                 print(f"Đã tải thành công {len(self.sats)} vệ tinh.")
                 
            self.sat_names = [s.name for s in self.sats]
            
        except Exception as e:
            print(f"Lỗi khi tải TLE từ file: {e}")
            raise

        # 2. Load Trạm Mặt đất
        self.ground_stations = [Topos(lat=lat, lon=lon) for lat, lon in gs_coords]
        self.gs_names = [f'GS_{i}' for i in range(len(gs_coords))]
        
        self.all_nodes = self.sat_names + self.gs_names
        self.num_nodes = len(self.all_nodes)
        
        # Mapping từ tên nút sang ID (dùng cho ma trận)
        self.node_to_id = {name: i for i, name in enumerate(self.all_nodes)}
        self.id_to_node = {i: name for name, i in self.node_to_id.items()}
        print(f"Tổng số nút trong mạng: {self.num_nodes}")

    def get_network_graph(self, time_sec: float) -> nx.Graph:
        """
        Tính toán topo mạng và đồ thị tại thời điểm t.
        Đầu ra: nx.Graph với trọng số cạnh là độ trễ truyền lan (Propagation Delay).
        """
        t = self.ts.now() + time_sec / (24 * 60 * 60) # Chuyển đổi giây sang Skyfield time

        G = nx.Graph()
        G.add_nodes_from(self.all_nodes)

        # 1. Tính toán vị trí và khoảng cách giữa các vệ tinh (ISL)
        sat_positions_km = np.array([sat.at(t).position.km for sat in self.sats])
        distances = np.sqrt(np.sum((sat_positions_km[:, np.newaxis, :] - sat_positions_km[np.newaxis, :, :]) ** 2, axis=2))
        
        # Vectorized ISL Check
        isl_indices = np.argwhere(
            (distances > 0) & (distances <= MAX_ISL_RANGE_KM)
        )
        
        # Thêm liên kết ISL (chỉ xét i < j)
        for i, j in isl_indices:
            if i < j:
                dist = distances[i, j]
                delay_ms = (dist / SPEED_OF_LIGHT_KM_PER_S) * 1000 
                # Thêm trọng số chính (delay) và thông tin phụ (distance)
                G.add_edge(self.sat_names[i], self.sat_names[j], 
                           weight=delay_ms, dist=dist, type='ISL',
                           # Giả định ban đầu băng thông cố định cho ISL
                           bandwidth_Mbps=1000) 
        
        # 2. Thêm liên kết GSL (Ground-Satellite Links)
        for i, gs in enumerate(self.ground_stations):
            gs_name = self.gs_names[i]
            for j, sat in enumerate(self.sats):
                # Tính toán Góc cao (Elevation)
                topocentric = (gs - sat.at(t)).frame_latlon(gs)
                alt, az, dist_obj = topocentric.altaz()
                
                if alt.degrees > MIN_ELEVATION_DEG:
                    dist_km = dist_obj.km
                    delay_ms = (dist_km / SPEED_OF_LIGHT_KM_PER_S) * 1000
                    G.add_edge(gs_name, sat.name, 
                               weight=delay_ms, dist=dist_km, type='GSL',
                               # Giả định ban đầu băng thông cố định cho GSL
                               bandwidth_Mbps=500) 
        
        return G

    def generate_traffic_demands(self, num_demands: int, max_rate_Mbps: float = 100) -> List[Dict]:
        """Tạo các yêu cầu lưu lượng với tốc độ dữ liệu ngẫu nhiên."""
        demands = []
        gs_names = self.gs_names
        if len(gs_names) < 2: return demands

        for req_id in range(num_demands):
            # Chọn ngẫu nhiên 2 trạm mặt đất
            s, d = np.random.choice(gs_names, 2, replace=False)
            rate = np.random.uniform(5, max_rate_Mbps) # Tốc độ yêu cầu (Mbps)
            
            demands.append({
                'id': f'Req_{req_id}',
                'source': s,
                'destination': d,
                'rate_Mbps': rate
            })
        return demands


# --- Khởi tạo Demo ---
if __name__ == "__main__":
    
    # Các trạm mặt đất giả định 
    GS_COORDS = [
        (21.0285, 105.8542),   # GS_0: Hà Nội
        (40.7128, -74.0060),   # GS_1: New York
        (51.5074, 0.1278),     # GS_2: London
        (-33.8688, 151.2093),  # GS_3: Sydney
        (35.6895, 139.6917)    # GS_4: Tokyo
    ]
    
    TLE_PATH = 'data/starlink.txt'
    
    try:
        start_time = time.time()
        network = DynamicLEONetwork(TLE_PATH, GS_COORDS)
    except Exception as e:
        print(f"Không thể khởi tạo: {e}")
        exit()
        
    print("-" * 50)
    print(f"Bắt đầu mô phỏng {network.num_nodes} nút trong {SIM_DURATION_SECONDS} giây...")
    
    time_slots = np.arange(0, SIM_DURATION_SECONDS, TIME_SLOT_SIZE)
    
    # Tạo yêu cầu lưu lượng cố định (Hà Nội -> New York)
    source = network.gs_names[0] 
    destination = network.gs_names[1]
    
    for t_sec in time_slots:
        # Đo thời gian xây dựng topo
        t_start_topo = time.time()
        G_t = network.get_network_graph(t_sec)
        t_end_topo = time.time()
        
        num_links = G_t.number_of_edges()
        is_connected = nx.is_connected(G_t)
        
        log_msg = f"Time t={t_sec:.1f}s (Build: {t_end_topo - t_start_topo:.4f}s): Links={num_links}, Connected={is_connected}"

        # Tìm đường đi ngắn nhất (Dijkstra) làm tham chiếu
        if nx.has_path(G_t, source, destination):
            shortest_path_delay = nx.shortest_path_length(G_t, source=source, target=destination, weight='weight')
            log_msg += f", SP Delay={shortest_path_delay:.3f}ms"
        else:
            log_msg += f", NO PATH."
        
        print(log_msg)
            
    print("-" * 50)
    print(f"Tổng thời gian mô phỏng cơ bản: {time.time() - start_time:.2f} giây.")
