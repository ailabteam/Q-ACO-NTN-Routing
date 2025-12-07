import numpy as np
import networkx as nx
from skyfield.api import load, Topos
from typing import List, Tuple, Dict
import time
import os

# --- Cấu hình Mạng (Hằng số được định nghĩa ở cấp module) ---
NUM_SATS = 100              # Số lượng vệ tinh được sử dụng
SIM_DURATION_SECONDS = 300  # 5 phút mô phỏng
TIME_SLOT_SIZE = 1.0        # Mỗi 1.0 giây cập nhật topo
MAX_ISL_RANGE_KM = 3000.0   # Phạm vi liên kết giữa các vệ tinh (3000km là điển hình)
SPEED_OF_LIGHT_KM_PER_S = 299792.458
MIN_ELEVATION_DEG = 10.0    # Góc tối thiểu cho liên kết GSL

# Các trạm mặt đất giả định
GS_COORDS = [
    (21.0285, 105.8542),   # GS_0: Hà Nội
    (40.7128, -74.0060),   # GS_1: New York
    (51.5074, 0.1278),     # GS_2: London
    (-33.8688, 151.2093),  # GS_3: Sydney
    (35.6895, 139.6917)    # GS_4: Tokyo
]

TLE_PATH = 'data/starlink.txt' # Đường dẫn TLE

class DynamicLEONetwork:
    """Mô hình tạo ra đồ thị mạng LEO động theo thời gian."""

    def __init__(self, tle_file_path: str, gs_coords: List[Tuple[float, float]]):

        # 0. Tải Ephemeris và Timescale
        print("Đang tải ephemeris (có thể mất vài giây nếu chưa cache)...")
        self.eph = load('de421.bsp')
        self.ts = load.timescale()

        if not os.path.exists(tle_file_path):
            raise FileNotFoundError(f"File TLE không tìm thấy tại đường dẫn: {tle_file_path}")

        # 1. Tải Vệ tinh TLE từ file local
        print(f"Đang tải TLE từ file local: {tle_file_path}...")
        try:
            # load.tle_file() không cần ephemeris hay lazy=False
            full_sats = load.tle_file(tle_file_path)
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
        self.ground_station_topos = [] # Lưu Topos object gốc (chỉ tọa độ)
        self.ground_stations_ref = []  # Lưu Earth + Topos (dùng cho phép tính GSL)

        for lat, lon in gs_coords:
            # Sử dụng keyword an toàn cho Skyfield
            gs_loc = Topos(latitude_degrees=lat, longitude_degrees=lon)
            self.ground_station_topos.append(gs_loc)
            # Liên kết với Trái Đất để tạo reference frame
            self.ground_stations_ref.append(self.eph['earth'] + gs_loc)

        self.gs_names = [f'GS_{i}' for i in range(len(gs_coords))]

        self.all_nodes = self.sat_names + self.gs_names
        self.num_nodes = len(self.all_nodes)

        self.node_to_id = {name: i for i, name in enumerate(self.all_nodes)}
        self.id_to_node = {i: name for name, i in self.node_to_id.items()}
        print(f"Tổng số nút trong mạng: {self.num_nodes}")

    def get_network_graph(self, time_sec: float) -> nx.Graph:
        """
        Tính toán topo mạng và đồ thị tại thời điểm t.
        Trọng số cạnh là Độ trễ truyền lan (Propagation Delay).
        """
        t = self.ts.now() + time_sec / (24 * 60 * 60) # Cập nhật thời gian Skyfield

        G = nx.Graph()
        G.add_nodes_from(self.all_nodes)

        # 1. Tính toán ISL (Inter-Satellite Links)
        sat_positions_km = np.array([sat.at(t).position.km for sat in self.sats])
        distances = np.sqrt(np.sum((sat_positions_km[:, np.newaxis, :] - sat_positions_km[np.newaxis, :, :]) ** 2, axis=2))

        isl_indices = np.argwhere(
            (distances > 0) & (distances <= MAX_ISL_RANGE_KM)
        )

        for i, j in isl_indices:
            if i < j:
                dist = distances[i, j]
                delay_ms = (dist / SPEED_OF_LIGHT_KM_PER_S) * 1000
                G.add_edge(self.sat_names[i], self.sat_names[j],
                           weight=delay_ms, dist=dist, type='ISL',
                           bandwidth_Mbps=1000)

        # 2. Thêm liên kết GSL (Ground-Satellite Links)
        for i, gs_ref in enumerate(self.ground_stations_ref): 
            gs_name = self.gs_names[i]
            gs_topos = self.ground_station_topos[i] 
            
            for j, sat in enumerate(self.sats):
                difference = sat.at(t) - gs_ref.at(t)
                
                # Lấy thông tin topocentric từ difference.frame_latlon(gs_topos)
                # FIX: frame_latlon trả về tuple (alt, az, dist), không phải đối tượng Position
                alt, az, dist_obj = difference.frame_latlon(gs_topos) 
                
                # Bây giờ 'alt' là đối tượng Angle, có thể truy cập giá trị độ
                if alt.degrees > MIN_ELEVATION_DEG:
                    dist_km = dist_obj.km # dist_obj là đối tượng Distance
                    delay_ms = (dist_km / SPEED_OF_LIGHT_KM_PER_S) * 1000
                    G.add_edge(gs_name, sat.name, 
                               weight=delay_ms, dist=dist_km, type='GSL',
                               bandwidth_Mbps=500) 
        
        return G

    def generate_traffic_demands(self, num_demands: int, max_rate_Mbps: float = 100) -> List[Dict]:
        """Tạo các yêu cầu lưu lượng với tốc độ dữ liệu ngẫu nhiên."""
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


# --- Khởi tạo Demo (Phần main giữ nguyên) ---
if __name__ == "__main__":

    try:
        start_time = time.time()
        network = DynamicLEONetwork(TLE_PATH, GS_COORDS)
    except Exception as e:
        print(f"Không thể khởi tạo: {e}")
        exit()

    print("-" * 50)
    print(f"Bắt đầu mô phỏng {network.num_nodes} nút trong {SIM_DURATION_SECONDS} giây...")

    time_slots = np.arange(0, SIM_DURATION_SECONDS, TIME_SLOT_SIZE)
    source = network.gs_names[0]
    destination = network.gs_names[1]

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
