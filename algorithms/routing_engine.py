# File: algorithms/routing_engine.py

import cupy as cp
import numpy as np
import networkx as nx
import time
import sys
import os
import cupy as cp


# FIX: Đảm bảo import từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.network_model import DynamicLEONetwork, NUM_SATS, NUM_GROUND_STATIONS

# --- Cấu hình ACO/Q-ACO chung (Đã tối ưu hóa) ---
ALPHA = 1.0  # Trọng số Pheromone
BETA = 8.0   # Tăng cường Trọng số Heuristic (Độ trễ)
RHO = 0.1    # Tốc độ bay hơi Pheromone
Q0 = 1000.0  # Tăng cường lắng đọng Pheromone (học nhanh hơn)
NUM_ANTS = 1024 # Tăng số lượng Kiến (tận dụng GPU)
MAX_ITERATIONS = 50 

class RoutingEngine:
    def __init__(self, network_model):
        self.network = network_model
        self.num_nodes = self.network.num_nodes
        self.node_to_id = self.network.node_to_id

        # Pheromone Matrix (GPU - Tau)
        self.pheromone_matrix = cp.ones((self.num_nodes, self.num_nodes), dtype=cp.float32) * 1e-4

        # Heuristic Matrix (GPU - Eta = 1/Delay)
        self.heuristic_matrix = cp.zeros((self.num_nodes, self.num_nodes), dtype=cp.float32)

        # Ma trận chứa các thông tin phụ cần thiết cho việc di chuyển kiến (chỉ có 0 hoặc 1)
        self.adj_matrix_cp = cp.zeros((self.num_nodes, self.num_nodes), dtype=cp.bool_)

    def update_state(self, G: nx.Graph):
        """Cập nhật ma trận Heuristic và Ma trận kề (Adj Matrix) cho mạng hiện tại."""

        H_cpu = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        Adj_cpu = np.zeros((self.num_nodes, self.num_nodes), dtype=np.bool_)

        for u, v, data in G.edges(data=True):
            u_id = self.node_to_id[u]
            v_id = self.node_to_id[v]

            delay = data.get('weight', 1e9)

            # 1. Heuristic: Inverse of Delay
            if delay > 0:
                eta = 1.0 / (delay + 1e-6)
                H_cpu[u_id, v_id] = H_cpu[v_id, u_id] = eta

            # 2. Ma trận kề (Adj Matrix)
            Adj_cpu[u_id, v_id] = Adj_cpu[v_id, u_id] = True

        # Chuyển ma trận lên GPU
        self.heuristic_matrix = cp.asarray(H_cpu)
        self.adj_matrix_cp = cp.asarray(Adj_cpu)

        # 3. Bay hơi Pheromone (Evaporation)
        self.evaporate_pheromone()


    def evaporate_pheromone(self):
        """Thực hiện quá trình bay hơi Pheromone trên GPU."""
        # Tau_new = (1 - Rho) * Tau_old
        self.pheromone_matrix *= (1.0 - RHO)



    def calculate_transition_probs(self, current_node_id: int) -> cp.ndarray:
        """Tính ma trận xác suất chuyển tiếp cho nút hiện tại (Sử dụng GPU)."""

        tau_row = self.pheromone_matrix[current_node_id, :]
        eta_row = self.heuristic_matrix[current_node_id, :]
        adj_row = self.adj_matrix_cp[current_node_id, :]

        # 1. Tính toán Tử số (Numerator)
        # Numerator = Tau^Alpha * Eta^Beta

        # NOTE: Các giá trị eta_row=0 (không có link) sẽ làm cho numerator=0.
        # Chúng ta chỉ cần đảm bảo rằng ta không tính toán trên các link không tồn tại (adj_row=False).
        numerator = cp.power(tau_row, ALPHA) * cp.power(eta_row, BETA)

        # Áp dụng ràng buộc mạng (chỉ xét hàng xóm có liên kết)
        numerator = cp.where(adj_row, numerator, 0.0)

        # 2. Tính toán Mẫu số (Denominator)
        denominator = cp.sum(numerator)

        # 3. Tính Xác suất P
        if denominator == 0:
            return cp.zeros(self.num_nodes, dtype=cp.float32)

        # Sử dụng CuPy where để chia cho mẫu số, nếu mẫu số là 0 thì kết quả là 0
        # Mặc dù chúng ta đã kiểm tra denominator == 0, việc sử dụng cp.where là an toàn hơn
        probs = cp.where(denominator > 0, numerator / denominator, 0.0)

        return probs


    # --- Triển khai trong các file riêng ---
    # Classical ACO logic sẽ được đặt trong algorithms/classical_aco.py
    # Quantum ACO logic sẽ được đặt trong algorithms/q_aco.py


if __name__ == '__main__':
    # Kiểm tra CuPy
    if cp.cuda.is_available():
        gpu_id = 0
        cp.cuda.Device(gpu_id).use()
        print(f"CuPy sẵn sàng, sử dụng GPU ID: {cp.cuda.Device(gpu_id).id}")
    else:
        print("Cảnh báo: CuPy không khả dụng. Sẽ chạy trên CPU (NumPy).")

    # Chạy thử Engine
    network = DynamicLEONetwork()
    engine = RoutingEngine(network)

    # Kiểm tra cập nhật state
    G0 = network.get_network_graph(0.0)
    engine.update_state(G0)

    print(f"Kích thước Pheromone Matrix (GPU): {engine.pheromone_matrix.shape}")

    # Tính thử xác suất chuyển tiếp từ GS_0 (Hà Nội)
    gs0_id = engine.node_to_id['GS_0']
    probs = engine.calculate_transition_probs(gs0_id)

    print(f"Tổng xác suất chuyển tiếp từ GS_0: {cp.sum(probs):.4f}")
