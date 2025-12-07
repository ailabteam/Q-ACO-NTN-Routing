# File: algorithms/routing_engine.py

import cupy as cp
import numpy as np
import networkx as nx
import time

# --- Cấu hình ACO/Q-ACO chung ---
ALPHA = 1.0  # Trọng số Pheromone
BETA = 2.0   # Trọng số Heuristic (Độ trễ)
RHO = 0.1    # Tốc độ bay hơi Pheromone (Evaporation Rate)
Q0 = 1.0     # Hằng số tích lũy Pheromone
NUM_ANTS = 500 # Số lượng Kiến (Tăng cao để tận dụng GPU)
MAX_ITERATIONS = 50 # Số lần lặp tìm kiếm trong mỗi time slot

class RoutingEngine:
    def __init__(self, network_model):
        self.network = network_model
        self.num_nodes = self.network.num_nodes
        
        # Pheromone Matrix (Sử dụng CuPy cho tính toán GPU)
        # Khởi tạo ma trận Pheromone với giá trị ban đầu nhỏ trên GPU
        self.pheromone_matrix = cp.ones((self.num_nodes, self.num_nodes), dtype=cp.float32) * 1e-4

        # Ma trận Heuristic (Eta - 1/Delay)
        self.heuristic_matrix = cp.zeros((self.num_nodes, self.num_nodes), dtype=cp.float32)

    def update_heuristic_matrix(self, G: nx.Graph):
        """Cập nhật ma trận Heuristic dựa trên độ trễ của mạng hiện tại."""
        
        # Khởi tạo ma trận heuristic_matrix (eta) trên CPU
        H_cpu = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        # Lấy độ trễ (weight) cho tất cả các cạnh
        for u, v, data in G.edges(data=True):
            u_id = self.network.node_to_id[u]
            v_id = self.network.node_to_id[v]
            
            delay = data.get('weight', 1e9) # Độ trễ (ms)
            
            # Heuristic: Inverse of Delay (1 / Delay)
            # Thêm 1e-6 để tránh chia cho 0
            if delay > 0:
                eta = 1.0 / (delay + 1e-6)
                H_cpu[u_id, v_id] = H_cpu[v_id, u_id] = eta
        
        # Chuyển ma trận Heuristic lên GPU
        self.heuristic_matrix = cp.asarray(H_cpu)

    def evaporate_pheromone(self):
        """Thực hiện quá trình bay hơi Pheromone trên GPU."""
        # Tau_new = (1 - Rho) * Tau_old
        self.pheromone_matrix *= (1.0 - RHO)

    # placeholder cho logic tìm đường đi ACO cổ điển
    def run_classical_aco(self, source_id, dest_id, G: nx.Graph):
        # ... sẽ được triển khai ở file khác
        pass

    # placeholder cho logic tìm đường đi Q-ACO
    def run_quantum_aco(self, source_id, dest_id, G: nx.Graph):
        # ... sẽ được triển khai ở file khác
        pass


if __name__ == '__main__':
    # Kiểm tra CuPy đã sẵn sàng chưa
    print(f"CuPy version: {cp.__version__}")
    print(f"CuPy ready: {'Yes' if cp.cuda.is_available() else 'No'}")
    
    if cp.cuda.is_available():
        # Lấy thông tin GPU (nên là 0 hoặc 1, vì bạn có 2 GPU)
        gpu_id = 0
        cp.cuda.Device(gpu_id).use() 
        print(f"Sử dụng GPU ID: {cp.cuda.Device(gpu_id).id}")
    else:
        print("Cảnh báo: CuPy không khả dụng. Sẽ chạy trên CPU (NumPy).")
