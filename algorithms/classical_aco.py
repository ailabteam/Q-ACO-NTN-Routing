# File: algorithms/classical_aco.py

import cupy as cp
import numpy as np
import networkx as nx
import time # Import time
from typing import List, Tuple, Union
from routing_engine import RoutingEngine, NUM_ANTS, MAX_ITERATIONS, Q0, ALPHA, BETA 
from simulation.network_model import DynamicLEONetwork 

import time

# Hằng số ACO
LOCAL_DEPOSIT_RATE = 0.5 # Tỷ lệ pheromone lắng đọng cục bộ

def calculate_path_delay_gpu_optimized(path_ids_np: np.ndarray, delay_matrix_cp: cp.ndarray) -> float:
    """Tính độ trễ (delay) cho một đường đi (tính toán lookup trên GPU)."""
    delay = 0.0
    # Chuyển path_ids về CuPy để tra cứu
    path_ids_cp = cp.asarray(path_ids_np)

    # Tính tổng độ trễ bằng cách tra cứu ma trận delay trên GPU
    for i in range(len(path_ids_cp) - 1):
        u, v = path_ids_cp[i], path_ids_cp[i+1]

        # Lấy giá trị từ ma trận delay trên GPU
        d = delay_matrix_cp[u, v]

        if d == 0: # Không có liên kết
             return np.inf
        delay += d.get().item() # Dùng .item() để chuyển về float thuần túy

    return delay



def run_classical_aco(engine: RoutingEngine, G: nx.Graph, source_id: int, dest_id: int) -> Tuple[List[int], float]:
    """
    Thực thi Classical Ant Colony Optimization.
    Sử dụng CuPy để tăng tốc độ tính toán ma trận xác suất.
    """
    best_path_delay = np.inf
    best_path_cpu = None
    N = engine.num_nodes

    # ------------------ KHỞI TẠO GPU ------------------
    MAX_HOPS = N # Giới hạn số hop tối đa bằng số nút

    # Mảng lưu trữ đường đi: NUM_ANTS x Max_Hops
    ant_paths_cp = cp.full((NUM_ANTS, MAX_HOPS), -1, dtype=cp.int32)
    ant_paths_cp[:, 0] = source_id

    ants_current_node_cp = cp.full(NUM_ANTS, source_id, dtype=cp.int32)
    visited_mask_cp = cp.zeros((NUM_ANTS, N), dtype=cp.bool_)
    visited_mask_cp[:, source_id] = True
    ant_finished_cp = cp.zeros(NUM_ANTS, dtype=cp.bool_) # True nếu kiến đã đến đích

    # Tạo ma trận độ trễ trên GPU (đã được tối ưu hóa)
    delay_matrix_cpu = np.array(nx.adjacency_matrix(G, nodelist=engine.network.all_nodes, weight='weight').todense(), dtype=np.float32)
    delay_matrix_cp = cp.asarray(delay_matrix_cpu)

    start_time = time.time() # Để tính thời gian chạy ACO

    # ------------------ Vòng lặp ACO ------------------
    for iteration in range(MAX_ITERATIONS):

        # 1. Giai đoạn Tìm đường đi (Path Construction)
        for hop in range(1, MAX_HOPS):

            ants_moving_indices = cp.where(~ant_finished_cp)[0]
            if len(ants_moving_indices) == 0:
                break

            current_nodes_ids = ants_current_node_cp[ants_moving_indices]

            # Lấy ma trận Pheromone, Heuristic, Adjacency cho các nút hiện tại
            tau_mat = engine.pheromone_matrix[current_nodes_ids, :]
            eta_mat = engine.heuristic_matrix[current_nodes_ids, :]
            adj_mat = engine.adj_matrix_cp[current_nodes_ids, :]

            # SỬA LỖI: Sử dụng ALPHA và BETA đã được import
            # Numerator = Tau^ALPHA * Eta^BETA
            numerator = cp.power(tau_mat, ALPHA) * cp.power(eta_mat, BETA)

            # Loại bỏ các nút đã ghé thăm và các liên kết không tồn tại
            visited_moving = visited_mask_cp[ants_moving_indices, :]
            numerator = cp.where(~visited_moving, numerator, 0.0)
            numerator = cp.where(adj_mat, numerator, 0.0)

            # Denominator (Tổng xác suất cho mỗi kiến)
            denominator = cp.sum(numerator, axis=1, keepdims=True)

            # Xác suất (P): Tránh chia cho 0
            probs_matrix = cp.where(denominator > 0, numerator / denominator, 0.0)

            # --- CHỌN NÚT KẾ TIẾP (CPU Choice, tối ưu hóa I/O) ---

            probs_matrix_cpu = probs_matrix.get()
            N_nodes = probs_matrix_cpu.shape[1]

            new_next_nodes = cp.zeros(len(ants_moving_indices), dtype=cp.int32)

            for k in range(len(ants_moving_indices)):
                ant_idx = ants_moving_indices[k]
                probs_row = probs_matrix_cpu[k, :]

                if np.sum(probs_row) == 0:
                    # Kiến bị kẹt, giữ nguyên vị trí (không cần cập nhật)
                    continue

                # Sử dụng NumPy choice (CPU)
                next_id = np.random.choice(N_nodes, p=probs_row)
                new_next_nodes[k] = next_id

                # Cập nhật trạng thái trên GPU
                ants_current_node_cp[ant_idx] = next_id
                ant_paths_cp[ant_idx, hop] = next_id
                visited_mask_cp[ant_idx, next_id] = True

                if next_id == dest_id:
                    ant_finished_cp[ant_idx] = True


        # 2. Giai đoạn Đánh giá & Cập nhật Pheromone

        pheromone_delta = cp.zeros((N, N), dtype=cp.float32)
        
        ant_paths_cpu = ant_paths_cp.get() 
        num_ants_at_dest = 0 # Khởi tạo biến đếm
        
        for ant_idx in range(NUM_ANTS):
            
            if ant_paths_cpu[ant_idx, hop] == dest_id: 
                num_ants_at_dest += 1 # Đếm số kiến đến đích
                
                path_ids_np = ant_paths_cpu[ant_idx, ant_paths_cpu[ant_idx, :] != -1]
                path_delay = calculate_path_delay_gpu_optimized(path_ids_np, delay_matrix_cp)
                
                if path_delay == np.inf: 
                    continue
                
                if path_delay < best_path_delay:
                    best_path_delay = path_delay
                    best_path_cpu = path_ids_np.tolist()
                    
                    time_elapsed = time.time() - start_time
                    print(f"  > Iter {iteration}: New best delay = {best_path_delay:.3f}ms (Hops: {len(best_path_cpu) - 1}) [Time: {time_elapsed:.2f}s]")

                # Cập nhật Pheromone cục bộ
                tau_deposit = Q0 / path_delay 
                
                for k in range(len(path_ids_np) - 1):
                    u, v = path_ids_np[k], path_ids_np[k+1]
                    pheromone_delta[u, v] += tau_deposit
                    pheromone_delta[v, u] += tau_deposit
        
        # DEBUG LOG: In số lượng kiến đến đích
        print(f"  > DEBUG: Iter {iteration}: Ants reached destination: {num_ants_at_dest}/{NUM_ANTS}")

        # Cập nhật Pheromone toàn cục
        engine.pheromone_matrix += pheromone_delta



        # Reset trạng thái kiến cho lần lặp tiếp theo
        ant_paths_cp[:, :] = -1
        ant_paths_cp[:, 0] = source_id
        ants_current_node_cp = cp.full(NUM_ANTS, source_id, dtype=cp.int32)
        visited_mask_cp = cp.zeros((NUM_ANTS, N), dtype=cp.bool_)
        visited_mask_cp[:, source_id] = True
        ant_finished_cp = cp.zeros(NUM_ANTS, dtype=cp.bool_)


    return best_path_cpu, best_path_delay




def calculate_path_delay(G: nx.Graph, path_ids: List[int]) -> float:
    """Tính độ trễ (delay) cho một đường đi (trên CPU)."""
    delay = 0.0
    for i in range(len(path_ids) - 1):
        u_name = G.graph['id_to_node'][path_ids[i]]
        v_name = G.graph['id_to_node'][path_ids[i+1]]

        # Phải dùng try-except vì liên kết có thể đã bị xóa khỏi G trong khi kiến đang di chuyển (dù ít khả năng trong mô hình synthetic)
        if G.has_edge(u_name, v_name):
             delay += G[u_name][v_name]['weight']
        else:
             return cp.inf # Đường đi không hợp lệ
    return delay

if __name__ == '__main__':
    # Khởi tạo mô hình
    network = DynamicLEONetwork()

    # Gán ID to Name mapping vào Graph để hàm calculate_path_delay có thể truy cập
    network_model = DynamicLEONetwork()
    G_t = network_model.get_network_graph(0.0)
    G_t.graph['id_to_node'] = network_model.id_to_node # Gán mapping vào graph

    engine = RoutingEngine(network_model)
    engine.update_state(G_t)

    source = network_model.gs_names[2] # London
    dest = network_model.gs_names[3] # Tokyo
    s_id = network_model.node_to_id[source]
    d_id = network_model.node_to_id[dest]

    print(f"Chạy Classical ACO cho {source} -> {dest}...")
    path, delay = run_classical_aco(engine, G_t, s_id, d_id)

    print(f"\nKết quả Classical ACO:")
    print(f"  Độ trễ tốt nhất: {delay:.3f}ms")
    if path:
        path_names = [network_model.id_to_node[i] for i in path]
        print(f"  Đường đi: {' -> '.join(path_names)}")
