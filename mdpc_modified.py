import numpy as np
import time
from scipy.spatial import distance
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import HuberRegressor

def find_density_peaks(rho, delta, M=3):
    """改进的密度峰值检测(基于Huber回归)"""
    log_rho = np.log(rho.clip(1e-10))
    log_delta = np.log(delta.clip(1e-10))
    
    # Huber回归拟合
    huber = HuberRegressor(epsilon=1.35).fit(log_rho.reshape(-1,1), log_delta)
    residuals = log_delta - huber.predict(log_rho.reshape(-1,1))
    threshold = huber.predict(log_rho.reshape(-1,1)) + M * np.std(residuals)
    
    return log_delta > threshold

def mdpc_plus_modified(data, k=None, num_cluster=3, M=3):
    """改进版MDPC+算法（核心逻辑与原始算法相同，仅修改峰值检测部分）"""
    n, dim = data.shape
    alpha = 0.5
    lambda_ = 2
    
    if k is None:
        k = round(np.sqrt(n))
    k_b = min(2 * int(np.floor(np.log(n))), k)
    
    start_time = time.time()
    
    # KNN计算（与原始算法相同）
    if dim <= 11:
        nbrs = NearestNeighbors(n_neighbors=max(k, k_b), algorithm='kd_tree').fit(data)
        knn_dist, knn = nbrs.kneighbors(data)
    else:
        dist_matrix = distance.squareform(distance.pdist(data))
        knn = np.argsort(dist_matrix, axis=1)
        knn_dist = np.sort(dist_matrix, axis=1)
    
    # 密度计算（与原始算法相同）
    rho = np.sum(np.exp(-knn_dist[:, 1:k])**2, axis=1)
    ord_rho = np.argsort(rho)[::-1]
    
    # ========== 关键改进：计算delta并检测峰值 ==========
    delta = np.zeros(n)
    for i in range(n):
        higher_mask = rho > rho[i]
        if np.any(higher_mask):
            delta[i] = np.min(np.linalg.norm(data[i] - data[higher_mask], axis=1))
        else:
            delta[i] = np.max(np.linalg.norm(data - data[i], axis=1))
    
    # 使用Huber回归检测峰值（替换原始逻辑）
    peaks_mask = find_density_peaks(rho, delta, M)
    peaks = np.where(peaks_mask)[0].tolist()
    n_p = len(peaks)
    
    # ========== 后续流程与原始算法完全一致 ==========
    # 包括父节点关联、子簇合并、中心选择等步骤
    # 初始化父节点关联参数
    phi = np.zeros(n)
    pn = np.full(n, -1, dtype=int)

    # 构建密度关联树（使用改进后的peaks列表）
    ord_rho = np.argsort(rho)[::-1]
    for i in range(n):
        point = ord_rho[i]
        if point not in peaks:  # 仅在非峰值点寻找父节点
            higher_mask = rho[knn[point]] > rho[point]
            valid_neigh = knn[point][higher_mask]
            
            if len(valid_neigh) > 0:
                # 选择最近的高密度邻居作为父节点
                pn[point] = valid_neigh[np.argmin(knn_dist[point][higher_mask])]
                phi[point] = (abs(rho[pn[point]] - rho[point]) / rho[pn[point]]) ** lambda_

    # 密度偏差累积成本计算
    denisty_deviation_cost = np.zeros(n)
    for i in range(n):
        if pn[ord_rho[i]] != -1:
            denisty_deviation_cost[ord_rho[i]] = denisty_deviation_cost[pn[ord_rho[i]]] + phi[ord_rho[i]]

    # 子簇分配
    sub_l = np.full(n, -1)
    n_p = len(peaks)
    sub_l[peaks] = np.arange(n_p)  # 子簇编号从0开始
    
    for i in ord_rho:
        if sub_l[i] == -1 and pn[i] != -1:
            sub_l[i] = sub_l[pn[i]]

    # 构建子簇连接图
    edges = np.full((n_p, n_p), np.inf)
    for i in range(n):
        for j in range(1, k_b):
            neighbor = knn[i, j]
            if sub_l[i] != sub_l[neighbor]:
                # 计算连接成本（公式9）
                cost = (denisty_deviation_cost[i] + denisty_deviation_cost[neighbor] + 
                       (np.abs(rho[i]-rho[neighbor])/np.maximum(rho[i],rho[neighbor]))**lambda_)
                
                u = sub_l[i]
                v = sub_l[neighbor]
                if cost < edges[u, v]:
                    edges[u, v] = edges[v, u] = cost

    # 计算子簇间最短路径
    dist_matrix, predecessors = dijkstra(edges, directed=False, return_predecessors=True)

    # 动态合并子簇
    delta_peaks = np.full(n_p, np.inf)
    pn_peaks = np.full(n_p, -1, dtype=int)
    ord_peaks = np.argsort(rho[peaks])[::-1]  # 按密度排序子簇
    
    for i in range(1, n_p):
        current = ord_peaks[i]
        for j in range(i):
            candidate = ord_peaks[j]
            if dist_matrix[current, candidate] < delta_peaks[current]:
                delta_peaks[current] = dist_matrix[current, candidate]
                pn_peaks[current] = candidate

    # 确定最终簇中心
    gamma = rho[peaks] * delta_peaks
    selected_peaks = np.argsort(gamma)[-num_cluster:][::-1]
    centers = [peaks[i] for i in selected_peaks]

    # 分配最终标签
    cluster_map = {old: new+1 for new, old in enumerate(selected_peaks)}
    final_labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        original_cluster = sub_l[i]
        if original_cluster in cluster_map:
            final_labels[i] = cluster_map[original_cluster]
        else:
            # 处理未映射的子簇
            while pn_peaks[original_cluster] != -1:
                original_cluster = pn_peaks[original_cluster]
            final_labels[i] = cluster_map.get(original_cluster, 0)

    # 计算运行时间
    runtime = time.time() - start_time
    
    return final_labels, num_cluster, centers, runtime