import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score

def mdpc_plus(data, k, num_cluster):
    # Parameters
    n, dim = data.shape
    alpha = 0.5
    lambda_ = 2
    
    if k is None:
        k = round(np.sqrt(n))  # Number of neighbors
    k_b = min(2 * int(np.floor(np.log(n))), k)
    
    # Timing start
    start_time = time.time()
    
    # Fast KNN based on KD-tree (when dimension is not larger than 10)
    if dim <= 11:
        nbrs = NearestNeighbors(n_neighbors=max(k, k_b), algorithm='kd_tree').fit(data)
        knn_dist, knn = nbrs.kneighbors(data)
    else:
        dist_matrix = distance.squareform(distance.pdist(data))
        knn = np.argsort(dist_matrix, axis=1)
        knn_dist = np.sort(dist_matrix, axis=1)
    
    # KNN-based Density
    rho = np.sum(np.exp(-knn_dist[:, 1:k])**2, axis=1)
    
    # Allocation
    ord_rho = np.argsort(rho)[::-1]  # Sort in descending order
    
    # Identification of local density peaks, sub-clusters, and center-association degree
    peaks = []
    phi = np.zeros(n)
    pn = np.zeros(n, dtype=int) - 1  # Parent nodes initialized to -1
    print(k)
    for i in range(n):
        point = ord_rho[i]
        all_neigh = knn[point, 1:k]
        all_neigh_dist = knn_dist[point, 1:k]
        big_neigh = all_neigh[rho[all_neigh] > rho[point]]
        
        bb = (rho[all_neigh] - rho[point]) / rho[all_neigh]
        dd = all_neigh_dist / (all_neigh_dist[-1] if len(all_neigh_dist) > 0 else 1)
        bb = bb[bb > 0]
        dd = dd[:len(bb)]  # Ensure dd and bb have the same length
        # 修改为（添加epsilon防止除以零）：
        epsilon = 1e-8
        bb = (bb - np.min(bb)) / (np.max(bb) - np.min(bb) + epsilon) if len(bb) > 0 else bb
        dd = (dd - np.min(dd)) / (np.max(dd) - np.min(dd) + epsilon) if len(dd) > 0 else dd
        bb_dd = alpha * bb + (1 - alpha) * dd
        
        if len(bb) > 0:
            best_big_index = np.argmin(bb_dd)
            best_big = big_neigh[best_big_index]
            neigh = best_big
        else:
            neigh = None
        
        if neigh is not None:
            phi[point] = (abs(rho[neigh] - rho[point]) / rho[neigh]) ** lambda_
            pn[point] = neigh
        else:
            phi[point] = 0
            peaks.append(point)
    
    # Density deviation cost to peak
    denisty_deviation_cost_to_peak = np.zeros(n)
    for i in range(n):
        if pn[ord_rho[i]] != -1:
            denisty_deviation_cost_to_peak[ord_rho[i]] = (
                denisty_deviation_cost_to_peak[pn[ord_rho[i]]] + phi[ord_rho[i]]
            )
        else:
            denisty_deviation_cost_to_peak[ord_rho[i]] = 0
    
    # Label initialization
    sub_l = np.full(n, -1)
    n_p = len(peaks)
    sub_l[peaks] = np.arange(1, n_p + 1)
    
    for i in range(n):
        if sub_l[ord_rho[i]] == -1:
            sub_l[ord_rho[i]] = sub_l[pn[ord_rho[i]]]
    
    # Edges matrix
    rho_peaks = rho[peaks]
    ord_rho_peaks = np.argsort(rho_peaks)[::-1]
    edges = np.full((n_p, n_p), np.inf)
    
    for i in range(n):
        BB = rho[i] * np.ones(k_b - 1)
        CC = rho[knn[i, 1:k_b]]
        denisty_deviation_cost_of_link_set = (np.abs(BB - CC) / np.maximum(BB, CC)) ** lambda_
        
        for j in range(1, k_b):
            jj = knn[i, j]
            AA = denisty_deviation_cost_to_peak[i] + denisty_deviation_cost_to_peak[jj]
            if sub_l[i] != sub_l[jj] and edges[sub_l[i] - 1, sub_l[jj] - 1] > AA:
                if i in knn[jj, 1:k]:
                    denisty_deviation_cost_of_link = denisty_deviation_cost_of_link_set[j - 1]
                    edges[sub_l[i] - 1, sub_l[jj] - 1] = AA + denisty_deviation_cost_of_link
                    edges[sub_l[jj] - 1, sub_l[i] - 1] = AA + denisty_deviation_cost_of_link
    
    # Peak Graph
    G = np.where(edges != np.inf, edges, 0)
    dim_matrix, _ = dijkstra(G, return_predecessors=True)
    delta_peaks = np.full(n_p, np.inf)
    pn_peaks = np.full(n_p, -1)
    
    for i in range(1, n_p):
        ii = ord_rho_peaks[i]
        for j in range(i):
            jj = ord_rho_peaks[j]
            if delta_peaks[ii] > dim_matrix[ii, jj]:
                delta_peaks[ii] = dim_matrix[ii, jj]
                pn_peaks[ii] = jj
    
    # Delta of peaks
    delta = np.zeros(n)
    delta[peaks] = delta_peaks
    if n_p > 1:
        must_c = np.sum(delta == np.inf)
        if must_c > 1:
            delta[delta == np.inf] = np.max(delta[delta != np.inf]) * 1.2
            delta_peaks = delta[peaks]
        else:
            delta[delta == np.inf] = np.max(delta[delta != np.inf]) * 1.2
            delta_peaks = delta[peaks]
    
    time1 = time.time() - start_time
    
    
    # Center confirm
    start_time = time.time()
    NC = 0
    cl_peaks = np.full(n_p, -1)
    icl = []

    rho_delta_product = rho_peaks * delta_peaks

    cluster_indices = np.argsort(rho_delta_product)[-num_cluster:][::-1]

    for idx in cluster_indices:
        NC += 1
        cl_peaks[idx] = NC
        icl.append(idx)

    
    for i in range(n_p):
        if cl_peaks[ord_rho_peaks[i]] == -1:
            cl_peaks[ord_rho_peaks[i]] = cl_peaks[pn_peaks[ord_rho_peaks[i]]]
    
    # Allocation
    CL = np.zeros(n, dtype=int)
    for i in range(n_p):
        CL[sub_l == (i + 1)] = cl_peaks[i]
    
    centers = [peaks[i] for i in icl]
    time2 = time.time() - start_time
    runtime = time1 + time2
    
    return CL, centers, runtime
import scipy.io as scio
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio
import time
from mdpc_original import mdpc_plus 
# if __name__=="__main__":
    # data1 = scio.loadmat(r"real_all/iris.mat")
    # data_keys = list(data1.keys())
    # data = data1[data_keys[-2]].astype(np.float32)
    # y_true = np.ravel(data1[data_keys[-1]])
    # n,d=data.shape

    # # 数据预处理
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(data)
    # n_clusters = len(np.unique(y_true))


    # labels, centers, runtime = mdpc_plus(data, k=None, num_cluster=n_clusters)

    # # 计算评估指标
    # from sklearn.metrics import adjusted_mutual_info_score
    # ami = adjusted_mutual_info_score(y_true, labels)

    # print(ami)