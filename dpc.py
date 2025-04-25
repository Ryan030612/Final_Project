import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
from evaluation import compute_score
def calculate_acc(y_true, y_pred):
    """计算聚类准确率(ACC) - 修复版本"""
    # 确保标签从0开始连续编号
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # 重新映射标签到0开始的连续整数
    true_remap = {old: new for new, old in enumerate(unique_true)}
    pred_remap = {old: new for new, old in enumerate(unique_pred)}
    
    y_true_remap = np.array([true_remap[x] for x in y_true])
    y_pred_remap = np.array([pred_remap[x] for x in y_pred])
    
    # 计算混淆矩阵
    n_classes = max(len(unique_true), len(unique_pred))
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for i in range(len(y_true_remap)):
        confusion_matrix[y_true_remap[i], y_pred_remap[i]] += 1
    
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    return confusion_matrix[row_ind, col_ind].sum() / len(y_true)

def dpc_cluster(data, dc_percent=2.0, n_clusters=None):
    """
    DPC密度峰值聚类核心算法
    参数:
        data: 输入数据 (n_samples, n_features)
        dc_percent: 截断距离百分比 (0-100)
        n_clusters: 指定聚类数,若为None则自动确定
    返回:
        labels: 聚类标签
        centers: 聚类中心索引
        runtime: 运行时间
    """
    start_time = time.time()
    
    # 1. 计算距离矩阵
    dists = squareform(pdist(data))
    
    # 2. 自动确定截断距离
    dc = np.percentile(dists, dc_percent)
    
    # 3. 计算局部密度 (高斯核)
    rho = np.sum(np.exp(-(dists/dc)**2), axis=1) - 1
    
    # 4. 计算相对距离
    deltas = np.zeros(len(rho))
    nearest_neighbor = np.zeros(len(rho), dtype=int)
    
    # 按密度降序排序
    rho_order = np.argsort(-rho)
    
    for i, idx in enumerate(rho_order):
        if i == 0:
            deltas[idx] = np.max(dists[idx])
            nearest_neighbor[idx] = -1
            continue
            
        higher_rho_indices = rho_order[:i]
        deltas[idx] = np.min(dists[idx, higher_rho_indices])
        nearest_neighbor[idx] = higher_rho_indices[np.argmin(dists[idx, higher_rho_indices])]
    
    # 5. 确定聚类中心
    if n_clusters is None:
        # 自动确定中心 (rho*delta乘积最大的点)
        rho_delta = rho * deltas
        centers = np.argsort(-rho_delta)[:int(np.sqrt(len(rho)))]  # 启发式选择中心数
    else:
        # 手动指定聚类数
        centers = np.argsort(-rho * deltas)[:n_clusters]
    
    # 6. 分配标签
    labels = -np.ones(len(rho), dtype=int)
    for cluster_id, center in enumerate(centers):
        labels[center] = cluster_id
    
    for idx in rho_order:
        if labels[idx] == -1:
            labels[idx] = labels[nearest_neighbor[idx]]
    
    runtime = time.time() - start_time
    return labels, centers, runtime

def plot_dpc_results(data, labels, centers, rho, deltas):
    """可视化DPC聚类结果"""
    plt.figure(figsize=(15, 5))
    
    # 决策图
    plt.subplot(131)
    plt.scatter(rho, deltas, c='b', s=10)
    plt.scatter(rho[centers], deltas[centers], c='r', marker='*', s=200)
    plt.xlabel('Density (ρ)')
    plt.ylabel('Distance (δ)')
    plt.title('Decision Graph')
    
    # 前两维的聚类结果
    plt.subplot(132)
    for k in range(len(np.unique(labels))):
        plt.scatter(data[labels==k, 0], data[labels==k, 1], s=10, label=f'Cluster {k+1}')
    plt.scatter(data[centers, 0], data[centers, 1], c='k', marker='x', s=100)
    plt.title('First Two Dimensions')
    plt.legend()
    
    # 随机选择两个维度的聚类结果
    if data.shape[1] >= 4:
        dim1, dim2 = np.random.choice(data.shape[1], 2, replace=False)
        plt.subplot(133)
        for k in range(len(np.unique(labels))):
            plt.scatter(data[labels==k, dim1], data[labels==k, dim2], s=10, label=f'Cluster {k+1}')
        plt.scatter(data[centers, dim1], data[centers, dim2], c='k', marker='x', s=100)
        plt.title(f'Dimensions {dim1+1} & {dim2+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 加载.mat数据
    mat_path = "real_all/waveform.mat"  # 替换为您的.mat文件路径
    mat_data = scio.loadmat(mat_path)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    data = mat_data[data_keys[0]].astype(np.float32)
    y_true = np.ravel(mat_data[data_keys[1]])
    y_true = y_true - np.min(y_true)  # 如果标签不是从0开始
    # 2. 数据预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    n_clusters = len(np.unique(y_true))
    a = []
    b = []
    c = []
    for dc in np.arange(0.1,2.1,0.1):
    # 3. 运行DPC聚类
        labels, centers, runtime = dpc_cluster(data, dc_percent=dc, n_clusters=n_clusters)
        
       
        ARI, NMI, ACC = compute_score(labels, y_true)  # score
        a.append(ARI)
        b.append(NMI)
        c.append(ACC)
        print('dc: {d:.2f}\nARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(d=dc, a=ARI, b=NMI, c=ACC))
    a1 = np.argmax(a)
    b1 = np.argmax(b)
    c1 = np.argmax(c)
    print(np.argmax(a)+1,np.argmax(b)+1,np.argmax(c)+1)
    print('ARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(a=a[a1], b=b[b1], c=c[c1]))
