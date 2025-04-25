import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mdpc_original import mdpc_plus
from mdpc_umap import mdpc_plus_umap
import umap

import time

def load_mat_data(mat_path):
    data1 = scio.loadmat("real_all/CANE.mat")
    data_keys = list(data1.keys())
    data = data1[data_keys[-2]].astype(np.float32)
    y_true = np.ravel(data1[data_keys[-1]])
    n, d = data.shape

  
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    n_clusters = len(np.unique(y_true))
    print(n,d,n_clusters)
    t1=time.time()


    labels_ours, centers, runtime = mdpc_plus_umap(data, k=None, num_cluster=n_clusters, dim=18)
    t2=time.time()
    labels_mdpc, centers, runtime1= mdpc_plus(data, k=None, num_cluster=n_clusters)
    t3=time.time()
    # print(t2-t1,t3-t2)
    print(runtime,runtime1)
    reducer = umap.UMAP(
            n_components=2,
            n_neighbors=50,        # 增大邻域范围
            min_dist=0.5,         # 缩小点间距
            spread=1.5,            # 控制簇间距
            metric='cosine',       # 改用余弦距离
            random_state=42
        )
    data1 = reducer.fit_transform(data)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data1 = scaler.fit_transform(data)
    
    data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
    
    vis_projection = data1
    

    # # 创建统一的颜色映射
    # unique_labels = np.unique(np.concatenate([y_true, labels_mdpc, labels_ours]))
    # cmap = plt.cm.get_cmap('Spectral', len(unique_labels))
    unique_labels = np.unique(np.concatenate([y_true, labels_mdpc, labels_ours]))
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('Spectral', n_clusters)  # 明确指定颜色数量
    
    # 创建图形
    plt.figure(figsize=(18, 6))
    
    # 子图1: 真实分布
    plt.subplot(1, 3, 1)
    for label in unique_labels:
        mask = (y_true == label)
        plt.scatter(vis_projection[mask, 0], vis_projection[mask, 1], 
                   color=cmap(label - min(unique_labels)), s=10, alpha=0.8, label=f'Cluster {label}')
    plt.title("True Distribution", fontsize=12)
    plt.legend()
    
    # 子图2: 原始MDPC
    plt.subplot(1, 3, 2)
    for label in unique_labels:
        mask = (labels_mdpc == label)
        plt.scatter(vis_projection[mask, 0], vis_projection[mask, 1], 
                   color=cmap(label - min(unique_labels)), s=10, alpha=0.8, label=f'Cluster {label}')
    plt.title("Distribution of MDPC+ Results", fontsize=12)
    plt.legend()
    
    # 子图3: UMAP+MDPC
    plt.subplot(1, 3, 3)
    for label in unique_labels:
        mask = (labels_ours == label)
        plt.scatter(vis_projection[mask, 0], vis_projection[mask, 1], 
                   color=cmap(label - min(unique_labels)), s=10, alpha=0.8, label=f'Cluster {label}')
    plt.title("Distribution of UMAP_MDPC+ Results", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result_CANE.svg")
      
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 配置参数
    mat_file = "real_all/CANE.mat"  # 替换为实际路径
    umap_dimension = 7  # 根据数据集特性调整
    
    # 运行测试
    load_mat_data(mat_file)