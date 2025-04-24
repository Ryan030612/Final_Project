import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.io as scio
import time
from mdpc_original import mdpc_plus
from mdpc_umap import mdpc_plus_umap
os.environ['OMP_NESTED'] = 'FALSE'
from evaluation import compute_score
import sys
import warnings

os.environ["OMP_NESTED"] = "FALSE"
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["OMP_WARNINGS"] = "FALSE"
# 过滤特定警告
warnings.filterwarnings("ignore", 
                       message="n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.")

def process_dataset(mat_path, k=None, output_file='results.txt'):
    """处理单个数据集的函数(已添加完整指标输出)"""
    # 加载数据
    data1 = scio.loadmat(mat_path)
    data_keys = list(data1.keys())
    data = data1[data_keys[-2]].astype(np.float32)
    y_true = np.ravel(data1[data_keys[-1]])
    n, d = data.shape
    
    # 跳过不符合条件的数据集
    if n < 1000 or n > 8000:
        return None
    if d < 20: 
        return None
    a = []
    b = []
    c = []
    # 数据预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    n_clusters = len(np.unique(y_true))
    
        # 打开输出文件
    with open(output_file, 'a') as f:
        # 重定向输出
        original_stdout = sys.stdout
        sys.stdout = f
        
        print(f"\n=== 处理数据集: {os.path.basename(mat_path)} ===")

        # 添加UMAP降维选择
        if d >= 20 and d <= 50:  # 20-50维数据集
            for d1 in range(2, 11):
                labels, centers, runtime = mdpc_plus_umap(data, k=None, num_cluster=n_clusters, dim=d1)
                
                ARI, NMI, ACC = compute_score(labels, y_true)  # score
                a.append(ARI)
                b.append(NMI)
                c.append(ACC)
                print('d: {d:.2f}\nARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(d=d1, a=ARI, b=NMI, c=ACC))
        elif d > 50:  # 50维以上数据集
            for d1 in range(2, 21):
                labels, centers, runtime = mdpc_plus_umap(data, k=None, num_cluster=n_clusters, dim=d1)
                
                ARI, NMI, ACC = compute_score(labels, y_true)  # score
                a.append(ARI)
                b.append(NMI)
                c.append(ACC)
                print('d: {d:.2f}\nARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(d=d1, a=ARI, b=NMI, c=ACC))
        a1 = np.argmax(a)
        b1 = np.argmax(b)
        c1 = np.argmax(c)
        print(np.argmax(a)+2,np.argmax(b)+2,np.argmax(c)+2)
        
        print('ARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(a=a[a1], b=b[b1], c=c[c1]))
        # 恢复标准输出
        sys.stdout = original_stdout
def batch_process_datasets(folder_path, output_file='results.txt'):
    """批量处理函数，输出到文件"""
    # 清空或创建输出文件
    with open(output_file, 'w') as f:
        f.write("=== 聚类实验输出结果 ===\n")
    
    for filename in sorted(os.listdir(folder_path))[137:]:
        if filename.endswith('.mat'):
            filepath = os.path.join(folder_path, filename)
            process_dataset(filepath, k=None, output_file=output_file)
            
if __name__ == "__main__":
    datasets_folder = './real_all'
    output_file = 'clustering_results1.txt'  # 输出文件名
    batch_process_datasets(datasets_folder, output_file)
    print(f"所有结果已保存到 {output_file}")