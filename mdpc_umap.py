# mdpc_umap.py
import umap
from mdpc_original import mdpc_plus
import os
os.environ['OMP_NESTED'] = 'FALSE'

# 修改mdpc_umap.py
def mdpc_plus_umap(data, k, num_cluster, dim):
    if data.shape[1] > dim:
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=50,        # 增大邻域范围
            min_dist=0.5,         # 缩小点间距
            spread=1.5,            # 控制簇间距
            metric='cosine',       # 改用余弦距离
            random_state=42
        )
        data = reducer.fit_transform(data)
    return mdpc_plus(data, k=None, num_cluster=num_cluster)  # 降低k值
