import matplotlib.pyplot as plt
import numpy as np

# 示例数据 - 请替换为您的实际数据
dimensions = np.arange(2, 11)  # 维度从2到10
acc_scores = [0.85, 0.87, 0.89, 0.91, 0.90, 0.88, 0.87, 0.86, 0.85]  # 准确率
ari_scores = [0.78, 0.80, 0.82, 0.84, 0.83, 0.81, 0.80, 0.79, 0.78]  # 调整兰德指数
nmi_scores = [0.72, 0.74, 0.76, 0.78, 0.77, 0.75, 0.74, 0.73, 0.72]  # 标准化互信息

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制三条折线
plt.plot(dimensions, acc_scores, marker='o', label='ACC', linewidth=2)
plt.plot(dimensions, ari_scores, marker='s', label='ARI', linewidth=2)
plt.plot(dimensions, nmi_scores, marker='^', label='NMI', linewidth=2)

# 添加标题和标签
plt.title('Algorithm Robustness Across Different Dimensions', fontsize=14)
plt.xlabel('Number of Dimensions', fontsize=12)
plt.ylabel('Score', fontsize=12)

# 设置x轴刻度为整数
plt.xticks(dimensions)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()