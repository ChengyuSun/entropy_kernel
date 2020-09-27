import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from graphlet.graphlet_rep import graph_reps
from data.kPCA import rbf_kpca
# 数据


graph_rep_matrix,graph_labels=graph_reps()
print('before pca shape: ',graph_rep_matrix.shape)
data1 = rbf_kpca(graph_rep_matrix, gamma=15, k=3)
print(' after pca shape: ',data1.shape)

# 数据１
#data1 = np.arange(24).reshape((8, 3))
# data的值如下：
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]
#  [12 13 14]
#  [15 16 17]
#  [18 19 20]
#  [21 22 23]]
x1 = data1[:, 0]
y1 = data1[:, 1]
z1 = data1[:, 2]


# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)

colors=['r','k','b','w','y','c']

for index in range(len(x1)):
    ax.scatter(x1[index], y1[index], z1[index], c=colors[graph_labels[index]])
    #plt.scatter(x1[index], y1[index], c=colors[graph_labels[index]])


# 绘制图例
ax.legend(loc='best')

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

# 展示
plt.show()