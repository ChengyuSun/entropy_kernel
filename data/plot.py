import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from graphlet.graphlet_rep import graph_reps
from data.kPCA import rbf_kpca, pca

# 数据

def read_data():
    array = open('../data/finacial/E_256_27_1000.csv').readlines()
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix


#graph_rep_matrix, graph_labels = graph_reps('NCI1')
graph_rep_matrix=read_data()
print('before pca shape: ', graph_rep_matrix.shape)
data1 = rbf_kpca(graph_rep_matrix, gamma=15, k=3)
#data1 = pca(graph_rep_matrix, 3)
print(' after pca shape: ', data1.shape)

x1 = data1[:, 0]
y1 = data1[:, 1]
z1 = data1[:, 2]
#z1=[0 for i in range(len(data1))]
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)

colors = ['r', 'k', 'b', 'y', 'c', 'w']

for i in range(len(x1)):
    if i>=86 and i<=92:
        ci=0
    elif i>=4021 and i <=4028:
        ci=1
    elif i>=5204 and i <=5212:
        ci=2
    elif i>=5378 and i <=5388:
        ci=3
    else:
        ci=4
    #ax.scatter(x1[i], y1[i], z1[i], c=colors[graph_labels[index]])
    ax.scatter(x1[i], y1[i], z1[i], c=colors[ci])
    # plt.scatter(x1[index], y1[index], c=colors[graph_labels[index]])

# 绘制图例
ax.legend(loc='best')

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

# 展示
plt.show()
