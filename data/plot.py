import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from graphlet.graphlet_rep import graph_reps,store_matrix,read_data,read_graph_label
from data.kPCA import rbf_kpca



graph_rep_matrix, graph_labels = graph_reps('MUTAG')
# graph_rep_matrix=read_data(filename='../data/processed/PTC_graphlet_count_pca.txt')
# graph_labels=read_graph_label('PTC_FR')
print('before pca shape: ', graph_rep_matrix.shape)


#data1 = rbf_kpca(graph_rep_matrix, gamma=15, k=3)
#data1 = pca(graph_rep_matrix, 3)
data1=graph_rep_matrix

print(' after pca shape: ', data1.shape)


fig = plt.figure()
ax = Axes3D(fig)


# y=[]
# for i in range(len(data1)):
#     if i>=86 and i<=92:
#         y.append(1)
#     elif i>=4021 and i <=4028:
#         y.append(3)
#     elif i>=5204 and i <=5212:
#         y.append(4)
#     elif i>=5378 and i <=5388:
#         y.append(5)
#     else:
#         y.append(0)
# y=np.array(y)
#
# ax.scatter(data1[y==1, 0], data1[y==1, 1], data1[y==1, 2], c='k', marker='^', label='Black Monday',s=40)
# ax.scatter(data1[y==3, 0], data1[y==3, 1], data1[y==3, 2], c='b', marker='^', label='Irac War',s=40)
# ax.scatter(data1[y==4, 0], data1[y==4, 1], data1[y==4, 2], c='r', marker='^', label='Subprime Mortgage Crisis',s=40)
# ax.scatter(data1[y==5, 0], data1[y==5, 1], data1[y==5, 2], c='m', marker='^', label='Bankrauptcy of Lehman Brothers',s=40)
# ax.scatter(data1[y==0, 0], data1[y==0, 1], data1[y==0, 2], c='c', marker='o', label='background', s=5,alpha=.1)
#

#
#
ax.scatter(data1[graph_labels==1, 0], data1[graph_labels==1, 1], data1[graph_labels==1, 2],
           c='r', marker='^',s=40)

ax.scatter(data1[graph_labels==-1, 0], data1[graph_labels==-1, 1], data1[graph_labels==-1, 2],
           c='b', marker='o',s=40)




# 绘制图例
ax.legend(loc='best')

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

# 展示
plt.show()
