# encoding:utf-8
import numpy as np
import csv
from scipy.io import loadmat

np.set_printoptions(suppress=True)

m = loadmat("../data/PPIS/PPIs.mat")
data = m.get('PPIs')
for i in data:
    print(i)
    #todo
for i in range(86):
    matrix = data[0][i]
    #print('{}  matrix shape {}'.format(i,matrix.shape))
    #np.savetxt("./dataset/Cancer_data/UCEC_Graph/graph" + str(i + 1) + ".csv", matrix, fmt="%d", delimiter=",")


# for graph in range(0, 4127):
#     node_num = len(data[0][graph][2])
#     adj_matrix = np.zeros([node_num, node_num], np.int)
#     for line in range(0, node_num):
#         if len(data[0][graph][2][line][0]) == 0:   # 考虑该节点没有和其他节点相连
#             print graph, line
#             continue
#         else:
#             for edge in range(len(data[0][graph][2][line][0][0])):
#                 adj_matrix[line, data[0][graph][2][line][0][0][edge]-1] = 1
#     np.savetxt("./dataset/nci109/graph" + str(graph+1) + ".csv", adj_matrix, fmt="%d", delimiter=",")














