# encoding:utf-8
import numpy as np
import csv
from scipy.io import loadmat
import scipy.sparse as sp
def read_ppi():
    res_graph=[]
    res_node_label=[]
    np.set_printoptions(suppress=True)
    m = loadmat("../data/PPIS/PPIs.mat")
    data = m.get('PPIs')
    for i in range(86):
        res_graph.append(np.array(data[0][i][0].todense()).astype(int))
        print(np.array(data[0][i][0].todense()).astype(int))
        temp=[]
        for j in data[0][i][1][0][0][0]:
            temp.append(j[0])
        res_node_label.append(temp)
    return res_graph,res_node_label
    #np.savetxt("./dataset/Cancer_data/UCEC_Graph/graph" + str(i + 1) + ".csv", matrix, fmt="%d", delimiter=",")
read_ppi()

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














