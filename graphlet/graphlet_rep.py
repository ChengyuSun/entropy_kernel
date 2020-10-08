# import os
# import sys
# rootpath=str("/home/scy/entropy_kernel")
# syspath=sys.path
# sys.path=[]
# sys.path.append(rootpath)
# sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[0]!="."])
# sys.path.extend(syspath)



import numpy as np
from graphlet.count_graphlet import graph_rep_sum,graphlet_matrix
from entropy.CountMotif_and_node import count_Motifs
from entropy.Entropy import graphEntropy
import os
from data.kPCA import rbf_kpca
import math
import utils.util as util

GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def graph_reps(dataset):

    data=util.read_data_txt(dataset)
    graph_ids = set(data['_graph_indicator.txt'])

    node_label_num = max(data[NODE_LABELS_SUFFIX]) - min(data[NODE_LABELS_SUFFIX]) + 1
    print('node labels number: ', node_label_num)

    adj = data[ADJACENCY_SUFFIX]
    edge_index = 0
    node_index_begin = 0

    graph_reps_matrix=[]


    f = open("../data/processed/{}_graphlet_count.txt".format(dataset), "w")
    graphlet_matrix_dataset=np.array([])

    for g_id in set(graph_ids):
        #print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()

        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1

        node_index_begin += temp_nodN

        if g_id==1:
            graphlet_matrix_dataset=graphlet_matrix(temp_A,temp_nodN)
        else:
            graphlet_matrix_dataset = np.append(graphlet_matrix_dataset,graphlet_matrix(temp_A,temp_nodN),axis=0)

    # graphlet_matrix_dataset=np.array(graphlet_matrix_dataset)
    print(graphlet_matrix_dataset.shape)

    return np.matrix(graph_reps_matrix)


graph_reps('MUTAG')