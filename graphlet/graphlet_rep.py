import numpy as np
from graphlet.count_graphlet import graph_rep,graph_rep_concat
import os

GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def complete_path(folder, fname):
    return os.path.join(folder, fname)

def graph_reps():
    data = dict()
    dataset_name = 'MUTAG'
    dirpath = '../data/{}/unzipped/{}'.format(dataset_name, dataset_name)
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    graph_ids = set(data['_graph_indicator.txt'])

    node_label_num = max(data[NODE_LABELS_SUFFIX]) - min(data[NODE_LABELS_SUFFIX]) + 1
    print('node labels number: ', node_label_num)

    adj = data[ADJACENCY_SUFFIX]
    edge_index = 0
    node_index_begin = 0

    graph_reps_matrix=[]
    graph_reps_concat=np.array([])

    for g_id in set(graph_ids):
        #print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()

        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        #print('nodN: ', temp_nodN)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1

        #print(temp_A)

        node_labels = data[NODE_LABELS_SUFFIX][node_ids]
        #
        node_index_begin += temp_nodN

        #graph_reps_matrix.append(graph_rep(temp_A,node_labels,node_label_num))
        graph_reps_matrix.append(graph_rep_concat(temp_A,node_labels,node_label_num))

    return np.array(graph_reps_matrix),data[GRAPH_LABELS_SUFFIX]



