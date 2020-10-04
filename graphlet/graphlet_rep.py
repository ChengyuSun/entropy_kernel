import numpy as np
from graphlet.count_graphlet import graph_rep_sum,graph_rep_concat
from entropy.CountMotif_and_node import count_Motifs
from entropy.Entropy import graphEntropy
import os
from data.kPCA import rbf_kpca
import math

GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def complete_path(folder, fname):
    return os.path.join(folder, fname)

def read_graph_label(dataset):
    filename='../data/{}/{}_graph_labels.txt'.format(dataset,dataset)
    print('reading data labels...')
    graph_labels = np.loadtxt(filename, dtype=np.float, delimiter=',')
    return graph_labels

def graph_reps(dataset):
    data = dict()
    dataset_name = str(dataset)
    dirpath = '/new_disk_B/scy/{}'.format(dataset_name)
    print('reading data...')
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        #print(suffix)
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

    f_NCI1 = open("/new_disk_B/scy/processed/NCI1_graphlet_count_concat.txt", "w")
    for g_id in set(graph_ids):
        print('正在处理图：' + str(g_id))
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

        #graph_reps_matrix.append(graph_rep_sum(temp_A,node_labels,node_label_num))
        #graph_reps_matrix.append(graph_rep_concat(temp_A,node_labels,node_label_num))

        f_NCI1.write(str(graph_rep_concat(temp_A,node_labels,node_label_num)))

        # motif_count,_=count_Motifs(temp_A)
        # motif_entropy=graphEntropy(motif_count,temp_nodN)
        # graph_reps_matrix.append(motif_entropy)


    return np.array(graph_reps_matrix),data[GRAPH_LABELS_SUFFIX]


def read_adjMatrix(file_index):
    array = open('D:/PycharmProjects/motif/undirected-motif-entropy/data/graph'+str(file_index+1)+'.csv').readlines()
    N = len(array)
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [int(float(x)) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix,N

def store_count_and_entropy():
    f1=open("../data/processed/motif_count.txt", "w")
    f2=open("../data/processed/graph_entropy.txt", "w")
    for file_index in range(5976):
        print(file_index)
        matrix,nodN=read_adjMatrix(file_index)
        motif_count, _ = count_Motifs(matrix)
        f1.write(str(motif_count)+'\n')
        graph_entropy = graphEntropy(motif_count, nodN)
        f2.write(str(graph_entropy)+'\n')

def store_matrix(matrix,filename):
    f1 = open(filename, "w")
    for line in matrix:
        for item in line:
            f1.write(str(item) + ',')
        f1.write('\n')


def read_data(filename):
    array = open(filename).readlines()
    matrix = []
    for line in array:
        line = line.strip('\r\n[],').split(',')
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix

graph_reps('NCI1')

