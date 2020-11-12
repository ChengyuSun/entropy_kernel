import numpy as np
import os
import json
from networkx.readwrite import json_graph


def complete_path(folder, fname):
    return os.path.join(folder, fname)


def read_graph_label(dataset):
    print('reading data labels...')
    if dataset=='PPI':
        return read_ppi_graph_label()
    filename = '../data/{}/{}_graph_labels.txt'.format(dataset, dataset)
    graph_labels = np.loadtxt(filename, dtype=np.float, delimiter=',')
    return graph_labels


def read_adjMatrix(file_index):
    array = open(
        'D:/PycharmProjects/motif/undirected-motif-entropy/data/graph' + str(file_index + 1) + '.csv').readlines()
    N = len(array)
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [int(float(x)) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix, N


def store_matrix(matrix, filename):
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


def read_data_txt(dataset):
    data = dict()
    dataset_name = str(dataset)
    # dirpath = '/new_disk_B/scy/{}'.format(dataset_name)
    dirpath = '../data/{}'.format(dataset_name)
    print('reading data...')
    for f in os.listdir(dirpath):
        if "README" in f or '.txt' not in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        # print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    return data

def acc_calculator(accs):
    n = len(accs)
    print('acc number: ', n)
    avg = sum(accs) / n
    print('avg: ', avg)
    print('max: ', max(accs))
    print('min: ', min(accs))
    max_dis = max(accs) - avg
    min_dis = avg - min(accs)
    if max_dis > min_dis:
        print('distance:', max_dis)
    else:
        print('distance:', min_dis)
    return avg


from scipy.io import loadmat

def read_ppi():
    print('reading ppi')
    res_graph=[]
    res_node_label=[]
    np.set_printoptions(suppress=True)
    m = loadmat("../data/PPIS/PPIs.mat")
    data = m.get('PPIs')
    for i in range(86):
        res_graph.append(np.array(data[0][i][0].todense()).astype(int))
        temp=[]
        for j in data[0][i][1][0][0][0]:
            temp.append(j[0])
        res_node_label.append(np.array(temp))
    return res_graph,res_node_label



import csv
def read_ppi_graph_label():
    res=[]
    with open('../data/PPIS/PPI_label.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            res.append(1) if row[0]=='1' else res.append(0)
    return np.array(res)