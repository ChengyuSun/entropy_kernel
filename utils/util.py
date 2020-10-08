import numpy as np
import os

def complete_path(folder, fname):
    return os.path.join(folder, fname)

def read_graph_label(dataset):
    filename='../data/{}/{}_graph_labels.txt'.format(dataset,dataset)
    print('reading data labels...')
    graph_labels = np.loadtxt(filename, dtype=np.float, delimiter=',')
    return graph_labels


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