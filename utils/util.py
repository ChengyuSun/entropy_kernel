import numpy as np
import os
import json
from networkx.readwrite import json_graph


def complete_path(folder, fname):
    return os.path.join(folder, fname)


def read_graph_label(dataset):
    filename = '../data/{}/{}_graph_labels.txt'.format(dataset, dataset)
    print('reading data labels...')
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


def load_data_ppi(prefix='../data/ppi/ppi', normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


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
