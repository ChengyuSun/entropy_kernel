import numpy as np
import os
import sys
import codecs
import pandas as pd
from sympy.core.evalf import N

def complete_path(folder, fname):
    return os.path.join(folder, fname)


def read_graph_label(dataset):
    print('reading data labels...')
    if dataset=='PPIs':
        return read_ppi_graph_label()
    elif dataset=='PTC':
        return read_ptc_labels()
    filename = '/new_disk_B/scy/{}/{}_graph_labels.txt'.format(dataset, dataset)
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
    dirpath = '/new_disk_B/scy/{}'.format(dataset_name)
    print('reading data {}...'.format(dataset_name))
    for f in os.listdir(dirpath):
        if "README" in f or 'graph_labels' in f or 'readme' in f or 'attributes' in f or 'edge_labels' in f:
            continue
        fpath = complete_path(dirpath, f)
        suffix = f.replace(dataset_name, '')
        print(suffix)
        if 'attributes' in suffix:
            data[suffix] = np.loadtxt(fpath, dtype=np.float, delimiter=',')
        else:
            data[suffix] = np.loadtxt(fpath, dtype=np.int, delimiter=',')

    return data



def acc_calculator(accs):
    n = len(accs)
    avg = sum(accs) / n
    max_dis = max(accs) - avg
    min_dis = avg - min(accs)
    distance=max(max_dis,min_dis)
    return avg,distance


from scipy.io import loadmat

def read_ppi():
    print('reading ppi')
    res_graph=[]
    res_node_label=[]
    np.set_printoptions(suppress=True)
    m = loadmat("/new_disk_B/scy/PPIS/PPIs.mat")
    data = m.get('PPIs')
    for i in range(86):
        res_graph.append(np.array(data[0][i][0].todense()).astype(int))
        print(len(res_graph[i]))
        print(res_graph[i])
        temp=[]
        for j in data[0][i][1][0][0][0]:
            temp.append(j[0])
        res_node_label.append(np.array(temp))
    return res_graph,res_node_label

def write_ppi():
    print('reading ppi')
    np.set_printoptions(suppress=True)
    m = loadmat("/new_disk_B/scy/PPIS/PPIs.mat")
    data = m.get('PPIs')
    f_A=open('/new_disk_B/scy/PPIS/PPIs_A.txt','w')
    f_indicator=open('/new_disk_B/scy/PPIS/PPIs_graph_indicator.txt','w')
    f_node_label=open('/new_disk_B/scy/PPIS/PPIs_node_labels.txt','w')
    node_labels=np.array([])
    edge_list=[]
    indicator=np.array([])
    node_start=1
    for i in range(86):
        temp_A=np.array(data[0][i][0].todense()).astype(int)
        temp_node_num=(len(temp_A))
        temp_node_labels=[]
        for j in data[0][i][1][0][0][0]:
            temp_node_labels.append(j[0])
        node_labels=np.append(node_labels,temp_node_labels)
        indicator=np.append(indicator,np.ones((temp_node_num),int)*i)
        for j in range(temp_node_num):
            for k in range(j+1,temp_node_num):
                if temp_A[j][k]==0:
                    continue
                edge='{},{}\n'.format(j+node_start,k+node_start)
                edge_list.append(edge)
        node_start+=temp_node_num
    node_labels=node_labels.astype(int)
    indicator=indicator.astype(int)
    print('ppi node num:{} and {}'.format(len(node_labels),len(indicator)))
    for node_idx in range(len(node_labels)):
        f_indicator.write(str(indicator[node_idx])+'\n')
        f_node_label.write(str(node_labels[node_idx])+'\n')
    print('ppi edge num:',len(edge_list))
    for edge in edge_list:
        f_A.write(edge)
    return
    



import csv
def read_ppi_graph_label():
    res=[]
    with open('/new_disk_B/scy/PPIs/PPI_label.csv')as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            res.append(1) if row[0]=='1' else res.append(0)
    return np.array(res)

def read_ptc():
    graphs=[]
    for i in range(1,345):
        with open('/new_disk_B/scy/ptc/graph_'+str(i)+'.csv')as f:
            graph=csv.reader(f)
            temp_graph = []
            for line in graph:
                temp_line=[]
                for item in line:
                    s=0 if item=='0' else 1
                    temp_line.append(s)
                temp_graph.append(temp_line)
            graphs.append(np.array(temp_graph))

    labels=[]
    for i in range(1, 345):
        with open('/new_disk_B/scy/ptc/feature_' + str(i) + '.csv')as f:
            fea = csv.reader(f)
            node_labels=[]
            for line in fea:
                temp_label = 0
                for item in line:
                    if item=='1':
                        break
                    temp_label+=1
                node_labels.append(temp_label)
            #print(node_labels)
            labels.append(np.array(node_labels))
    return graphs,labels

def write_ptc():
    print('reading ptc')
    f_A=open('/new_disk_B/scy/PTC/PTC_A.txt','w')
    f_indicator=open('/new_disk_B/scy/PTC/PTC_graph_indicator.txt','w')
    f_node_label=open('/new_disk_B/scy/PTC/PTC_node_labels.txt','w')
    node_labels=np.array([])
    edge_list=[]
    indicator=np.array([])
    node_start=1
    for i in range(1,345):
        temp_A=np.loadtxt('/new_disk_B/scy/ptc/graph_'+str(i)+'.csv', dtype=np.int, delimiter=',')
        temp_feature=np.loadtxt('/new_disk_B/scy/ptc/graph_'+str(i)+'.csv', dtype=np.int, delimiter=',')
        temp_node_labels = np.array([np.argmax(one_hot) for one_hot in temp_feature])
        node_labels=np.append(node_labels,temp_node_labels)
        temp_node_num=(len(temp_A))
        indicator=np.append(indicator,np.ones((temp_node_num),int)*i)
        for j in range(temp_node_num):
            for k in range(j+1,temp_node_num):
                if temp_A[j][k]==0:
                    continue
                edge='{},{}\n'.format(j+node_start,k+node_start)
                edge_list.append(edge)
        node_start+=temp_node_num
    node_labels=node_labels.astype(int)
    indicator=indicator.astype(int)
    print('ptc node num:{} and {}'.format(len(node_labels),len(indicator)))
    for node_idx in range(len(node_labels)):
        f_indicator.write(str(indicator[node_idx])+'\n')
        f_node_label.write(str(node_labels[node_idx])+'\n')
    print('ptc edge num:',len(edge_list))
    for edge in edge_list:
        f_A.write(edge)
    return
        


def read_ptc_labels():
    labels = []
    with open('/new_disk_B/scy/ptc/ptc_label.csv')as f2:
        l = csv.reader(f2)
        for row in l:
            labels.append(1) if row[0] == '1' else labels.append(0)
    return np.array(labels)

def xlsx_to_csv_pd(filename):
    data_xls = pd.read_excel('{}.xlsx'.format(filename),header=None,names=None)
    data_xls.to_csv('{}.csv'.format(filename), encoding='utf-8',header=0,index=0)

def read_data_2():
    # A=[]
    # node_labels=[]
    # graph_labels=[]
    # indicator=[]
    # node_start_idx=0
    folder='/new_disk_B/scy/u_sto_net/'
    # with open('/home/scy/TL-GNN/GIN_dataset/Synthetic.txt', 'r') as f:
    #     n_g = int(f.readline().strip())
    #     for i in range(n_g):
    #         row = f.readline().strip('\n').split(' ')
    #         temp_nodN, temp_graph_label = [int(w) for w in row]
    #         graph_labels.append([temp_graph_label])
            
    #         for node_idx in range(temp_nodN):
    #             row = f.readline().strip().split()
    #             node_labels.append([int(row[0])])
    #             indicator.append([i])
    #             for k in range(2, len(row)):
    #                 A.append([node_idx+node_start_idx,int(row[k])+node_start_idx])
                    
    #         node_start_idx+=temp_nodN
    for graph_id in range(1,5976+1):
        print('graph_id:',graph_id)
        filename=folder+'graph{}'.format(graph_id)
        if not os.path.exists(filename+'.csv'):
            xlsx_to_csv_pd(filename=filename)
        temp_A=np.loadtxt(filename+'.csv',dtype=np.int,delimiter=',')
        temp_nodN=347
        A=[]
        f_A=open('/new_disk_B/scy/financial/financial_A_{}.txt'.format(graph_id),"w")
        # f_indicator=open('/new_disk_B/scy/financial/financial_graph_indicator.txt','a')
        for i in range(temp_nodN):
            for j in range(temp_nodN):
                if temp_A[i][j]==1:
                    A.append([i,j])
                    # indicator.append([graph_id])
        np.savetxt(f_A,np.array(A),fmt='%d',delimiter=',')
        # np.savetxt(f_indicator,np.array(indicator),fmt='%d',delimiter=',')
        f_A.close()
        # f_indicator.close()