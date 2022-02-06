import sys
# print(sys.path)
sys.path.append('../utis')
sys.path.append('../graphlet')
from numpy.lib.function_base import gradient

import copy
from os import linesep
import numpy as np
import utils.util as util
from entropy.Entropy import graphlet_entropy
from graphlet.count_motif import allocate
import math

class GraphletCoder:
    label_number=0
    graphlet_index_2 =0
    graphlet_index_3_1 = 0
    graphlet_index_3_2 = 0
    graphlet_index_3_3 = 0
    graphlet_index_4_1 = 0
    graphlet_index_4_2 = 0
    graphlet_index_4_3 = 0
    graphlet_index_4_4 = 0
    graphlet_types = 0

    def __init__(self, label_num):
        self.graphlet_index_2 = pow(label_num, 2)
        self.graphlet_index_3_1 = self.graphlet_index_2 + pow(label_num, 3)
        self.graphlet_index_3_2 = self.graphlet_index_3_1 + pow(label_num, 3)
        self.graphlet_index_3_3 = self.graphlet_index_3_2 + pow(label_num, 3)
        self.graphlet_index_4_1 = self.graphlet_index_3_3 + pow(label_num, 4)
        self.graphlet_index_4_2 = self.graphlet_index_4_1 + pow(label_num, 4)
        self.graphlet_index_4_3 = self.graphlet_index_4_2 + pow(label_num, 4)
        self.graphlet_index_4_4 = self.graphlet_index_4_3 + pow(label_num, 4)
        self.graphlet_types = self.graphlet_index_4_4 - 1
        #print('all graphlet type is: ',self.graphlet_types)
        self.graphlet_type_vactor = [0,self.graphlet_index_2, self.graphlet_index_3_1, self.graphlet_index_3_2,
                                     self.graphlet_index_3_3,
                                     self.graphlet_index_4_1, self.graphlet_index_4_2, self.graphlet_index_4_3]

        self.label_number=label_num
        return

    def code(self,number_vactor,graphlet_type):
        res = 0
        #print('find '+str(number_vactor)+' of type '+str(graphlet_type))
        number_vactor.reverse()
        for index, value in enumerate(number_vactor):
            res += value * pow(self.label_number, index)

        r=res+self.graphlet_type_vactor[graphlet_type]

        # print('index is '+str(r))
        return r

    def get_graphlet_types(self):
        return self.graphlet_types



def graphlet_diffuse(start_index,adj_original,node_labels,graphlet_coder):

    node_rep=[0 for i in range(graphlet_coder.get_graphlet_types())]

    neighbors_1=np.nonzero(adj_original[start_index])[0].tolist()
    for index_1_1,n1_1 in enumerate(neighbors_1):
        #2
        number_vactor_2=[node_labels[start_index],node_labels[n1_1]]
        node_rep[graphlet_coder.code(number_vactor_2,0)]+=1

        neighbors_2=np.nonzero(adj_original[n1_1])[0].tolist()
        if start_index in neighbors_2:
            neighbors_2.remove(start_index)
        for index_2_1,n2_3 in enumerate(neighbors_2):
            #3_1
            if adj_original[start_index][n2_3]==0:
                number_vactor_3_1 = [node_labels[start_index], node_labels[n1_1], node_labels[n2_3]]
                node_rep[graphlet_coder.code(number_vactor_3_1, 1)] += 1
                for n2_4 in neighbors_2[index_2_1+1:]:
                    #4_2
                    number_vactor_4_2 = [node_labels[start_index], node_labels[n1_1], node_labels[n2_3],node_labels[n2_4]]
                    node_rep[graphlet_coder.code(number_vactor_4_2, 5)] += 1

            neighbors_3=np.nonzero(adj_original[n2_3])[0].tolist()
            if start_index in neighbors_3:
                neighbors_3.remove(start_index)
            if n1_1 in neighbors_3:
                neighbors_3.remove(n1_1)
            for n3_1 in neighbors_3:
                #4_1
                number_vactor_4_1 = [node_labels[start_index], node_labels[n1_1], node_labels[n2_3], node_labels[n3_1]]
                node_rep[graphlet_coder.code(number_vactor_4_1, 4)] += 1


        for index_1_2,n1_2 in enumerate(neighbors_1[index_1_1+1:]):
            #3_3
            if adj_original[n1_1][n1_2]==1:
                number_vactor_3_3=[node_labels[start_index],node_labels[n1_1],node_labels[n1_2]]
                node_rep[graphlet_coder.code(number_vactor_3_3,3)]+=1
            else :
                #3_2
                number_vactor_3_2=[node_labels[start_index],node_labels[n1_1],node_labels[n1_2]]
                node_rep[graphlet_coder.code(number_vactor_3_2, 2)] += 1
                #4_3
                for n1_3 in neighbors_1[index_1_2+index_1_1 + 2:]:
                    if adj_original[n1_1][n1_3]==0 and adj_original[n1_2][n1_3]==0:
                        number_vactor_4_3 = [node_labels[start_index], node_labels[n1_1], node_labels[n1_2], node_labels[n1_3]]
                        node_rep[graphlet_coder.code(number_vactor_4_3, 6)] += 1
                #4_4
                for n2_1 in np.nonzero(adj_original[n1_1])[0]:
                    if n2_1!=start_index and n2_1!=n1_2 and \
                            adj_original[n1_2][n2_1]==0 and adj_original[start_index][n2_1]==0 :
                        number_vactor_4_4 = [node_labels[start_index], node_labels[n1_1], node_labels[n1_2], node_labels[n2_1]]
                        node_rep[graphlet_coder.code(number_vactor_4_4, 7)] += 1
                for n2_2 in np.nonzero(adj_original[n1_2])[0]:
                    if n2_2 != start_index and n2_2 != n1_1 and\
                            adj_original[n1_1][n2_2]==0 and adj_original[start_index][n2_2]==0 :
                        number_vactor_4_4 = [node_labels[start_index], node_labels[n1_2], node_labels[n1_1], node_labels[n2_2]]
                        node_rep[graphlet_coder.code(number_vactor_4_4, 7)] += 1
    return np.array(node_rep)

def graphlet_diffuse_no_label(start_index,adj_original):
    node_rep=[0 for i in range(8)]

    neighbors_1=np.nonzero(adj_original[start_index])[0].tolist()
    for index_1_1,n1_1 in enumerate(neighbors_1):
        #2
        node_rep[0]+=1
        neighbors_2=np.nonzero(adj_original[n1_1])[0].tolist()
        if start_index in neighbors_2:
            neighbors_2.remove(start_index)
        for index_2_1,n2_3 in enumerate(neighbors_2):

            if adj_original[start_index][n2_3]==0:
                #3_1
                node_rep[1] += 1
                #4_2
                node_rep[5] +=len(neighbors_2[index_2_1+1:])

            neighbors_3=np.nonzero(adj_original[n2_3])[0].tolist()
            if start_index in neighbors_3:
                neighbors_3.remove(start_index)
            if n1_1 in neighbors_3:
                neighbors_3.remove(n1_1)
             # 4_1
            node_rep[4] += len(neighbors_3)



        for index_1_2,n1_2 in enumerate(neighbors_1[index_1_1+1:]):
            #3_3
            if adj_original[n1_1][n1_2]==1:
                node_rep[3]+=1
            else :
                #3_2
                node_rep[2] += 1
                #4_3
                for n1_3 in neighbors_1[index_1_2+index_1_1 + 2:]:
                    if adj_original[n1_1][n1_3]==0 and adj_original[n1_2][n1_3]==0:
                        node_rep[6] += 1
                #4_4
                for n2_1 in np.nonzero(adj_original[n1_1])[0]:
                    if n2_1!=start_index and n2_1!=n1_2 and \
                            adj_original[n1_2][n2_1]==0 and adj_original[start_index][n2_1]==0 :
                        node_rep[7] += 1
                for n2_2 in np.nonzero(adj_original[n1_2])[0]:
                    if n2_2 != start_index and n2_2 != n1_1 and\
                            adj_original[n1_1][n2_2]==0 and adj_original[start_index][n2_2]==0 :
                        node_rep[7] += 1
    return np.array(node_rep)

def gen_graph_rep(adj_original,nodN,temp_node_labels,min_label,max_label):
    graphlet_of_nodes=[]
    graph_rep=np.array([])
    for index in range(nodN):
        graphlet_of_nodes.append(graphlet_diffuse_no_label(index,adj_original))

    graphlet_of_nodes=np.array(graphlet_of_nodes)
    _,dim=graphlet_of_nodes.shape

    for temp_label in range(min_label,max_label+1):
        nodes_reps=graphlet_of_nodes[temp_node_labels==temp_label]
        if len(nodes_reps)==0:
            summation=np.zeros((1,dim))
        else:
            summation=np.sum(nodes_reps,axis=0).reshape(1,dim)
        temp_entropy = graphlet_entropy(summation[0])
        graph_rep = np.append(graph_rep, np.array(temp_entropy))


    # log
    #log10-84 2-85
    graph_rep=graphlet_entropy(graph_rep.tolist())
    for i in range(len(graph_rep)):
        graph_rep[i]=math.log(graph_rep[i]+1,3)

    #enhance
    # graph_rep = graphlet_entropy(graph_rep.tolist())
    # for i in range(len(graph_rep)):
    #     graph_rep[i]=graph_rep[i]*10

    return np.array(graph_rep)


def graph_representation(node_in_graphlet,node_labels,min_label,max_label,kt,r,log_value,graphlet_normalize):
    graph_rep=np.array([])
    node_num,graphlet_type=node_in_graphlet.shape
    for label in range(min_label,max_label+1):
        nodes_reps=node_in_graphlet[node_labels==label]
        if len(nodes_reps)==0:
            summation=np.zeros((graphlet_type),int)
        else:
            summation=np.sum(nodes_reps,axis=0)
        
        if graphlet_normalize:
            if not sum(summation)==0:
                summation=[i/sum(summation) for i in summation]
    
        temp_entropy = graphlet_entropy(summation,kt,r)
        for i in range(len(temp_entropy)):
            if temp_entropy[i]>0:
                temp_entropy[i]=math.log(temp_entropy[i],log_value) 
            else:
                temp_entropy[i]=0
        graph_rep = np.append(graph_rep, np.array(temp_entropy))
    
        # graph_rep[i]=
    return graph_rep


GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def dataset_reps(dataset,kt,r,log_value,graphlet_normalize):
    dataset_graph_reps = []
    data=util.read_data_txt(dataset)
    graph_ids = set(data['_graph_indicator.txt'])
    min_label=min(data[NODE_LABELS_SUFFIX])
    max_label = max(data[NODE_LABELS_SUFFIX])

    node_label_num =  max_label-min_label + 1
    print('node labels number: ', node_label_num)

    adj = data[ADJACENCY_SUFFIX]
    edge_in_graphlet=data['_edge_in_graphlet.txt'] #PGD algorithm 
    edge_index_1=edge_index_2 = 0
    node_index_begin= 0
    graphlet_type_PGD=len(edge_in_graphlet[0])-2
    for g_id in set(graph_ids):
        print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()
        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int) #init temp Adj
        temp_node_in_graphlet=np.zeros((temp_nodN,graphlet_type_PGD),int)
        if dataset=='PROTEINS' or dataset=='NCI1':
            if dataset=='NCI1':
                edge_in_graphlet = np.loadtxt('/new_disk_B/scy/NCI1_individual_graphlet/NCI1_graphlet_{}.txt'.format(g_id), dtype=np.int, delimiter=',')
            else:
                edge_in_graphlet = np.loadtxt('/new_disk_B/scy/delete/proteins_file/PROTEINS_graphlet_{}.txt'.format(g_id), dtype=np.int, delimiter=',')
            for line in edge_in_graphlet:
                graphlet_of_edge=line[2:]
                temp_node_in_graphlet[line[0]-1]+=graphlet_of_edge
                temp_node_in_graphlet[line[1]-1]+=graphlet_of_edge

        else:
            while (edge_index_1 < len(edge_in_graphlet)) and (edge_in_graphlet[edge_index_1][0] - 1 in node_ids):
                line=edge_in_graphlet[edge_index_1]
                edge_index_1 += 1
                if line[1]-1 not in node_ids:
                    continue
                graphlet_of_edge=line[2:]
                temp_node_in_graphlet[line[0]-1-node_index_begin]+=graphlet_of_edge
                temp_node_in_graphlet[line[1]-1-node_index_begin]+=graphlet_of_edge

        while (edge_index_2<len(adj)) and (adj[edge_index_2][0]-1 in node_ids):
            e1=adj[edge_index_2][0]-1-node_index_begin
            e2=adj[edge_index_2][1]-1-node_index_begin
            temp_A[e1][e2]=temp_A[e2][e1]=1
            edge_index_2+=1
        
        node_in_motif_ax=allocate(temp_A)
        temp_node_in_graphlet=np.append(temp_node_in_graphlet,node_in_motif_ax,axis=1)
        temp_node_labels = data[NODE_LABELS_SUFFIX][node_index_begin:node_index_begin + temp_nodN]
        temp_graph_rep=graph_representation(temp_node_in_graphlet,temp_node_labels,min_label,max_label,kt,r,log_value,graphlet_normalize)
        dataset_graph_reps.append(temp_graph_rep)
        node_index_begin += temp_nodN


    return np.array(dataset_graph_reps)

def dataset_reps_financial():
    graph_num=5976
    node_num=347
    # graphlet_type_PGD=8
    res=[]
    for graph_id in range(1,graph_num+1):
        print('graph id:',graph_id)
        # temp_A=np.loadtxt('/new_disk_B/scy/financial/financial_A_{}.txt'.format(graph_id),dtype=np.int,delimiter=',')
        edge_in_graphlet = np.loadtxt('/new_disk_B/scy/financial/financial_graphlet_{}.txt'.format(graph_id), dtype=np.int, delimiter=',')
        temp_graph_rep=np.sum(edge_in_graphlet,0)[2:]
        res.append(temp_graph_rep)
    np.savetxt('/new_disk_B/scy/financial/graph_reps.txt',np.array(res),fmt='%d')
       
        
    