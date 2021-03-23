import copy
import numpy as np
import utils.util as util
from entropy.Entropy import graphlet_entropy
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

    # graphlet_of_graph=np.sum(graphlet_of_nodes,axis=0)
    # graph_entropy=np.array(graphlet_entropy(graphlet_of_graph.tolist()))

    #graph_rep_2=np.array([])
    for temp_label in range(min_label,max_label+1):
        nodes_reps=graphlet_of_nodes[temp_node_labels==temp_label]
        if len(nodes_reps)==0:
            summation=np.zeros((1,dim))
        else:
            summation=np.sum(nodes_reps,axis=0).reshape(1,dim)

        #allocate(complex) graphlet entropy
        # temp_entropy=[]
        # for j in range(dim):
        #     if graphlet_of_graph[j]==0:
        #         temp_entropy.append(0)
        #     else:
        #         temp_entropy.append(graph_entropy[j]*summation[0][j]/graphlet_of_graph[j])


        #simple graphlet entropy
        temp_entropy = graphlet_entropy(summation[0])
        graph_rep = np.append(graph_rep, np.array(temp_entropy))

        #only graphlet count
        #graph_rep=np.append(graph_rep,np.array(summation[0]))

        #graph_rep_2 = np.append(graph_rep_2, np.array(summation[0]))


    #print(graph_rep_2)


    # log
    # graph_rep=graphlet_entropy(graph_rep.tolist())
    # for i in range(len(graph_rep)):
    #     graph_rep[i]=math.log(graph_rep[i]+1,10)

    #enhance
    # graph_rep = graphlet_entropy(graph_rep.tolist())
    # for i in range(len(graph_rep)):
    #     graph_rep[i]=graph_rep[i]*10

#  *2 80.125+-2.625  *3 83.75+-3.75  *4  83.25+-4.5 *5 83.875+-3.875
#  *6 84.125+-3.375  *7 84.625+-2.875  *8 84.25+-5.5 *9 84.75+-2.25  *9.5 82.625+-2.6
    #  *10 84.75+-4.75  *11  83.375+-4.125  *12 83.625+-3.625
# *15  83.125+-3.125   *20 82.875+-4.125
    # *100 84.25+-4.25  *1000 83.5+-4.75



    #print(graph_rep)
    # distribution
    # sum_entropy=sum(graph_rep)
    # for i in range(len(graph_rep)):
    #     graph_rep[i]=graph_rep[i]/sum_entropy
    return np.array(graph_rep)



GRAPH_LABELS_SUFFIX = '_graph_labels.txt'
NODE_LABELS_SUFFIX = '_node_labels.txt'
ADJACENCY_SUFFIX = '_A.txt'
GRAPH_ID_SUFFIX = '_graph_indicator.txt'


def dataset_reps(dataset):
    dataset_graph_reps = []

    if dataset == 'PPI' or dataset=='PTC':
        if dataset=='PPI':
            graphs,node_labels = util.read_ppi()
            N=86
        else:
            graphs, node_labels = util.read_ptc()
            N=344
        node_label_all=[]
        for i in node_labels:
            for j in i:
                node_label_all.append(j)
        min_label=min(node_label_all)
        max_label=max(node_label_all)
        print('minlabel {}  maxlabel {}'.format(min_label,max_label))
        for i in range(N):
            temp_A = graphs[i]
            temp_node_labels = node_labels[i]
            temp_nodN = len(temp_node_labels)
            temp_graph_rep = gen_graph_rep(temp_A, temp_nodN, temp_node_labels, min_label, max_label)
            dataset_graph_reps.append(temp_graph_rep)


    else:
        data=util.read_data_txt(dataset)
        graph_ids = set(data['_graph_indicator.txt'])
        min_label=min(data[NODE_LABELS_SUFFIX])
        max_label = max(data[NODE_LABELS_SUFFIX])
        node_label_num =  max_label-min_label + 1
        print('node labels number: ', node_label_num)

        adj = data[ADJACENCY_SUFFIX]
        edge_index = 0
        node_index_begin = 0



        for g_id in set(graph_ids):
            #print('正在处理图：' + str(g_id))
            node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
            node_ids.sort()

            temp_nodN = len(node_ids)
            temp_A = np.zeros([temp_nodN, temp_nodN], int)
            while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
                temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
                edge_index += 1

            temp_node_labels = data[NODE_LABELS_SUFFIX][node_index_begin:node_index_begin + temp_nodN]

            temp_graph_rep = gen_graph_rep(temp_A, temp_nodN, temp_node_labels, min_label, max_label)
            dataset_graph_reps.append(temp_graph_rep)

            node_index_begin += temp_nodN


    return np.array(dataset_graph_reps)


