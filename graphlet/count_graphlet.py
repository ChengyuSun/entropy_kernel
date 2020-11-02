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



# def graph_rep_sum(adj_original,node_labels,label_num):
#     coder=GraphletCoder(label_num)
#     N=len(adj_original)
#     rep_graph=np.zeros(0)
#     for start_index in range(N):
#         rep_node=graphlet_diffuse(start_index,adj_original,node_labels,coder)
#         if start_index==0:
#             rep_graph=rep_node
#         else:
#             rep_graph+=rep_node
#     return  rep_graph
#
# def graph_rep_concat(adj_original,node_labels,label_num):
#     #按照degree升序排列并反转为降序排列
#     degree_rank = np.argsort(sum(np.transpose(adj_original)))
#     coder = GraphletCoder(label_num)
#     degree_list=list(degree_rank)
#     degree_list.reverse()
#     degree_rank=np.array(degree_list)
#
#     rep_node_0=graphlet_diffuse(degree_rank[0], adj_original, node_labels, coder)
#     rep_graph = np.array([rep_node_0])
#     rep_node_len=len(rep_node_0)
#
#     for index in range(1,10):#将degree最大的十个节点表示concat 作为图表示
#         if index<len(degree_rank):
#             rep_node = graphlet_diffuse(degree_rank[index], adj_original, node_labels, coder)
#             rep_graph=np.append(rep_graph,rep_node)
#         else:
#             rep_graph = np.append(rep_graph, np.zeros(rep_node_len,int))
#     return rep_graph


def gen_graph_rep(adj_original,nodN,temp_node_labels,min_label,max_label):
    graphlet_of_nodes=[]
    graph_rep=np.array([])
    for index in range(nodN):
        graphlet_of_nodes.append(graphlet_diffuse_no_label(index,adj_original))

    graphlet_of_nodes=np.array(graphlet_of_nodes)

    _,dim=graphlet_of_nodes.shape

    graphlet_of_graph=np.sum(graphlet_of_nodes,axis=0)
    graph_entropy=np.array(graphlet_entropy(graphlet_of_graph.tolist()))

    #graph_rep_2=np.array([])
    for temp_label in range(min_label,max_label+1):
        nodes_reps=graphlet_of_nodes[temp_node_labels==temp_label]
        #print('label '+str(temp_label)+' has nodes '+str(nodes_reps.shape))
        summation=np.sum(nodes_reps,axis=0).reshape(1,dim)
        is_zero=True
        for i in range(dim):
            if summation[0][i] !=0:
                is_zero=False
                break
        if is_zero:
            summation=np.zeros(dim).reshape(1,dim)

        # temp_entropy=[]
        # for j in range(dim):
        #     if graphlet_of_graph[j]==0:
        #         temp_entropy.append(0)
        #     else:
        #         temp_entropy.append(graph_entropy[j]*summation[0][j]/graphlet_of_graph[j])

        temp_entropy = graphlet_entropy(summation[0])

        graph_rep = np.append(graph_rep, np.array(temp_entropy))
        #graph_rep=np.append(graph_rep,np.array(summation[0]))

        #graph_rep_2 = np.append(graph_rep_2, np.array(summation[0]))

    # print(graph_rep)
    # print(graph_rep_2)


    # log
    graph_rep=graphlet_entropy(graph_rep.tolist())
    for i in range(len(graph_rep)):
        graph_rep[i]=math.log(graph_rep[i]+1,10)

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

    data=util.read_data_txt(dataset)
    graph_ids = set(data['_graph_indicator.txt'])
    min_label=min(data[NODE_LABELS_SUFFIX])
    max_label = max(data[NODE_LABELS_SUFFIX])
    node_label_num =  max_label-min_label + 1
    print('node labels number: ', node_label_num)

    adj = data[ADJACENCY_SUFFIX]
    edge_index = 0
    node_index_begin = 0

    dataset_graph_reps=[]

    for g_id in set(graph_ids):
        #print('正在处理图：' + str(g_id))
        node_ids = np.argwhere(data['_graph_indicator.txt'] == g_id).squeeze()
        node_ids.sort()

        temp_nodN = len(node_ids)
        temp_A = np.zeros([temp_nodN, temp_nodN], int)
        while (edge_index < len(adj)) and (adj[edge_index][0] - 1 in node_ids):
            temp_A[adj[edge_index][0] - 1 - node_index_begin][adj[edge_index][1] - 1 - node_index_begin] = 1
            edge_index += 1

        temp_node_labels=data[NODE_LABELS_SUFFIX][node_index_begin:node_index_begin+temp_nodN]

        temp_graph_rep=gen_graph_rep(temp_A,temp_nodN,temp_node_labels,min_label,max_label)
        dataset_graph_reps.append(temp_graph_rep)

        node_index_begin += temp_nodN

    return np.array(dataset_graph_reps)


