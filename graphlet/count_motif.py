import numpy as np
SCALE=6
BASE=2
# tree_deg_thresh_low=3
# tree_deg_thresh_high = 
# #tree 0 1 2 3
path_len_thresh_low = 4
path_len_thresh_high=8
#path   0 1 2
circuit_len_thresh_low = 4
circuir_len_thresh_high=10
#curcuit 0 1 2
subgraph_type=1+circuir_len_thresh_high-circuit_len_thresh_low
def count_motif2(A):

    paths=[]
    trees=[]
    circuits=[]
    
    trees_set=[]
    circuits_set=[]
    paths_set=[]

    node2neis=[]
    neis2node=[]

    max_hop=[0 for i in range(A.shape[0])]

    #find 1-hop neighbours
    for node_id in range(A.shape[0]):
        line = A[node_id]
        neighbors = set(np.where(line > 0)[0])
        if len(neighbors)!=0:
            max_hop[node_id]=1
        node2neis.append([neighbors,neighbors.copy()])
        neis2node.append([set() for i in range(A.shape[0])])

        # if tree existed
        
        # if len(neighbors) + 1 > tree_deg_thresh_low and len(neighbors) + 1 < tree_deg_thresh_high:
        #     neighbors.add(node_id)
        #     tree_set=neighbors
        #     if tree_set not in trees_set:
        #         trees_set.append(tree_set)
        #         trees.append(list(tree_set))

    # find 2-hop neighbours
    for node_id in range(A.shape[0]):
        neis2=set()
        for nei_id in node2neis[node_id][1]:
            neis2_set=(node2neis[nei_id][1]-{node_id})
            if len(neis2_set)!=0:
                max_hop[node_id]=2
            neis2=neis2|neis2_set
            for ns in neis2_set:

                #if 3-nodes circuit existing
                if ns in node2neis[node_id][1]:
                    if {node_id,ns,nei_id} not in circuits_set:
                        circuits.append([node_id,ns,nei_id])
                        circuits_set.append({node_id,ns,nei_id})
                    continue

                neis2node[node_id][ns].add(nei_id)
        node2neis[node_id].append(neis2)
        node2neis[node_id][0]=node2neis[node_id][0]|neis2

    find_pown_hop(A,neis2node,node2neis,max_hop,circuits_set,circuits)

    #find paths
    for node_id in range(A.shape[0]):
        # print(max_hop)
        max_hop_2=max_hop[node_id]
        if max_hop_2+2<=path_len_thresh_low or max_hop_2+2>=path_len_thresh_high:
            continue
        for nei_id in node2neis[node_id][max_hop_2]:
            path_set=neis2node[node_id][nei_id]|{node_id,nei_id}
            if path_set not in paths_set:
                paths_set.append(path_set)
                paths.append(list(path_set))

    return trees,paths,circuits


def find_pown_hop(A,neis2node,node2neis,max_hop,circuits_set,circuits):
    for j in range(1,SCALE):
        powj = pow(BASE, j)
        powj1 = pow(BASE, j + 1)
        for node_id in range(A.shape[0]):
            existing_neis = set()
            neis = [set() for i in range(powj1-powj)]
            for i in range(powj+1, powj1+1):
                for nei_id in node2neis[node_id][powj]:
                    neis_set = (node2neis[nei_id][i-powj]-{node_id})
                    neis_set_new = set()
                    for ns in neis_set:

                        #back path
                        if ns in existing_neis or ns in neis2node[node_id][nei_id] or \
                    len(neis2node[node_id][ns]&neis2node[node_id][nei_id])!=0 or\
                    len(neis2node[node_id][ns]&neis2node[nei_id][ns])!=0 or\
                    len(neis2node[node_id][nei_id]&neis2node[nei_id][ns])!=0:
                            continue

                        # if circuit existed
                        if len(neis2node[node_id][ns])!=0 and ns not in neis2node[node_id][nei_id]:
                            circuit_set=neis2node[node_id][ns]|neis2node[ns][nei_id]|neis2node[node_id][nei_id]|{node_id,nei_id,ns}
                            if circuit_set not in circuits_set:
                                circuits_set.append(circuit_set)
                                circuits.append(list(circuit_set))
                            continue
                        '''
                        if ns in existing_neis or len(neis2node[node_id][ns]) != 0 or len(
                                neis2node[nei_id][ns] & neis2node[node_id][nei_id]) != 0:
                            continue
                        '''
                        neis_set_new.add(ns)
                        neis2node[node_id][ns] = neis2node[node_id][ns].union(
                            neis2node[node_id][nei_id]|neis2node[nei_id][ns])
                        neis2node[node_id][ns].add(nei_id)

                    neis[i-powj-1] = neis[i-powj-1] | neis_set_new
                    existing_neis = existing_neis | neis[i-powj-1]

                if len(neis[i-powj-1])!=0:
                    max_hop[node_id]+=1
                node2neis[node_id].append(neis[i-powj-1])
                node2neis[node_id][0] = node2neis[node_id][0] | existing_neis
# A=np.array([
#     [0,1,0,0,0,0,0,0,0,0,0],
#     [1, 0, 1, 0, 0, 0, 0,0,0,0,0],
#     [0,1,0,1,0,0,0,1,0,0,0],
#     [0, 0, 1, 0, 1,0, 0,0,0,0,0],
#     [0, 0, 0, 1, 0, 1, 0,0,0,0,0],
#     [0,0,0,0,1, 0, 1, 0, 0, 0, 0],
#     [0,0,0,0,0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 1,0,1,0,0],
#     [0,0,0,0, 0, 0, 0, 1, 0, 1,1],
#     [0,0,0,0, 0, 0, 0, 0,1,0,0],
#     [0,0,0,0, 0, 0, 0, 0,1,0,0]
# ])
def allocate(A):
    node_N=A.shape[0]
    node_in_subgraph=np.zeros((node_N,subgraph_type),int)
    tree,path,circuit=count_motif2(A)
    # for i in tree:
    #     type_id=len(i)-tree_deg_thresh_low
    #     for node_id in i:
    #         node_in_subgraph[node_id][type_id]+=1
    # for j in path:
    #     type_id=len(j)-path_len_thresh_low
    #     for node_id in j:
    #         node_in_subgraph[node_id][type_id]+=1
    for k in circuit:
        if len(k)<circuit_len_thresh_low or len(k)>circuir_len_thresh_high:
            continue
        type_id=len(k)-circuit_len_thresh_low
        for node_id in k:
            node_in_subgraph[node_id][type_id]+=1
    return node_in_subgraph
    # print(node_in_subgraph)

def count_curcuit(A):
    node_in_subgraph=np.zeros((6),int)
    tree,path,circuit=count_motif2(A)
    for k in circuit:
        if len(k)<circuit_len_thresh_low or len(k)>circuir_len_thresh_high:
            continue
        else:
            node_in_subgraph[len(k)-3]+=1
    return node_in_subgraph

def _main():
    res=[]
    for i in range(1,5976+1):
        print(i)
        filename='/new_disk_B/scy/u_sto_net/graph{}.csv'.format(i)
        temp_A=np.loadtxt(filename,dtype=int,delimiter=',')
        res.append(count_curcuit(temp_A))
    np.savetxt('./graphlet/log_circle.txt',np.array(res))
# _main()    