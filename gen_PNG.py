import os
import copy
import numpy as np
from util import load_data
from shutil import copyfile


def gen_PNG_randomly():
    graph_num = 100
    max_node_num = 30
    min_node_num = 10
    min_edge_num = min_node_num * 3

    output_path = r'dataset/Synthetic/Synthetic.txt'
    with open(output_path, 'w') as f:
        f.write(str(graph_num) + '\n')
        for graph_idx in range(graph_num):
            node_num = np.random.randint(min_node_num, max_node_num)
            edge_num = np.random.randint(min_edge_num, (node_num * (node_num - 1) * 0.5))
            adj_mat = np.zeros((node_num, node_num))

            node_degree = {}
            node_neis = {}
            for node_idx in range(node_num):
                node_degree[node_idx] = 0
                node_neis[node_idx] = []

            # generate edges of the graph
            for edge_idx in range(edge_num):
                i, j = 0, 0
                while (i == j or adj_mat[i][j] == 1):
                    i = np.random.randint(0, node_num)
                    j = np.random.randint(0, node_num)
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

                node_degree[i] += 1
                node_degree[j] += 1
                node_neis[i].append(j)
                node_neis[j].append(i)

            # generate the PNG by changing a pair of edges randomly
            PNG_adj_mat = adj_mat.copy()
            PNG_node_neis = copy.deepcopy(node_neis)

            node1, node2 = -1, -1
            while (node1 == -1 or node_degree[node1] <= 2):
                node1 = np.random.randint(0, node_num)
            for node_idx in range(node_num):
                node2 = node_idx
                neis1 = node_neis[node1]
                neis2 = node_neis[node2]

                if node_idx != node1 and node_degree[node_idx] == node_degree[node1] \
                        and len(set(neis1) - set(neis2)) > 0 and len(set(neis2) - set(neis1)) > 0:
                    break

            nei1 = list(set(neis1) - set(neis2))[0]
            nei2 = list(set(neis2) - set(neis1))[0]
            PNG_adj_mat[node1][nei1], PNG_adj_mat[node2][nei2], PNG_adj_mat[nei1][node1], PNG_adj_mat[nei2][
                node2] = 0, 0, 0, 0
            PNG_adj_mat[node1][nei2], PNG_adj_mat[node2][nei1], PNG_adj_mat[nei2][node1], PNG_adj_mat[nei1][
                node2] = 1, 1, 1, 1

            PNG_node_neis[node1].remove(nei1)
            PNG_node_neis[node2].remove(nei2)
            PNG_node_neis[node1].append(nei2)
            PNG_node_neis[node2].append(nei1)

            f.write(str(node_num) + ' 1\n')
            for node_idx in range(node_num):
                node_str = str(node_degree[node_idx]) + ' ' + str(node_degree[node_idx]) + ' '
                for nei_idx in node_neis[node_idx]:
                    node_str += str(nei_idx) + ' '
                node_str += '\n'
                f.write(node_str)

            f.write(str(node_num) + ' 0\n')
            for node_idx in range(node_num):
                node_str = str(node_degree[node_idx]) + ' ' + str(node_degree[node_idx]) + ' '
                for nei_idx in PNG_node_neis[node_idx]:
                    node_str += str(nei_idx) + ' '
                node_str += '\n'
                f.write(node_str)

        f.close()

def gen_PNG_from_dataset(dataset):
    g_list,_,_,_=load_data(dataset,False)
    graph_num = 100

    graph_n=0
    output_path = r'dataset/Synthetic_'+dataset+'/'+'Synthetic_'+dataset+'.txt'
    if not os.path.exists(r'dataset/Synthetic_'+dataset):
        os.mkdir(r'dataset/Synthetic_'+dataset)
    with open(output_path, 'w') as f:
        f.write(str(graph_num) + '\n')
        for g in g_list:
            if g.label == 0:
                node_num = len(g.node_features)
                node1, node2 = -1, -1
                for node1 in range(node_num):
                    if node1 == -1 or len(g.neighbors[node1]) <= 2:
                        node1 = np.random.randint(0, node_num)
                if node1 == -1:
                    continue

                for node_idx in range(node_num):
                    node2 = node_idx
                    neis1 = g.neighbors[node1]
                    neis2 = g.neighbors[node2]

                    if node_idx != node1 and len(neis1) == len(neis2) and node2 not in neis1\
                            and len(set(neis1) - set(neis2)) > 0 and len(set(neis2) - set(neis1)) > 0:
                        break
                if node2 == -1:
                    continue

                if len(set(neis1) - set(neis2))==0 or len(set(neis2) - set(neis1))==0 or\
                        set(neis1) - set(neis2)==set(neis2) - set(neis1):
                    continue

                PNG_node_neis = copy.deepcopy(g.neighbors)
                nei1 = list(set(neis1) - set(neis2))[0]
                nei2 = list(set(neis2) - set(neis1))[0]

                PNG_node_neis[node1].remove(nei1)
                PNG_node_neis[node2].remove(nei2)
                PNG_node_neis[node1].append(nei2)
                PNG_node_neis[node2].append(nei1)

                f.write(str(node_num) + ' 1\n')
                for node_idx in range(node_num):
                    node_str = str(len(g.neighbors[node_idx])) + ' ' + str(len(g.neighbors[node_idx])) + ' '
                    for nei_idx in g.neighbors[node_idx]:
                        node_str += str(nei_idx) + ' '
                    node_str += '\n'
                    f.write(node_str)

                f.write(str(node_num) + ' 0\n')
                for node_idx in range(node_num):
                    node_str = str(len(PNG_node_neis[node_idx])) + ' ' + str(len(PNG_node_neis[node_idx])) + ' '
                    for nei_idx in PNG_node_neis[node_idx]:
                        node_str += str(nei_idx) + ' '
                    node_str += '\n'
                    f.write(node_str)

                graph_n += 1
                if graph_n >= graph_num:
                    break
        f.close()
    copyfile(output_path,'GIN_dataset'+'/'+'Synthetic_'+dataset+'.txt')
    print(graph_n)


datasets=['MUTAG','PTC_MR','NCI1','PROTEINS','COX2','IMDB-BINARY','IMDB-MULTI']
for dataset in datasets:
    gen_PNG_from_dataset(dataset)










