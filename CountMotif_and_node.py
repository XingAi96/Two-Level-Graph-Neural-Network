# encoding: utf-8
import numpy as np
import csv
import copy
import os,shutil
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

Nm = 8
SCALE=6
BASE=2
OVERLAP_THRESH=1
MIN_MOTIFS=5#70
MOTIF_COUNT_THRESH=1e3#1e4

class gen_hyper_graph:

    def __init__(self):
        self.tree_deg_thresh = 3
        self.path_len_thresh = 4

        self.paths = []
        self.trees = []
        self.circuits = []

        self.paths_set = []
        self.trees_set = []
        self.circuits_set = []

        self.paths_label = []
        self.trees_label = []
        self.circuits_label = []

        self.data_path=r'D:\QGNN\data'
        self.label_path=r'dataset'
        self.output_path=r'dataset'
        self.datasets=['MUTAG','PTC_MR','NCI1','IMDB-BINARY','IMDB-MULTI',]
        self.node_label_count={'AIDS':37,'COLLAB':0,'IMDB-BINARY':0,
                          'IMDB-MULTI':0,'MUTAG':7,'NCI1':37,
                          'PROTEINS_full':61,'PTC_FM':18,'PTC_FR':19,
                          'PTC_MM':20,'PTC_MR':18}

        self.pass_num=[207800]

        self.A = np.zeros((7, 7))
        edges = [[0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]]
        for e in edges:
            self.A[e[0]][e[1]] = 1
            self.A[e[1]][e[0]] = 1

        self.MAX_DEPTH=10
        self.depth=0
        self.is_cluster=False


    def count_motif(self, node_idx, stack,stack_label):

        if (len(self.trees)+len(self.paths)+len(self.circuits))>=MOTIF_COUNT_THRESH or self.depth>self.MAX_DEPTH:
            return

        '''
        for node_idx in range(self.A.shape[0]):
            line = self.A[node_idx]
            neighbors = np.where(line > 0)[0]
            print(node_idx,neighbors)
        return
        '''
        line = self.A[node_idx]
        neighbors = np.where(line > 0)[0]
        # find trees with more than dt children
        if neighbors.shape[0] > self.tree_deg_thresh:
            tree_list = [node_idx] + neighbors.tolist()
            tree_label_list=[self.node_features[ni] for ni in tree_list]

            if set(tree_list) not in self.trees_set:
                self.trees.append(np.array(tree_list))
                self.trees_set.append(set(tree_list))
                self.trees_label.append([tree_label_list,1])

        # find circuits
        for nei in neighbors:
            if len(stack) != 0 and nei in stack[:-1]:
                idx = np.where(np.array(stack[:-1]) == nei)[0][0]
                stack_copy = stack.copy()
                stack_label_copy = stack_label.copy()

                if len(self.circuits) != 0 and stack_copy in self.circuits[-1]:
                    del self.circuits[-1]
                    del self.circuits_label[-1]
                    del self.circuits_set[-1]

                stack_copy.append(node_idx)
                stack_label_copy.append(self.node_features[node_idx])

                if set(stack_copy[idx:]) not in self.circuits_set:
                    self.circuits.append(np.array(stack_copy[idx:]))
                    self.circuits_set.append(set(stack_copy[idx:]))
                    self.circuits_label.append([stack_label_copy[idx:],2])

        # find paths
        if len(stack) > self.path_len_thresh:
            stack_copy = stack.copy()
            stack_label_copy = stack_label.copy()

            if len(self.paths)!=0 and stack_copy in self.paths[-1]:
                del self.paths[-1]
                del self.paths_label[-1]
                del self.paths_set[-1]

            stack_copy.append(node_idx)
            stack_label_copy.append(self.node_features[node_idx])

            if set(stack_copy) not in self.paths_set:
                self.paths.append(np.array(stack_copy))
                self.paths_set.append(set(stack_copy))
                self.paths_label.append([stack_label_copy, 0])

        for nei in neighbors:
            if nei not in stack:
                stack_copy = stack.copy()
                stack_label_copy = stack_label.copy()

                stack_copy.append(node_idx)
                stack_label_copy.append(self.node_features[node_idx])
                self.depth+=1
                self.count_motif(nei, stack_copy,stack_label_copy)
                self.depth-=1
        return


    def run(self):
        for data_name in self.datasets:
            if os.path.exists('dataset/%s/%s.txt' % (data_name, data_name + '_motif2A')):
                os.remove('dataset/%s/%s.txt' % (data_name, data_name + '_motif2A'))

            dataset_path=os.path.join(self.data_path,data_name)
            label_path=os.path.join(self.label_path,data_name+'_label.txt')

            labels=[]
            with open(label_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    labels.append(int(row[0]))

            motif_labels=[]
            graph_strs=str(len(labels))+'\n'
            motif_strs=[str(len(labels))+'\n']

            for file_name in os.listdir(dataset_path):
                if 'graph' in file_name:
                    graph_num = int(file_name.split('.')[0].split('_')[-1])
                    print(data_name, graph_num)

                    if graph_num in self.pass_num:
                        continue

                    graph_path = os.path.join(dataset_path, file_name)
                    node_feature_path = os.path.join(dataset_path, 'node_feature_' + str(graph_num) + '.csv')
                    edge_feature_path = os.path.join(dataset_path, 'edge_feature_' + str(graph_num) + '.csv')

                    self.A = []
                    max_deg=0
                    max_deg_nodeid=0
                    with open(graph_path) as gf:
                        reader = csv.reader(gf)
                        node_id=0
                        for row in reader:
                            self.A.append([int(float(i)) for i in row])
                            aline = np.array(self.A[-1])
                            deg = np.where(aline > 0)[0]
                            if len(deg) > max_deg:
                                max_deg = len(deg)
                                max_deg_nodeid =node_id
                            node_id+=1
                        gf.close()

                    self.A = np.array(self.A)

                    self.node_features = []
                    if os.path.exists(node_feature_path):
                        with open(node_feature_path) as nfp:
                            reader = csv.reader(nfp)
                            for row in reader:
                                self.node_features.append(int(float(row[0])))
                            nfp.close()
                    if len(self.node_features)==0:
                        self.node_features=[2]*self.A.shape[0]

                    graph_str = str(self.A.shape[0]) + ' ' + str(labels[graph_num - 1]) + '\n'
                    for i in range(self.A.shape[0]):
                        neis = np.where(self.A[i] > 0)[0]
                        if len(self.node_features) == 0:
                            graph_str += '0 '
                        else:
                            graph_str += str(self.node_features[i]) + ' '
                        graph_str += str(len(neis))
                        for nei in neis:
                            graph_str += ' '
                            graph_str += str(nei)
                        graph_str += '\n'
                    graph_strs += graph_str

                    deg=np.where(self.A>0)[0]
                    if deg.shape[0]>100:
                        self.MAX_DEPTH=7
                    elif deg.shape[0]>70 and self.A.shape[0]<40:
                        self.MAX_DEPTH=10
                    else:
                        self.MAX_DEPTH=80

                    hyperA, motif2A, motif_label = self.gen_hyperA(max_deg_nodeid)

                    if motif2A.shape[0] == 0:
                        hyperA=self.A.copy()
                        motif_label = np.array([[1, 4] for As in range(self.A.shape[0])])
                        motif2A=np.eye(self.A.shape[0])
                        print(graph_num,self.A.shape[0])
                    motif_labels.append(motif_label)

                    motif_str=[str(hyperA.shape[0]) + ' ' + str(labels[graph_num - 1]) + '\n']
                    for i in range(hyperA.shape[0]):
                        line = hyperA[i]
                        neis = np.where(line > 0)[0]
                        m_str=str(len(neis)) + ' '
                        edge_types = line[neis]
                        for nei in neis:
                            m_str+=(str(nei) + ' ')
                        # for et in edge_types:
                        # f.write(str(et)+' ')
                        m_str+='\n'
                        motif_str.append(m_str)
                    motif_strs.append(motif_str)

                    if not os.path.exists('dataset/%s/%s.txt' % (data_name, data_name + '_motif2A')):
                        with open('dataset/%s/%s.txt' % (data_name, data_name + '_motif2A'), 'w') as m2f:
                            m2f.close()
                    with open('dataset/%s/%s.txt' % (data_name, data_name + '_motif2A'), 'a') as mf:
                        mf.write(str(motif2A.shape[0]) + ' ' + str(motif2A.shape[1]) + '\n')
                        for i in range(motif2A.shape[0]):
                            for ele in motif2A[i]:
                                mf.write(str(ele) + ' ')
                            mf.write('\n')
                        mf.close()

            with open('dataset/%s/%s.txt' % (data_name, data_name),'w') as f:
                f.write(graph_strs)
                f.close()

            mls = []
            for i in range(len(motif_labels)):
                if motif_labels[i].shape[0] == 0:
                    continue
                for j in range(motif_labels[i].shape[0]):
                    ml = [np.mean(motif_labels[i][j][0])]
                    if motif_labels[i][j].shape[0]!=1:
                        ml.append(motif_labels[i][j][1])
                    mls.append(ml)
            mls = np.array(mls)
            if self.is_cluster:
                motif_y = KMeans(n_clusters=self.node_label_count[data_name], random_state=9).fit_predict(mls)
                score = metrics.calinski_harabasz_score(mls, motif_y)
            else:
                motif_y=[1]*mls.shape[0]

            hyper_node_num=0
            with open('dataset/%s/%s.txt' % (data_name, data_name + '_motifs'), 'w') as f:
                f.write(motif_strs[0])
                for i in range(1,len(motif_strs)):
                    gms=motif_strs[i][0]
                    f.write(gms)
                    hyper_node_count=int(gms.split(' ')[0])

                    for j in range(hyper_node_count):
                        f.write(str(motif_y[hyper_node_num+j])+' '+motif_strs[i][j+1])
                    hyper_node_num+=hyper_node_count

                f.close()


    def gen_hyperA(self,init_nodeid=0):
        self.count_motif2()
        '''
        #self.count_motif(init_nodeid, [],[])
        if len(self.trees)+len(self.paths)+len(self.circuits)==0:
            for i in range(self.A.shape[0]):
                self.count_motif(i, [],[])
                if len(self.trees) + len(self.paths) + len(self.circuits) != 0:
                    break
        '''
        motifs = np.array(self.trees + self.paths + self.circuits)
        motifs_label=self.trees_label+self.paths_label+self.circuits_label
        self.trees, self.paths, self.circuits = [], [], []
        self.trees_label,self.paths_label,self.circuits_label=[],[],[]
        self.paths_set,self.trees_set,self.circuits_set=[],[],[]

        motifA = np.zeros((len(motifs), len(motifs)))
        same_motifs = []
        for i1 in range(len(motifs)):
            for i2 in range(i1 + 1, len(motifs)):

                common = [x for x in motifs[i1] if x in motifs[i2]]

                # two motifs are highly overlap or same
                if motifs_label[i1]==motifs_label[i2] and (len(motifs)-len(same_motifs))>MIN_MOTIFS:
                    if np.array(motifs[i1]).shape[0] - len(common) <= OVERLAP_THRESH and i1 not in same_motifs:
                        same_motifs.append(i1)
                        break
                    elif np.array(motifs[i2]).shape[0] - len(common) <= OVERLAP_THRESH and i2 not in same_motifs:
                        same_motifs.append(i2)
                        continue

                if np.array(motifs[i1]).shape[0] == np.array(motifs[i2]).shape[0]:
                    # same kind but different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 2
                        motifA[i2][i1] = 2

                else:
                    # different kinds and different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 3
                        motifA[i2][i1] = 3

                if len(common) == 0:
                    for j1 in motifs[i1]:
                        for j2 in motifs[i2]:
                            if self.A[j1][j2] == 1:
                                motifA[i1][i2] = 1

        motifA = np.delete(motifA, same_motifs, axis=0)
        motifA = np.delete(motifA, same_motifs, axis=1)

        motifs=np.array(motifs)
        motifs = np.delete(motifs, same_motifs, axis=0)
        if motifs.shape[0]>MOTIF_COUNT_THRESH:
            print(motifs.shape[0])

        motifs_label=np.array(motifs_label)
        motifs_label = np.delete(motifs_label, same_motifs, axis=0)

        motif2A = np.zeros((motifs.shape[0], self.A.shape[0]))
        for i in range(motifs.shape[0]):
            motif=motifs[i]
            for node in motif:
                motif2A[i][node]=1

        return motifA,motif2A,motifs_label


    #non-recursion count method
    def count_motif2(self):

        self.paths=[]
        self.trees=[]
        self.circuits=[]

        self.node2neis=[]
        self.neis2node=[]

        self.max_hop=[0 for i in range(self.A.shape[0])]

        #find 1-hop neighbours
        for node_id in range(self.A.shape[0]):
            line = self.A[node_id]
            neighbors = set(np.where(line > 0)[0])
            if len(neighbors)!=0:
                self.max_hop[node_id]=1
            self.node2neis.append([neighbors,neighbors.copy()])
            self.neis2node.append([set() for i in range(self.A.shape[0])])

            # if tree existed
            if len(neighbors) + 1 > self.tree_deg_thresh:
                neighbors.add(node_id)
                tree_set=neighbors
                if tree_set not in self.trees_set:
                    self.trees_set.append(tree_set)
                    self.trees.append(list(tree_set))
                    self.trees_label.append([len(list(tree_set)),2])

        # find 2-hop neighbours
        for node_id in range(self.A.shape[0]):
            neis2=set()
            for nei_id in self.node2neis[node_id][1]:
                neis2_set=(self.node2neis[nei_id][1]-{node_id})
                if len(neis2_set)!=0:
                    self.max_hop[node_id]=2
                neis2=neis2|neis2_set
                for ns in neis2_set:

                    #if 3-nodes circuit existing
                    if ns in self.node2neis[node_id][1]:
                        if {node_id,ns,nei_id} not in self.circuits_set:
                            self.circuits.append([node_id,ns,nei_id])
                            self.circuits_set.append({node_id,ns,nei_id})
                            self.circuits_label.append([3,3])
                        continue

                    self.neis2node[node_id][ns].add(nei_id)
            self.node2neis[node_id].append(neis2)
            self.node2neis[node_id][0]=self.node2neis[node_id][0]|neis2

        self.find_pown_hop(SCALE)

        #find paths
        for node_id in range(self.A.shape[0]):
            max_hop=self.max_hop[node_id]
            if max_hop+2<=self.path_len_thresh:
                continue
            for nei_id in self.node2neis[node_id][max_hop]:
                path_set=self.neis2node[node_id][nei_id]|{node_id,nei_id}
                if path_set not in self.paths_set:
                    self.paths_set.append(path_set)
                    self.paths.append(list(path_set))
                    self.paths_label.append([len(list(path_set)),1])

        return


    def find_pown_hop(self, n):
        for j in range(1,n):
            powj = pow(BASE, j)
            powj1 = pow(BASE, j + 1)
            for node_id in range(self.A.shape[0]):
                existing_neis = set()
                neis = [set() for i in range(powj1-powj)]
                for i in range(powj+1, powj1+1):
                    for nei_id in self.node2neis[node_id][powj]:
                        neis_set = (self.node2neis[nei_id][i-powj]-{node_id})
                        neis_set_new = set()
                        for ns in neis_set:

                            #back path
                            if ns in existing_neis or ns in self.neis2node[node_id][nei_id] or \
                        len(self.neis2node[node_id][ns]&self.neis2node[node_id][nei_id])!=0 or\
                        len(self.neis2node[node_id][ns]&self.neis2node[nei_id][ns])!=0 or\
                        len(self.neis2node[node_id][nei_id]&self.neis2node[nei_id][ns])!=0:
                                continue

                            # if circuit existed
                            if len(self.neis2node[node_id][ns])!=0 and ns not in self.neis2node[node_id][nei_id]:
                                circuit_set=self.neis2node[node_id][ns]|self.neis2node[ns][nei_id]|self.neis2node[node_id][nei_id]|{node_id,nei_id,ns}
                                if circuit_set not in self.circuits_set:
                                    self.circuits_set.append(circuit_set)
                                    self.circuits.append(list(circuit_set))
                                    self.circuits_label.append([len(list(circuit_set)),3])
                                continue
                            '''
                            if ns in existing_neis or len(self.neis2node[node_id][ns]) != 0 or len(
                                    self.neis2node[nei_id][ns] & self.neis2node[node_id][nei_id]) != 0:
                                continue
                            '''
                            neis_set_new.add(ns)
                            self.neis2node[node_id][ns] = self.neis2node[node_id][ns].union(
                                self.neis2node[node_id][nei_id]|self.neis2node[nei_id][ns])
                            self.neis2node[node_id][ns].add(nei_id)

                        neis[i-powj-1] = neis[i-powj-1] | neis_set_new
                        existing_neis = existing_neis | neis[i-powj-1]

                    if len(neis[i-powj-1])!=0:
                        self.max_hop[node_id]+=1
                    self.node2neis[node_id].append(neis[i-powj-1])
                    self.node2neis[node_id][0] = self.node2neis[node_id][0] | existing_neis

class gen_hyper_graph2:

    def __init__(self):
        self.tree_deg_thresh = 3
        self.path_len_thresh = 4

        self.paths = []
        self.trees = []
        self.circuits = []

        self.paths_set = []
        self.trees_set = []
        self.circuits_set = []

        self.paths_label = []
        self.trees_label = []
        self.circuits_label = []

        self.data_path=r'GIN_dataset'#r'D:\QGNN\data'
        self.label_path=r'dataset'
        self.output_path=r'C:\Users\Administrator\Desktop\D variation'
        self.datasets=['MUTAG','PTC_MR','NCI1','COX2',
                       'PROTEINS','IMDB-BINARY','IMDB-MULTI']#'PTC_MR','MUTAG','NCI1','IMDB-BINARY','IMDB-MULTI']
        self.node_label_count={'AIDS':37,'COLLAB':0,'IMDB-BINARY':0,
                          'IMDB-MULTI':0,'MUTAG':7,'NCI1':37,
                          'PROTEINS_full':61,'PTC_FM':18,'PTC_FR':19,
                          'PTC_MM':20,'PTC_MR':18}

        self.pass_num=[207800]

        self.A = np.zeros((7, 7))
        edges = [[0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]]
        for e in edges:
            self.A[e[0]][e[1]] = 1
            self.A[e[1]][e[0]] = 1

        self.MAX_DEPTH=10
        self.depth=0
        self.is_cluster=False


    def run(self):
        for data_name in self.datasets:
            if os.path.exists(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motif2A')):
                os.remove(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motif2A'))

            #read origin graph
            dataset_path=os.path.join(self.data_path,data_name+'.txt')

            graph_count=0
            Adjs=[]
            labels=[]
            node_labels=[]
            motif_labels=[]
            with open(dataset_path, 'r') as f:
                lines = f.readlines()
                graph_count = int(lines[0])
                n = 1
                for i in range(graph_count):
                    labels.append(int(lines[n].split(' ')[1]))
                    node_num = int(lines[n].split(' ')[0])

                    Adj=np.zeros((node_num,node_num))
                    node_label = []
                    for j in range(node_num):
                        neis=lines[n+j+1].split(' ')
                        node_label.append(int(neis[0]))
                        neis_num=int(neis[1])
                        neis=neis[2:]
                        for k in range(neis_num):
                            Adj[j][int(neis[k])]=1

                    node_labels.append(node_label)
                    n += (node_num + 1)
                    Adjs.append(Adj)

                f.close()

            #gen generated graph
            motif_strs = [str(len(labels)) + '\n']

            for graph_num in range(1,graph_count+1):
                print(data_name, graph_num)
                if graph_num in self.pass_num:
                    continue

                self.A = Adjs[graph_num-1]
                hyperA, motif2A, motif_label = self.gen_hyperA()

                if motif2A.shape[0] == 0 or motif2A.shape[0] == 1:
                    hyperA = self.A.copy()
                    motif_label = np.array([[1, 4] for As in range(self.A.shape[0])])
                    motif2A = np.eye(self.A.shape[0])
                    print(graph_num, self.A.shape[0])
                motif_labels.append(motif_label)

                motif_str = [str(hyperA.shape[0]) + ' ' + str(labels[graph_num - 1]) + '\n']
                for i in range(hyperA.shape[0]):
                    line = hyperA[i]
                    neis = np.where(line > 0)[0]
                    m_str = str(len(neis)) + ' '
                    for nei in neis:
                        m_str += (str(nei) + ' ')
                    m_str += '\n'
                    motif_str.append(m_str)
                motif_strs.append(motif_str)

                if not os.path.exists(self.output_path + '/%s' % data_name):
                    os.mkdir(self.output_path + '/%s' % data_name)
                if not os.path.exists(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motif2A')):
                    with open(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motif2A'), 'w') as m2f:
                        m2f.close()
                with open(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motif2A'), 'a') as mf:
                    mf.write(str(motif2A.shape[0]) + ' ' + str(motif2A.shape[1]) + '\n')
                    for i in range(motif2A.shape[0]):
                        for ele in motif2A[i]:
                            mf.write(str(ele) + ' ')
                        mf.write('\n')
                    mf.close()


            hyper_node_num=0
            with open(self.output_path+'/%s/%s.txt' % (data_name, data_name + '_motifs'), 'w') as f:
                f.write(motif_strs[0])
                for i in range(1,len(motif_strs)):
                    gms=motif_strs[i][0]
                    f.write(gms)
                    hyper_node_count=int(gms.split(' ')[0])

                    for j in range(hyper_node_count):
                        f.write(str(motif_labels[i-1][j][0])+' '+str(motif_labels[i-1][j][1])+' '+motif_strs[i][j+1])
                    hyper_node_num+=hyper_node_count

                f.close()


    def gen_hyperA(self,init_nodeid=0):
        self.count_motif2()
        '''
        #self.count_motif(init_nodeid, [],[])
        if len(self.trees)+len(self.paths)+len(self.circuits)==0:
            for i in range(self.A.shape[0]):
                self.count_motif(i, [],[])
                if len(self.trees) + len(self.paths) + len(self.circuits) != 0:
                    break
        '''
        motifs = np.array(self.trees + self.paths + self.circuits)
        motifs_label=self.trees_label+self.paths_label+self.circuits_label
        self.trees, self.paths, self.circuits = [], [], []
        self.trees_label,self.paths_label,self.circuits_label=[],[],[]
        self.paths_set,self.trees_set,self.circuits_set=[],[],[]

        motifA = np.zeros((len(motifs), len(motifs)))
        same_motifs = []
        for i1 in range(len(motifs)):
            for i2 in range(i1 + 1, len(motifs)):

                common = [x for x in motifs[i1] if x in motifs[i2]]

                # two motifs are highly overlap or same
                if motifs_label[i1]==motifs_label[i2] and (len(motifs)-len(same_motifs))>MIN_MOTIFS:
                    if np.array(motifs[i1]).shape[0] - len(common) <= OVERLAP_THRESH and i1 not in same_motifs:
                        same_motifs.append(i1)
                        break
                    elif np.array(motifs[i2]).shape[0] - len(common) <= OVERLAP_THRESH and i2 not in same_motifs:
                        same_motifs.append(i2)
                        continue

                if np.array(motifs[i1]).shape[0] == np.array(motifs[i2]).shape[0]:
                    # same kind but different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 2
                        motifA[i2][i1] = 2

                else:
                    # different kinds and different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 3
                        motifA[i2][i1] = 3

                if len(common) == 0:
                    for j1 in motifs[i1]:
                        for j2 in motifs[i2]:
                            if self.A[j1][j2] == 1:
                                motifA[i1][i2] = 1
                                break
                        if motifA[i1][i2]==1:
                            break

        motifA = np.delete(motifA, same_motifs, axis=0)
        motifA = np.delete(motifA, same_motifs, axis=1)

        motifs=np.array(motifs)
        motifs = np.delete(motifs, same_motifs, axis=0)
        if motifs.shape[0]>MOTIF_COUNT_THRESH:
            print(motifs.shape[0])

        motifs_label=np.array(motifs_label)
        motifs_label = np.delete(motifs_label, same_motifs, axis=0)

        motif2A = np.zeros((motifs.shape[0], self.A.shape[0]))
        for i in range(motifs.shape[0]):
            motif=motifs[i]
            for node in motif:
                motif2A[i][node]=1

        return motifA,motif2A,motifs_label


    #non-recursion count method
    def count_motif2(self):

        self.paths=[]
        self.trees=[]
        self.circuits=[]

        self.node2neis=[]
        self.neis2node=[]

        self.max_hop=[0 for i in range(self.A.shape[0])]

        #find 1-hop neighbours
        for node_id in range(self.A.shape[0]):
            line = self.A[node_id]
            neighbors = set(np.where(line > 0)[0])
            if len(neighbors)!=0:
                self.max_hop[node_id]=1
            self.node2neis.append([neighbors,neighbors.copy()])
            self.neis2node.append([set() for i in range(self.A.shape[0])])

            # if tree existed
            if len(neighbors) + 1 > self.tree_deg_thresh:
                neighbors.add(node_id)
                tree_set=neighbors
                if tree_set not in self.trees_set and len(self.trees)<=MOTIF_COUNT_THRESH:
                    self.trees_set.append(tree_set)
                    self.trees.append(list(tree_set))
                    self.trees_label.append([len(list(tree_set)),2])

        # find 2-hop neighbours
        for node_id in range(self.A.shape[0]):
            neis2=set()
            for nei_id in self.node2neis[node_id][1]:
                neis2_set=(self.node2neis[nei_id][1]-{node_id})
                if len(neis2_set)!=0:
                    self.max_hop[node_id]=2
                neis2=neis2|neis2_set
                for ns in neis2_set:

                    #if 3-nodes circuit existing
                    if ns in self.node2neis[node_id][1]:
                        if {node_id,ns,nei_id} not in self.circuits_set:
                            self.circuits.append([node_id,ns,nei_id])
                            self.circuits_set.append({node_id,ns,nei_id})
                            self.circuits_label.append([3,3])
                        continue

                    self.neis2node[node_id][ns].add(nei_id)
            self.node2neis[node_id].append(neis2)
            self.node2neis[node_id][0]=self.node2neis[node_id][0]|neis2

        self.find_pown_hop(SCALE)

        #find paths
        for node_id in range(self.A.shape[0]):
            max_hop=self.max_hop[node_id]
            if max_hop+2<=self.path_len_thresh or len(self.paths)>=MOTIF_COUNT_THRESH:
                continue
            for nei_id in self.node2neis[node_id][max_hop]:
                path_set=self.neis2node[node_id][nei_id]|{node_id,nei_id}
                if path_set not in self.paths_set:
                    self.paths_set.append(path_set)
                    self.paths.append(list(path_set))
                    self.paths_label.append([len(list(path_set)),1])

        return


    def find_pown_hop(self, n):
        for j in range(1,n):
            powj = pow(BASE, j)
            powj1 = pow(BASE, j + 1)
            for node_id in range(self.A.shape[0]):
                existing_neis = set()
                neis = [set() for i in range(powj1-powj)]
                for i in range(powj+1, powj1+1):
                    for nei_id in self.node2neis[node_id][powj]:
                        neis_set = (self.node2neis[nei_id][i-powj]-{node_id})
                        neis_set_new = set()
                        for ns in neis_set:

                            #back path
                            if ns in existing_neis or ns in self.neis2node[node_id][nei_id] or \
                        len(self.neis2node[node_id][ns]&self.neis2node[node_id][nei_id])!=0 or\
                        len(self.neis2node[node_id][ns]&self.neis2node[nei_id][ns])!=0 or\
                        len(self.neis2node[node_id][nei_id]&self.neis2node[nei_id][ns])!=0:
                                continue

                            # if circuit existed
                            if len(self.neis2node[node_id][ns])!=0 and ns not in self.neis2node[node_id][nei_id]:
                                circuit_set=self.neis2node[node_id][ns]|self.neis2node[ns][nei_id]|self.neis2node[node_id][nei_id]|{node_id,nei_id,ns}
                                if circuit_set not in self.circuits_set and len(self.circuits)<=MOTIF_COUNT_THRESH:
                                    self.circuits_set.append(circuit_set)
                                    self.circuits.append(list(circuit_set))
                                    self.circuits_label.append([len(list(circuit_set)),3])
                                continue
                            '''
                            if ns in existing_neis or len(self.neis2node[node_id][ns]) != 0 or len(
                                    self.neis2node[nei_id][ns] & self.neis2node[node_id][nei_id]) != 0:
                                continue
                            '''
                            neis_set_new.add(ns)
                            self.neis2node[node_id][ns] = self.neis2node[node_id][ns].union(
                                self.neis2node[node_id][nei_id]|self.neis2node[nei_id][ns])
                            self.neis2node[node_id][ns].add(nei_id)

                        neis[i-powj-1] = neis[i-powj-1] | neis_set_new
                        existing_neis = existing_neis | neis[i-powj-1]

                    if len(neis[i-powj-1])!=0:
                        self.max_hop[node_id]+=1
                    self.node2neis[node_id].append(neis[i-powj-1])
                    self.node2neis[node_id][0] = self.node2neis[node_id][0] | existing_neis

class gen_hyper_graph3:

    def __init__(self):
        self.tree_deg_thresh = 3
        self.path_len_thresh = 4

        self.paths = []
        self.trees = []
        self.circuits = []

        self.paths_set = []
        self.trees_set = []
        self.circuits_set = []

        self.paths_label = []
        self.trees_label = []
        self.circuits_label = []

        self.data_path=r'GIN_dataset'#r'D:\QGNN\data'
        self.label_path=r'dataset'
        self.output_path=r'C:\Users\Administrator\Desktop\D variation'
        self.datasets=['Synthetic']#'PTC_MR','MUTAG','NCI1','IMDB-BINARY','IMDB-MULTI']
        self.node_label_count={'AIDS':37,'COLLAB':0,'IMDB-BINARY':0,
                          'IMDB-MULTI':0,'MUTAG':7,'NCI1':37,
                          'PROTEINS_full':61,'PTC_FM':18,'PTC_FR':19,
                          'PTC_MM':20,'PTC_MR':18}

        self.pass_num=[207800]

        self.A = np.zeros((7, 7))
        edges = [[0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]]
        for e in edges:
            self.A[e[0]][e[1]] = 1
            self.A[e[1]][e[0]] = 1

        self.MAX_DEPTH=10
        self.depth=0
        self.is_cluster=False


    def run(self):
        for data_name in self.datasets:
            if os.path.exists(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motif2A')):
                os.remove(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motif2A'))

            #read origin graph
            dataset_path=os.path.join(self.data_path,data_name+'.txt')

            graph_count=0
            Adjs=[]
            labels=[]
            node_labels=[]
            motif_labels=[]
            with open(dataset_path, 'r') as f:
                lines = f.readlines()
                graph_count = int(lines[0])
                n = 1
                for i in range(graph_count):
                    labels.append(int(lines[n].split(' ')[1]))
                    node_num = int(lines[n].split(' ')[0])

                    Adj=np.zeros((node_num,node_num))
                    node_label = []
                    for j in range(node_num):
                        neis=lines[n+j+1].split(' ')
                        node_label.append(int(neis[0]))
                        neis_num=int(neis[1])
                        neis=neis[2:]
                        for k in range(neis_num):
                            Adj[j][int(neis[k])]=1

                    node_labels.append(node_label)
                    n += (node_num + 1)
                    Adjs.append(Adj)

                f.close()

            #gen generated graph
            motif_strs = [str(len(labels)) + '\n']

            for graph_num in range(1,graph_count+1):
                print(data_name, graph_num)
                if graph_num in self.pass_num:
                    continue

                self.A = Adjs[graph_num-1]
                hyperA, motif2A, motif_label = self.gen_hyperA()

                if motif2A.shape[0] == 0 or motif2A.shape[0] == 1:
                    hyperA = self.A.copy()
                    motif_label = np.array([[1, 4] for As in range(self.A.shape[0])])
                    motif2A = np.eye(self.A.shape[0])
                    print(graph_num, self.A.shape[0])
                motif_labels.append(motif_label)

                motif_str = [str(hyperA.shape[0]) + ' ' + str(labels[graph_num - 1]) + '\n']
                for i in range(hyperA.shape[0]):
                    line = hyperA[i]
                    neis = np.where(line > 0)[0]
                    m_str = str(len(neis)) + ' '
                    for nei in neis:
                        m_str += (str(nei) + ' ')
                    m_str += '\n'
                    motif_str.append(m_str)
                motif_strs.append(motif_str)

                if not os.path.exists(self.output_path + '/%s/%s' % (data_name,str(SCALE))):
                    os.mkdir(self.output_path + '/%s/%s' % (data_name,str(SCALE)))
                if not os.path.exists(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motif2A')):
                    with open(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motif2A'), 'w') as m2f:
                        m2f.close()
                with open(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motif2A'), 'a') as mf:
                    mf.write(str(motif2A.shape[0]) + ' ' + str(motif2A.shape[1]) + '\n')
                    for i in range(motif2A.shape[0]):
                        for ele in motif2A[i]:
                            mf.write(str(ele) + ' ')
                        mf.write('\n')
                    mf.close()


            hyper_node_num=0
            with open(self.output_path+'/%s/%s/%s.txt' % (data_name,str(SCALE), data_name + '_motifs'), 'w') as f:
                f.write(motif_strs[0])
                for i in range(1,len(motif_strs)):
                    gms=motif_strs[i][0]
                    f.write(gms)
                    hyper_node_count=int(gms.split(' ')[0])

                    for j in range(hyper_node_count):
                        f.write(str(motif_labels[i-1][j][0])+' '+str(motif_labels[i-1][j][1])+' '+motif_strs[i][j+1])
                    hyper_node_num+=hyper_node_count

                f.close()

            shutil.copy(dataset_path,self.output_path+'/'+data_name+'/'+str(SCALE)+'/'+data_name + '.txt')


    def gen_hyperA(self,init_nodeid=0):
        self.count_motif2()
        '''
        #self.count_motif(init_nodeid, [],[])
        if len(self.trees)+len(self.paths)+len(self.circuits)==0:
            for i in range(self.A.shape[0]):
                self.count_motif(i, [],[])
                if len(self.trees) + len(self.paths) + len(self.circuits) != 0:
                    break
        '''
        motifs = np.array(self.trees + self.paths + self.circuits)
        motifs_label=self.trees_label+self.paths_label+self.circuits_label
        self.trees, self.paths, self.circuits = [], [], []
        self.trees_label,self.paths_label,self.circuits_label=[],[],[]
        self.paths_set,self.trees_set,self.circuits_set=[],[],[]

        motifA = np.zeros((len(motifs), len(motifs)))
        same_motifs = []
        for i1 in range(len(motifs)):
            for i2 in range(i1 + 1, len(motifs)):

                common = [x for x in motifs[i1] if x in motifs[i2]]

                # two motifs are highly overlap or same
                if motifs_label[i1]==motifs_label[i2] and (len(motifs)-len(same_motifs))>MIN_MOTIFS:
                    if np.array(motifs[i1]).shape[0] - len(common) <= OVERLAP_THRESH and i1 not in same_motifs:
                        same_motifs.append(i1)
                        break
                    elif np.array(motifs[i2]).shape[0] - len(common) <= OVERLAP_THRESH and i2 not in same_motifs:
                        same_motifs.append(i2)
                        continue

                if np.array(motifs[i1]).shape[0] == np.array(motifs[i2]).shape[0]:
                    # same kind but different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 2
                        motifA[i2][i1] = 2

                else:
                    # different kinds and different motifs have common nodes
                    if len(common) != 0:
                        motifA[i1][i2] = 3
                        motifA[i2][i1] = 3

                if len(common) == 0:
                    for j1 in motifs[i1]:
                        for j2 in motifs[i2]:
                            if self.A[j1][j2] == 1:
                                motifA[i1][i2] = 1
                                break
                        if motifA[i1][i2]==1:
                            break

        motifA = np.delete(motifA, same_motifs, axis=0)
        motifA = np.delete(motifA, same_motifs, axis=1)

        motifs=np.array(motifs)
        motifs = np.delete(motifs, same_motifs, axis=0)
        if motifs.shape[0]>MOTIF_COUNT_THRESH:
            print(motifs.shape[0])

        motifs_label=np.array(motifs_label)
        motifs_label = np.delete(motifs_label, same_motifs, axis=0)

        motif2A = np.zeros((motifs.shape[0], self.A.shape[0]))
        for i in range(motifs.shape[0]):
            motif=motifs[i]
            for node in motif:
                motif2A[i][node]=1

        return motifA,motif2A,motifs_label


    #non-recursion count method
    def count_motif2(self):

        self.paths=[]
        self.trees=[]
        self.circuits=[]

        self.node2neis=[]
        self.neis2node=[]

        self.max_hop=[0 for i in range(self.A.shape[0])]

        #find 1-hop neighbours
        for node_id in range(self.A.shape[0]):
            line = self.A[node_id]
            neighbors = set(np.where(line > 0)[0])
            if len(neighbors)!=0:
                self.max_hop[node_id]=1
            self.node2neis.append([neighbors,neighbors.copy()])
            self.neis2node.append([set() for i in range(self.A.shape[0])])

            # if tree existed
            if len(neighbors) + 1 > self.tree_deg_thresh:
                neighbors.add(node_id)
                tree_set=neighbors
                if tree_set not in self.trees_set and len(self.trees)<=MOTIF_COUNT_THRESH:
                    self.trees_set.append(tree_set)
                    self.trees.append(list(tree_set))
                    self.trees_label.append([len(list(tree_set)),2])

        # find 2-hop neighbours
        for node_id in range(self.A.shape[0]):
            neis2=set()
            for nei_id in self.node2neis[node_id][1]:
                neis2_set=(self.node2neis[nei_id][1]-{node_id})
                if len(neis2_set)!=0:
                    self.max_hop[node_id]=2
                neis2=neis2|neis2_set
                for ns in neis2_set:

                    #if 3-nodes circuit existing
                    if ns in self.node2neis[node_id][1]:
                        if {node_id,ns,nei_id} not in self.circuits_set:
                            self.circuits.append([node_id,ns,nei_id])
                            self.circuits_set.append({node_id,ns,nei_id})
                            self.circuits_label.append([3,3])
                        continue

                    self.neis2node[node_id][ns].add(nei_id)
            self.node2neis[node_id].append(neis2)
            self.node2neis[node_id][0]=self.node2neis[node_id][0]|neis2

        self.find_pown_hop(SCALE)

        #find paths
        for node_id in range(self.A.shape[0]):
            max_hop=self.max_hop[node_id]
            if max_hop+2<=self.path_len_thresh or len(self.paths)>=MOTIF_COUNT_THRESH:
                continue
            for nei_id in self.node2neis[node_id][max_hop]:
                path_set=self.neis2node[node_id][nei_id]|{node_id,nei_id}
                if path_set not in self.paths_set:
                    self.paths_set.append(path_set)
                    self.paths.append(list(path_set))
                    self.paths_label.append([len(list(path_set)),1])

        return


    def find_pown_hop(self, n):
        for j in range(1,n):
            powj = pow(BASE, j)
            powj1 = pow(BASE, j + 1)
            for node_id in range(self.A.shape[0]):
                existing_neis = set()
                neis = [set() for i in range(powj1-powj)]
                for i in range(powj+1, powj1+1):
                    for nei_id in self.node2neis[node_id][powj]:
                        neis_set = (self.node2neis[nei_id][i-powj]-{node_id})
                        neis_set_new = set()
                        for ns in neis_set:

                            #back path
                            if ns in existing_neis or ns in self.neis2node[node_id][nei_id] or \
                        len(self.neis2node[node_id][ns]&self.neis2node[node_id][nei_id])!=0 or\
                        len(self.neis2node[node_id][ns]&self.neis2node[nei_id][ns])!=0 or\
                        len(self.neis2node[node_id][nei_id]&self.neis2node[nei_id][ns])!=0:
                                continue

                            # if circuit existed
                            if len(self.neis2node[node_id][ns])!=0 and ns not in self.neis2node[node_id][nei_id]:
                                circuit_set=self.neis2node[node_id][ns]|self.neis2node[ns][nei_id]|self.neis2node[node_id][nei_id]|{node_id,nei_id,ns}
                                if circuit_set not in self.circuits_set and len(self.circuits)<=MOTIF_COUNT_THRESH:
                                    self.circuits_set.append(circuit_set)
                                    self.circuits.append(list(circuit_set))
                                    self.circuits_label.append([len(list(circuit_set)),3])
                                continue
                            '''
                            if ns in existing_neis or len(self.neis2node[node_id][ns]) != 0 or len(
                                    self.neis2node[nei_id][ns] & self.neis2node[node_id][nei_id]) != 0:
                                continue
                            '''
                            neis_set_new.add(ns)
                            self.neis2node[node_id][ns] = self.neis2node[node_id][ns].union(
                                self.neis2node[node_id][nei_id]|self.neis2node[nei_id][ns])
                            self.neis2node[node_id][ns].add(nei_id)

                        neis[i-powj-1] = neis[i-powj-1] | neis_set_new
                        existing_neis = existing_neis | neis[i-powj-1]

                    if len(neis[i-powj-1])!=0:
                        self.max_hop[node_id]+=1
                    self.node2neis[node_id].append(neis[i-powj-1])
                    self.node2neis[node_id][0] = self.node2neis[node_id][0] | existing_neis

    def prepare(self):
        for d in self.datasets:
            source_path=os.path.join(os.path.join(self.output_path,d),str(SCALE))
            target_path=os.path.join(r'C:\Users\Administrator\Desktop\dataset',d)
            if not os.path.exists(target_path):
                os.mkdir(target_path)
            for f in os.listdir(source_path):
                print(f)
                shutil.copy(source_path,target_path)


for i in range(1,9):
    SCALE=i
    GHG = gen_hyper_graph3()
    GHG.A = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ])

    GHG.node_features = list(np.zeros((11,)))

    GHG.run()  # gen_hyperA(0)

