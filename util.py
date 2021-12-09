import os
import networkx as nx
import numpy as np
import random
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold


class S2VGraph(object):
	def __init__(self, g, label, node_tags=None, node_features=None):
		'''
			g: a networkx graph
			label: an integer graph label
			node_tags: a list of integer node tags
			node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
			edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
			neighbors: list of neighbors (without self-loop)
		'''
		self.label = label
		self.g = g
		self.node_tags = node_tags
		self.neighbors = []
		self.node_features = node_features
		self.edge_mat = 0

		self.max_neighbor = 0


def load_data(dataset, degree_as_tag,scale):
	'''
		dataset: name of dataset
		test_proportion: ratio of test train split
		seed: random seed for random splitting of dataset
	'''

	print('loading data')
	g_list = []
	label_dict = {}
	feat_dict = {}
	hfeat_dict={}

	with open('dataset/%s/%s/%s.txt' % (dataset,str(scale), dataset), 'r') as f:
		n_g = int(f.readline().strip())
		for i in range(n_g):
			row = f.readline().strip().split()
			n, l = [int(w) for w in row]
			if not l in label_dict:
				mapped = len(label_dict)
				label_dict[l] = mapped
			g = nx.Graph()
			node_tags = []
			node_features = []
			n_edges = 0
			for j in range(n):
				g.add_node(j)
				row = f.readline().strip().split()
				tmp = int(row[1]) + 2
				if tmp == len(row):
					# no node attributes
					row = [int(w) for w in row]
					attr = None
				else:
					row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
				if not row[0] in feat_dict:
					mapped = len(feat_dict)
					feat_dict[row[0]] = mapped
				node_tags.append(feat_dict[row[0]])

				if tmp > len(row):
					node_features.append(attr)

				n_edges += row[1]
				for k in range(2, len(row)):
					g.add_edge(j, row[k])

			if node_features != []:
				node_features = np.stack(node_features)
				node_feature_flag = True
			else:
				node_features = None
				node_feature_flag = False

			assert len(g) == n

			g_list.append(S2VGraph(g, l, node_tags))

	with open('dataset/%s/%s/%s.txt' % (dataset, str(scale),dataset+'_motif2A'), 'r') as f:
		mat_list = []
		line = f.readline()
		while line:
			line = line.split(' ')
			rows = int(line[0])
			mat=[]
			for i in range(rows):
				row = f.readline()
				row = row.split(' ')[:-1]
				mat.append([int(float(ele)) for ele in row])

			mat_list.append(mat)
			line = f.readline()
		f.close()


	#add labels and edge_mat
	for g in g_list:
		g.neighbors = [[] for i in range(len(g.g))]
		for i, j in g.g.edges():
			g.neighbors[i].append(j)
			g.neighbors[j].append(i)
		degree_list = []
		for i in range(len(g.g)):
			g.neighbors[i] = g.neighbors[i]
			degree_list.append(len(g.neighbors[i]))
		g.max_neighbor = max(degree_list)

		g.label = label_dict[g.label]

		edges = [list(pair) for pair in g.g.edges()]
		edges.extend([[i, j] for j, i in edges])

		deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
		g.edge_mat = torch.LongTensor(edges).transpose(0,1)

	if degree_as_tag:
		for g in g_list:
			g.node_tags = list(dict(g.g.degree).values())

	#Extracting unique tag labels
	tagset = set([])
	for g in g_list:
		tagset = tagset.union(set(g.node_tags))

	tagset = list(tagset)
	tag2index = {tagset[i]:i for i in range(len(tagset))}

	for g in g_list:
		g.node_features = torch.zeros(len(g.node_tags), len(tagset))
		g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


	print('# classes: %d' % len(label_dict))
	print('# maximum node tag: %d' % len(tagset))
	LEN_TAGSET=len(tagset)

	print("# data: %d" % len(g_list))

	return g_list,len(label_dict),mat_list,LEN_TAGSET


def load_hyper_data(dataset, degree_as_tag,scale):
	'''
		dataset: name of dataset
		test_proportion: ratio of test train split
		seed: random seed for random splitting of dataset
	'''

	print('loading data')
	g_list = []
	label_dict = {}
	feat_dict = {}
	hfeat_dict={}

	with open('dataset/%s/%s/%s_motifs.txt' % (dataset,str(scale), dataset), 'r') as f:
		n_g = int(f.readline().strip())
		for i in range(n_g):
			row = f.readline().strip().split()
			n, l = [int(w) for w in row]
			if not l in label_dict:
				mapped = len(label_dict)
				label_dict[l] = mapped
			g = nx.Graph()
			node_tags = []
			node_features = []
			n_edges = 0
			for j in range(n):
				g.add_node(j)
				row = f.readline().strip().split()
				tmp = int(row[2]) + 3
				if tmp == len(row):
					# no node attributes
					row = [int(w) for w in row]
					attr = np.array([float(row[0]),float(row[1])])
				else:
					row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
				node_tags.append(attr)
				node_features.append(attr)

				n_edges += row[1]
				for k in range(3, len(row)):
					g.add_edge(j, row[k])

			if node_features != []:
				node_features = torch.from_numpy(np.array(node_features))
				node_feature_flag = True
			else:
				node_features = None
				node_feature_flag = False

			assert len(g) == n

			g_list.append(S2VGraph(g, l, node_tags,node_features))

	#add labels and edge_mat
	ng=0
	for g in g_list:
		ng+=1
		g.neighbors = [[] for i in range(len(g.g))]
		for i, j in g.g.edges():
			g.neighbors[i].append(j)
			g.neighbors[j].append(i)
		degree_list = []
		if len(g.g)==0 or len(g.g)==1:
			degree_list.append(0)
		for i in range(len(g.g)):
			g.neighbors[i] = g.neighbors[i]
			degree_list.append(len(g.neighbors[i]))
		g.max_neighbor = max(degree_list)

		g.label = label_dict[g.label]
		edges = [list(pair) for pair in g.g.edges()]
		if len(edges)==0 or len(edges)==1:
			g.edge_mat=[]
		else:
			edges.extend([[i, j] for j, i in edges])
			deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
			g.edge_mat = torch.LongTensor(edges).transpose(0,1)

	if degree_as_tag:
		for g in g_list:
			g.node_tags = list(dict(g.g.degree).values())


	print('# classes: %d' % len(label_dict))

	print("# data: %d" % len(g_list))

	return g_list,len(label_dict)


def load_node_data(dataset="cora"):
	path = os.path.join(r'dataset',dataset)

	print('Loading {} dataset...'.format(dataset))

	idx_features_labels = np.genfromtxt("{}\{}.content".format(path, dataset), dtype=np.dtype(str))
	features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
	labels = encode_onehot(idx_features_labels[:, -1])

	# build graph
	idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
	idx_map = {j: i for i, j in enumerate(idx)}
	edges_unordered = np.genfromtxt("{}\{}.cites".format(path, dataset), dtype=np.int32)
	edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
	adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

	# build symmetric adjacency matrix
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	features = normalize_features(features)
	adj = normalize_adj(adj + sp.eye(adj.shape[0]))

	idx_train = range(140)
	idx_val = range(200, 500)
	idx_test = range(500, 1500)

	adj = torch.FloatTensor(np.array(adj.todense()))
	features = torch.FloatTensor(np.array(features.todense()))
	labels=np.where(labels)[1]
	num_classes = max(labels) + 1
	labels = torch.LongTensor(labels)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)

	return adj, features, labels, num_classes, idx_train, idx_val, idx_test



def encode_onehot(labels):
	# The classes must be sorted before encoding to enable static class encoding.
	# In other words, make sure the first class always maps to index 0.
	classes = sorted(list(set(labels)))
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
	return labels_onehot


def normalize_adj(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv_sqrt = np.power(rowsum, -0.5).flatten()
	r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
	r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
	return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def separate_data(graph_list,hyper_graph_list,motif2A_list, seed, fold_idx):
	assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
	skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

	labels = [graph.label for graph in graph_list]
	idx_list = []
	for idx in skf.split(np.zeros(len(labels)), labels):
		idx_list.append(idx)
	train_idx, test_idx = idx_list[fold_idx]

	train_graph_list = [graph_list[i] for i in train_idx]
	train_hyper_graph_list=[hyper_graph_list[i] for i in train_idx]
	train_motif2A_list=[motif2A_list[i] for i in train_idx]

	test_graph_list = [graph_list[i] for i in test_idx]
	test_hyper_graph_list = [hyper_graph_list[i] for i in test_idx]
	test_motif2A_list=[motif2A_list[i] for i in test_idx]

	return train_graph_list, train_hyper_graph_list,train_motif2A_list,test_graph_list,test_hyper_graph_list,test_motif2A_list


def separate_data_DQGNN(seed):
	skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
	idx_list = []
	for idx in skf.split(np.zeros(467), np.zeros(467)):
		idx_list.append(idx)

	for fold_idx in range(1,11):
		train_idx, test_idx = idx_list[fold_idx-1]
		with open('train_idx-'+str(fold_idx)+'.txt','w') as f:
			for ti in train_idx:
				f.write(str(ti)+'\n')
			f.close()
		with open('test_idx-'+str(fold_idx)+'.txt','w') as f:
			for ti in test_idx:
				f.write(str(ti)+'\n')
			f.close()
	return


def RWGNN_data_style(input_path,output_path,dataset):

	A,graph_labels,node_labels='','',''
	start_node_id=1
	graph_indicators=[]
	with open(input_path, 'r') as f:
		line = f.readline()
		graph_num = int(line)
		line = f.readline()
		while line:

			line_str = line.split(' ')
			node_num = int(line_str[0])

			graph_indicators.append(node_num)
			graph_labels += (line_str[1] + '\n')

			for node_idx in range(node_num):
				line = f.readline()
				line_str = line.split(' ')

				node_labels += (line_str[0] + '\n')
				neis_num = int(line_str[1])

				for nei_idx in range(neis_num):
					A += str(start_node_id + int(line_str[2 + nei_idx])) + ', ' + str(start_node_id + node_idx) + '\n'
					#A += str(start_node_id + node_idx) + ', ' + str(start_node_id + int(line_str[2 + nei_idx])) + '\n'

			start_node_id += node_num
			line = f.readline()

		f.close()

	with open(os.path.join(output_path, dataset + '_A.txt'), 'w') as Af:
		Af.write(A)
		Af.close()
	with open(os.path.join(output_path, dataset + '_graph_labels.txt'), 'w') as glf:
		glf.write(graph_labels)
		glf.close()
	with open(os.path.join(output_path, dataset + '_node_labels.txt'), 'w') as nlf:
		nlf.write(node_labels)
		nlf.close()
	with open(os.path.join(output_path, dataset + '_graph_indicator.txt'), 'w') as gif:
		for gi in range(len(graph_indicators)):
			for i in range(graph_indicators[gi]):
				gif.write((str(gi) + '\n'))
		gif.close()

#RWGNN_data_style(r'D:\Two-level GNN\dataset\Synthetic\Synthetic.txt','','Synthetic')


# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	digits = datasets.load_digits(n_class=10)
	data = digits.data
	label = digits.target
	n_samples, n_features = data.shape
	return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# 遍历所有样本
	for i in range(data.shape[0]):
		ax.scatter(data[i, 0], data[i, 1],data[i, 2], color=plt.cm.Set1(label[i] / 10))
	plt.xticks()
	plt.yticks()
	plt.title(title, fontsize=14)
	return fig

def main():
	data, label , _, _ = get_data()
	print('Starting compute t-SNE Embedding...')
	ts = TSNE(n_components=3, init='pca', random_state=0)
	reslut = ts.fit_transform(data)
	fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
	plt.show()


if __name__ == '__main__':
	#main()
	glist,_,_,_=load_data('NCI1',False,1)
	edges=0
	nodes=0
	graphs=len(glist)
	max_nodes=0
	for g in glist:
		edges+=g.edge_mat.shape[-1]
		nodes+=len(g.node_tags)
		if len(g.node_tags)>max_nodes:
			max_nodes=len(g.node_tags)
	print(graphs,nodes/graphs,edges/graphs,max_nodes)
