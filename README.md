# Two-Level-Graph-Neural-Network

Code for "[Two-Level-Graph-Neural-Network](https://arxiv.org/abs/2101.)"

## Overview

- main.py: the entry of model training and testing.
- CountMotif_and_node.py: the subgraph searching algorithm.
- tlgnn.py: including the basic layers we used in the main model.
- gen_PNG.py: the graph sythetic code. 

## Datasets

- MUTAG dataset consists of 188 chemical compounds divided into two
  classes according to their mutagenic effect on a bacterium.
- PTC:PTC is a collection of 344 chemical compounds represented as graphs which report the carcinogenicity for rats. 
  There are 19 node labels for each node.
- The NCI1 dataset comes from the cheminformatics domain, where each input graph is used as representation of a chemical compound: each vertex stands for an atom of the molecule, and edges between vertices represent bonds between atoms. This dataset is relative to anti-cancer screens where the chemicals are assessed as positive or negative to cell lung cancer.
- IMDB-BINARY is a movie collaboration dataset that consists of the ego-networks of 1,000 actors/actresses who played roles in movies in IMDB. In each graph, nodes represent actors/actress, and there is an edge between them if they appear in the same movie. These graphs are derived from the Action and Romance genres.
- IMDB-MULTI is a relational dataset that consists of a network of 1000 actors or actresses who played roles in movies in IMDB. A node represents an actor or actress, and an edge connects two nodes when they appear in the same movie. In IMDB-MULTI, the edges are collected from three different genres: Comedy, Romance and Sci-Fi.
- PROTEINS is a dataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids and two nodes are connected by an edge if they are less than 6 Angstroms apart.

## Setting

1. setting python env using pip install -r requirements.txt
2. python main.py  (all the parameters could be viewed in the train.py)

## Parameters
````
     --dataset #dataset name
     --device #which gpu to use if any
     --batch_size #input batch size for training
     --iters_per_epoch #number of iterations per each epoch
     --epochs #umber of epochs to train
     --lr #learning rate
     --seed #random seed for splitting the dataset into 10
     --fold_idx  #the index of fold in 10-fold validation. Should be less then 10.
     --num_layers #number of layers for MLP EXCLUDING the input one
     --hidden_dim #number of hidden units 
     --final_dropout #final layer dropout
     --graph_pooling_type #Pooling for over nodes in a graph: sum or average
     --neighbor_pooling_type #Pooling for over neighboring nodes: sum, average or max
     --learn_eps #Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though
     --degree_as_tag #let the input node features be the degree of nodes (heuristics for unlabeled graph)
     --filename # output file
 ````

 

## Reference
````

````
