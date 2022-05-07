# Linkage Based Face Clustering By Graph Convolutional Networks

## INTRODUCTION

We implement a method for face clustering that is both accurate and scalable, proposed in this research paper:
[**Linkage Based Face Clustering via Graph Convolutional Networks**](https://arxiv.org/abs/1903.11306)

The problem is to categorise a group of faces based on their possible identities. This problem is formulated as a link prediction problem: if two faces have the same identity, there is a link between them. The basic concept is that the local context in the feature space surrounding an instance(face) carries a wealth of information about the linkage relationship between this instance and its neighbours. The graph convolution network (GCN) is used to do reasoning and infer the likelihood of linkage between pairings in the sub-graphs by creating sub-graphs around each instance as input data, which describe the local context.


## REQUIREMENTS

The project has been implemented in python 3.9.7, pytorch build 1.11.0, CUDA 11.3, Windows 10 OS

## DATASET

Subsets of CASIA, IJB-B dataset have been used for training and testing dataset respectively.

## TRAINING

We train the GCN to learn the weights. Training has been done in batches of 32.

## TESTING

During inference, the test script will dynamically output the pairwise precision/recall/accuracy. After each subgraph is processed, the test script will output the final B-Cubed precision/recall/F-score and NMI score.

## MODULES

### AUXI

  1. **feeder:**
  - **__init__:** Generate a sub-graph from the feature graph centered at some node, and now the sub-graph has a fixed depth, i.e. 2
  - **__getitem__:** return the vertex feature and the adjacent matrix A, together with the indices of the center node and its 1-hop nodes
  2. **graph:**
  - **class data:** initialise the data, return links, add links in the graph
  - **connected_components:** searching the connected components      
  - **connected_components_constraint:** only use edges whose scores are above *th* if a component is larger than *max_sz*, all the nodes in this component are added into *remain* and returned for next iteration.

  3. **logging:** used for managing the logs (checkpoints).
  4. **meters:** Computes and stores the average and current value.
  5. **serialization:** used to save and load checkpoints.

### GCN

  1. **MeanAggregator:** Performs mean aggregation on the adjacency matrix of IPS **D<sup>-1/2</sup>AD<sup>-1/2</sup>** to obtain aggregated matrix G.
  2. **GraphConv:** Performs the convolution operation on aggregated matrix **(X||G.X).W** (for a single layer)
  3. **GCN:** performs forward propagation

### ARGS

  1. **train args:** arguments passed during training are initialised in this module
  2. **test args:** arguments passed during testing rae initialised in this module
