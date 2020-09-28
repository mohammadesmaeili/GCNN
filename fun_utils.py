import numpy as np
import tensorflow as tf
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from fun_side import *


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    train_nodes = np.where(np.sum(y_train, axis=1)==1)[0]
    val_nodes = np.where(np.sum(y_val, axis=1)==1)[0]
    test_nodes = np.where(np.sum(y_test, axis=1)==1)[0]
    
    A = adj.toarray().astype('int32')
    X = features.toarray()
    x = np.argmax(labels, axis=1)

    return A, X, train_nodes, val_nodes, test_nodes, x

def fun_SBM_generator(classes):
    n = 2000
    p = 5 * np.log(n)/n
    q = 1 * np.log(n)/n
    x = np.random.randint(0,classes, size=(n, 1))
    X = np.eye(len(x))
    A = np.zeros((len(x), len(x)), dtype='int')
    for i in range(len(x)):
        ind, _ = np.where(x==x[i])
        A[i, ind] = np.random.binomial(1,p,size=(len(ind),))
        A[ind, i] = A[i, ind]
        ind, _ = np.where(x!=x[i])
        A[i, ind] = np.random.binomial(1,q,size=(len(ind),))
        A[ind, i] = A[i, ind]
    
    nodes = set()
    for i in range(classes):
        temp = set(list(np.squeeze(np.where(x==i)[0])[0:20]))
        nodes = set.union(nodes, temp)
    train_nodes = np.array(list(nodes))
    remain_nodes = np.array(list(set(np.arange(2000))-set(train_nodes)))
    val_nodes = remain_nodes[0:500]
    test_nodes = remain_nodes[500:1500]
    return A, X, train_nodes, val_nodes, test_nodes, x


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def fun_initialization(dataset, A, X, x, classifier=None, radius=None, side=None, Eu=None, epochs=None, beta=None):
    parameters = {}
    if dataset=='cora':
        parameters['classifier'] = 'GBC'
        parameters['radius'] = 4
        parameters['classes'] = tf.keras.utils.to_categorical(x, dtype=int).shape[1]
        parameters['samples'] = len(A)
        parameters['features'] = X.shape[1]
        parameters['side'] = 'Ar'
        parameters['layers'] = [parameters['features'], 128, parameters['classes']]
        parameters['beta'] = 8*10**(-5)*1
        parameters['learning_rate_phase_1'] = 0.01
        parameters['learning_rate_phase_2'] = 0.005
        parameters['epochs'] = 250
        parameters['Eu'] = 50
        parameters['Pth'] = 0.55
        parameters['Fth'] = 0.999
    elif dataset=='citeseer':
        parameters['classifier'] = 'GBC'
        parameters['radius'] = 3
        parameters['classes'] = tf.keras.utils.to_categorical(x, dtype=int).shape[1]
        parameters['samples'] = len(A)
        parameters['features'] = X.shape[1]
        parameters['side'] = 'X'
        parameters['layers'] = [parameters['features'], 128, parameters['classes']]
        parameters['beta'] = 8*10**(-5)*1
        parameters['learning_rate_phase_1'] = 0.01
        parameters['learning_rate_phase_2'] = 0.05
        parameters['epochs'] = 200
        parameters['Eu'] = 80
        parameters['Pth'] = 0.80
        parameters['Fth'] = 0.80
    elif dataset=='pubmed':
        parameters['classifier'] = 'GBC'
        parameters['radius'] = 1
        parameters['classes'] = tf.keras.utils.to_categorical(x, dtype=int).shape[1]
        parameters['samples'] = len(A)
        parameters['features'] = X.shape[1]
        parameters['side'] = 'Ar'
        parameters['layers'] = [parameters['features'], 64, parameters['classes']]
        parameters['beta'] = 4*10**(-4)*1
        parameters['learning_rate_phase_1'] = 0.01
        parameters['learning_rate_phase_2'] = 0.002
        parameters['epochs'] = 200
        parameters['Eu'] = 80
        parameters['Pth'] = 0.7
        parameters['Fth'] = 1.0
    elif dataset=='k-sbm':
        parameters['classifier'] = 'GCN'
        parameters['radius'] = 1
        parameters['classes'] = tf.keras.utils.to_categorical(x, dtype=int).shape[1]
        parameters['samples'] = len(A)
        parameters['features'] = X.shape[1]
        parameters['side'] = 'Ar'
        parameters['layers'] = [parameters['features'], 16, parameters['classes']]
        parameters['beta'] = 5*10**(-5)*1
        parameters['learning_rate_phase_1'] = 0.01
        parameters['learning_rate_phase_2'] = 0.01
        parameters['epochs'] = 300
        parameters['Eu'] = 150
        parameters['Pth'] = 0.5
        parameters['Fth'] = 0.5
        
    if classifier!=None:
        parameters['classifier'] = classifier
    if radius!=None:
        parameters['radius'] = radius
    if side!=None:
        parameters['side'] = side
    if Eu!=None:
        parameters['Eu'] = Eu
    if epochs!=None:
        parameters['epochs'] = epochs
    if beta!=None:
        parameters['beta'] = beta
    return parameters