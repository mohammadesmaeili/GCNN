import scipy
import numpy as np
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier
from fun_tensorflow import *
from fun_utils import *


def fun_loading_Ar(A, dataset, radius):
    file = 'Side.' + dataset + '.R' + str(radius)
    if dataset=='cora' or dataset=='citeseer' or dataset=='pubmed':
        try:
            A_r_sparse = scipy.sparse.load_npz(file+'.npz').astype('float32')
        except:
            G = nx.Graph(A)
            dic = {}
            for i in range(len(A)):
                dic[i] = set(nx.ego_graph(G, i, radius=radius))
            A_r = np.zeros(shape=A.shape)
            for i in range(len(A)):
                for j in range(i, len(A)):
                    A_r[i,j] = len(set.intersection(dic[i], dic[j]))/len(set.union(dic[i], dic[j]))
                    A_r[j,i] = A_r[i,j]
            A_r_sparse = scipy.sparse.csr_matrix(A_r)
            A_r_sparse = A_r_sparse.astype('float32')
            scipy.sparse.save_npz(file, A_r_sparse)
    elif dataset=='k-sbm':
        G = nx.Graph(A)
        dic = {}
        for i in range(len(A)):
            dic[i] = set(nx.ego_graph(G, i, radius=radius))
        A_r = np.zeros(shape=A.shape)
        for i in range(len(A)):
            for j in range(i, len(A)):
                A_r[i,j] = len(set.intersection(dic[i], dic[j]))/len(set.union(dic[i], dic[j]))
                A_r[j,i] = A_r[i,j]
        A_r_sparse = scipy.sparse.csr_matrix(A_r)
        A_r_sparse = A_r_sparse.astype('float32')
    return A_r_sparse
    
        

def fun_synthetic_side(dataset, initial, x, alpha, train_nodes):
    R = np.random.binomial(1, 1-alpha, size=(len(x),1))
    ind, _ = np.where(R>0)
    z = x.copy()
    for i in ind:
        if not i in train_nodes:
            for j in range(initial['classes']):
                if x[i]==j:
                    z[i] = np.random.randint(j+1, j+initial['classes'], 1)%initial['classes']
    z = z.reshape(-1)
    return z

def fun_side_information_Ar(dataset, classifier, A, X, x, train_nodes, initial, seed=None):
    Ar = fun_loading_Ar(A, dataset, initial['radius'])
    Ar = Ar.toarray()
    #Ar = normalize(Ar, norm='l1', axis=1)
    if classifier=='GBC':
        model = GradientBoostingClassifier(learning_rate=0.3, n_estimators=500, max_depth=5, random_state=seed)
        model.fit(Ar[train_nodes,:], x[train_nodes])
        z = model.predict(Ar)
    elif classifier=='GCN':
        z = fun_GCN_correlated(A, Ar, x, initial, train_nodes, seed=seed)
    elif classifier=='NN':
        z = fun_NN(Ar, Ar[train_nodes,:], x[train_nodes], initial, seed=seed)
    return z

def fun_side_information_X(dataset, classifier, A, X, x, train_nodes, initial, seed=None):
    X = fun_normalization(X, 'l1')
    if classifier=='GBC':
        model = GradientBoostingClassifier(learning_rate=0.3, n_estimators=500, max_depth=5, random_state=seed)
        model.fit(X[train_nodes,:], x[train_nodes])
        z = model.predict(X)
    elif classifier=='GCN':
        z = fun_GCN_correlated(A, X, x, initial, train_nodes, seed=seed)
    elif classifier=='NN':
        z = fun_NN(X, X[train_nodes,:], x[train_nodes], initial, seed=seed)
    return z