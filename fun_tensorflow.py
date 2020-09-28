# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:38:52 2020

@author: mohammad
"""
import numpy as np
import tensorflow as tf
import scipy
from fun_utils import *
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score


def fun_accuracy(x,y):
    return accuracy_score(x,y)

def fun_normalization(x, norm):
    Norm_X = normalize(x, norm=norm)
    return Norm_X

def fun_adj_preprocessing(A, norm):
    A_hat = A+np.eye(len(A))
    L = fun_normalization(A_hat, norm=norm)
    return scipy.sparse.csr_matrix(L).astype('float32')


def fun_convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def glorot(shape, name=None, seed=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32, seed=seed)
    return tf.Variable(initial, name=name)



def fun_Placeholders(samples, features, classes):
    Input = tf.placeholder(dtype=tf.float32, shape=(samples, features), name='Input')
    Output = tf.placeholder(dtype=tf.float32, shape=(samples, classes), name='Output')
    Nodes = tf.placeholder(dtype=tf.int32, shape=(None), name='Nodes')
    Learning = tf.placeholder(dtype=tf.float32, shape=(None), name='Learning')
    return Input, Output, Nodes, Learning

def fun_Placeholders_NN(features, classes):
    Input = tf.placeholder(dtype=tf.float32, shape=(None, features), name='Input')
    Output = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='Output')
    return Input, Output


def fun_Hidden_Layers_gcnn(Input, layers, A, seed=None):
    parameters = {}
    parameters['H'+str(0)] = Input
    for i in range(len(layers)-1):
        parameters['W'+str(i)] = tf.get_variable(name='W'+str(i),
                                                 shape=(layers[i], layers[i+1]), 
                                                 dtype=tf.float32, 
                                                 initializer=tf.keras.initializers.glorot_uniform(seed=seed))
        parameters['Z'+str(i+1)] = tf.matmul(tf.sparse.sparse_dense_matmul(A,parameters['H'+str(i)]), parameters['W'+str(i)])
        if i==len(layers)-2:
            parameters['H'+str(i+1)] = tf.nn.softmax( parameters['Z'+str(i+1)] )
        else:
            parameters['H'+str(i+1)] = tf.nn.relu( parameters['Z'+str(i+1)] )
    return parameters

def fun_Hidden_Layers_NN(Input, layers, seed=None):
    parameters = {}
    parameters['H'+str(0)] = Input
    for i in range(len(layers)-1):
        parameters['W'+str(i)] = tf.get_variable(name='W'+str(i),
                                                 shape=(layers[i], layers[i+1]), 
                                                 dtype=tf.float32, 
                                                 initializer=tf.keras.initializers.glorot_uniform(seed=seed))
        parameters['Z'+str(i+1)] = tf.matmul(parameters['H'+str(i)], parameters['W'+str(i)])
        if i==len(layers)-2:
            parameters['H'+str(i+1)] = tf.nn.softmax( parameters['Z'+str(i+1)] )
        else:
            parameters['H'+str(i+1)] = tf.nn.relu( parameters['Z'+str(i+1)] )
    return parameters



def fun_Hidden_Layers_gcnn_new(Input, layers, A, seed=None):
    parameters = {}
    parameters['H'+str(0)] = Input
    for i in range(len(layers)-1):
        parameters['W'+str(i)] = glorot( shape=(layers[i], layers[i+1]), name='W'+str(i), seed=seed )
        parameters['Z'+str(i+1)] = tf.matmul(tf.sparse.sparse_dense_matmul(A,parameters['H'+str(i)]), parameters['W'+str(i)])
        if i==len(layers)-2:
            parameters['H'+str(i+1)] = tf.nn.softmax( parameters['Z'+str(i+1)] )
        else:
            parameters['H'+str(i+1)] = tf.nn.relu( parameters['Z'+str(i+1)] )
    return parameters

def fun_regularization(beta, parameters, layers):
    reg = 0
    for i in range(len(layers)-1):
        reg += beta*tf.nn.l2_loss(parameters['W'+str(i)])
    return reg

def fun_NN(x, x_train, y_train, initial, seed=None):
    y_train = tf.keras.utils.to_categorical(y_train, initial['classes'], dtype=int)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    features = x_train.shape[1]
    classes = initial['classes']
    layers = [features, 16, classes]
    learning_rate = 0.01
    beta = 5*10**(-4)*1

    tf.reset_default_graph()
    Input, Output = fun_Placeholders_NN(features, classes)
    parameters = fun_Hidden_Layers_NN(Input, layers, seed=seed)
    labels = Output
    logits = parameters['Z'+str(len(layers)-1)]
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) )
    cost += fun_regularization(beta, parameters, layers)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    epochs = 200
    loss_train = []
    accuracy_train = []
    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        check = 1
        feed_train={Input: x_train, Output: y_train}
        feed_out={Input: x, Output: y_train}
        while check:
            epoch += 1
            sess.run(optimizer, feed_dict=feed_train)
            y_train_pre, out = sess.run([parameters['H'+str(len(layers)-1)], cost], feed_dict=feed_train)
            loss_train.append(out)

            accuracy_train.append( fun_accuracy(np.argmax(y_train, axis=1), np.argmax(y_train_pre, axis=1)) )
            #print(accuracy_train[-1])
            if epoch>=epochs or np.mean(accuracy_train[::-1][0:10])==1:
                check = 0
        y_pre = sess.run(parameters['H'+str(len(layers)-1)], feed_dict=feed_out)
        return np.argmax(y_pre, axis=1)
    
def fun_GCN_correlated(A, x_train, y_train, initial, train_nodes, seed=None):
    y_train = tf.keras.utils.to_categorical(y_train, initial['classes'], dtype=int)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    samples = x_train.shape[0]
    features = x_train.shape[1]
    classes = initial['classes']
    layers = [features, 16, classes]
    learning_rate = 0.01
    beta = 5*10**(-4)*1

    
    tf.reset_default_graph()
    Lst = fun_convert_sparse_matrix_to_sparse_tensor(fun_adj_preprocessing(A, 'l1'))
    Input, Output, Nodes, Learning_Rate = fun_Placeholders(samples, features, classes)
    parameters = fun_Hidden_Layers_gcnn(Input, layers, Lst, seed=seed)
    labels = tf.gather( Output, Nodes, axis=0)
    logits = tf.gather( parameters['Z'+str(len(layers)-1)], Nodes, axis=0)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) )
    cost += fun_regularization(beta, parameters, layers)
    optimizer = tf.train.AdamOptimizer(Learning_Rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    epochs = 100
    loss_train = []
    accuracy_train = []
    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        check = 1
        feed_train={Input: x_train, Output: y_train, Nodes: train_nodes, Learning_Rate: learning_rate}
        while check:
            epoch += 1
            sess.run(optimizer, feed_dict=feed_train)
            out, y_train_pre = sess.run([cost, parameters['H'+str(len(layers)-1)]], feed_dict=feed_train)
            loss_train.append(out)

            accuracy_train.append( fun_accuracy(np.argmax(y_train[train_nodes], axis=1), np.argmax(y_train_pre[train_nodes], axis=1)) )
            #print(accuracy_train[-1])
            if epoch>=epochs or np.mean(accuracy_train[::-1][0:10])==1:
                check = 0
        return np.argmax(y_train_pre, axis=1)