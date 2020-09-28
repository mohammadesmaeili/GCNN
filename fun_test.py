import numpy as np
import tensorflow as tf
from fun_utils import *
from fun_side import *
from fun_tensorflow import *

def fun_simulator(dataset, side_generator, k=None, alpha=None, classifier=None, radius=None, side=None, seed=None, Eu=None, epochs=None, beta=None):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if dataset=='k-sbm':
        A, X, train_nodes, val_nodes, test_nodes, x = fun_SBM_generator(k)
    elif dataset=='cora' or dataset=='citeseer' or dataset=='pubmed':
        A, X, train_nodes, val_nodes, test_nodes, x = load_data(dataset)

    initial = fun_initialization(dataset, A, X, x, classifier=classifier, radius=radius, side=side, Eu=Eu, epochs=epochs, beta=beta)
    
    
    if side_generator =='extracted':
        print('Extracting side information ...')
        if initial['side']=='Ar':
            z = fun_side_information_Ar(dataset, initial['classifier'], A, X, x, train_nodes, initial, seed=seed)
        if initial['side']=='X':
            z = fun_side_information_X(dataset, initial['classifier'], A, X, x, train_nodes, initial, seed=seed)
        print('Side information was extracted by ' + initial['classifier'] + ' and ' + initial['side'] + '.')
    elif side_generator =='synthetic':
        print('Generating synthetic side information ...')
        z = fun_synthetic_side(dataset, initial, x, alpha, train_nodes)
        print('Side information was generated with alpha=', str(alpha) + '.')
        
    print(initial)
    print('Fixed Training Accuracy Score:', fun_accuracy(x[train_nodes], z[train_nodes]))
    print('Validation Accuracy Score:', fun_accuracy(x[val_nodes], z[val_nodes]))
    print('Test Accuracy Score:', fun_accuracy(x[test_nodes], z[test_nodes]))
    print('Runing Active GCN ...')
    
    x_train = fun_normalization(X, 'l1')
    y_train = tf.keras.utils.to_categorical(x, initial['classes'], dtype=int)
    y_train_noisy = tf.keras.utils.to_categorical(z, initial['classes'], dtype=int)
    y_train_noisy[train_nodes,:] = y_train[train_nodes,:]
    train_nodes_fix = train_nodes.copy()
    
    tf.reset_default_graph()
    Lst = fun_convert_sparse_matrix_to_sparse_tensor(fun_adj_preprocessing(A, 'l1'))
    Input, Output, Nodes, Learning_Rate = fun_Placeholders(initial['samples'], initial['features'], initial['classes'])
    parameters = fun_Hidden_Layers_gcnn(Input, initial['layers'], Lst, seed=seed)
    labels = tf.gather( Output, Nodes, axis=0)
    logits = tf.gather( parameters['Z'+str(len(initial['layers'])-1)], Nodes, axis=0)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) )
    cost += fun_regularization(initial['beta'], parameters, initial['layers'])
    optimizer = tf.train.AdamOptimizer(Learning_Rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    loss_train = []
    loss_val = []
    accuracy_train_fix = []
    accuracy_train = []
    accuracy_test = []
    accuracy_val = []
    ch = 1
    with tf.Session() as sess:
        sess.run(init)
        epoch = 0
        check = 1
        feed_val={Input: x_train, Output: y_train, Nodes: val_nodes, Learning_Rate: initial['learning_rate_phase_1']}
        feed_train={Input: x_train, Output: y_train_noisy, Nodes: train_nodes, Learning_Rate: initial['learning_rate_phase_1']}
        while check:
            epoch += 1
            sess.run(optimizer, feed_dict=feed_train)
            y_train_pre, out = sess.run([parameters['H'+str(len(initial['layers'])-1)], cost], feed_dict=feed_train)
            loss_train.append(out)
            loss_val.append(cost.eval(feed_dict=feed_val))
    
            y_pre = np.argmax(y_train_pre, axis=1)
            accuracy_train_fix.append( fun_accuracy(x[train_nodes_fix], y_pre[train_nodes_fix]) )
            accuracy_train.append( fun_accuracy(x[train_nodes], y_pre[train_nodes]) )
            accuracy_test.append( fun_accuracy(x[test_nodes], y_pre[test_nodes]) )
            accuracy_val.append( fun_accuracy(x[val_nodes], y_pre[val_nodes]) )
            
            if (epoch>=initial['Eu']) and ch:
                Ind_1,_ = np.where(y_train_pre>initial['Pth'])
                Ind_2 = np.squeeze(np.where(np.argmax(y_train_pre, axis=1)==z.T))
                train_nodes = set.intersection( set(list(Ind_1)), set(list(Ind_2)) )
                train_nodes = set.union(train_nodes, set(list(train_nodes_fix)))
                train_nodes = np.array(list(train_nodes))
                feed_train={Input: x_train, Output: y_train_noisy, Nodes: train_nodes, Learning_Rate: initial['learning_rate_phase_2']}
            if epoch>=initial['epochs']:
                check = 0
            if epoch>=initial['Eu'] and accuracy_train_fix[-1]<=initial['Fth']:
                ch = 0
            if initial['Eu']==initial['epochs']:
                if loss_val[-1]>np.mean(loss_val[::-1][0:2]):
                    check = 0
            else:
                if epoch>initial['Eu']+50 and loss_val[-1]>np.mean(loss_val[::-1][0:2]) and ch==0:
                    check = 0
            if epoch<initial['Eu'] and loss_val[-1]>np.mean(loss_val[::-1][0:2]):
                initial['Eu'] = epoch
        print('Fixed Training Accuracy Score:', accuracy_train_fix[-1])
        print('Training Accuracy Score:', accuracy_train[-1])
        print('Validation Accuracy Score:', accuracy_val[-1])
        print('Test Accuracy Score:', accuracy_test[-1])
    return accuracy_test[-1]