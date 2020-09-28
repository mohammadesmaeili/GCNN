"""
This code runs the proposed architecture in the paper, while the synthetic side information is not combined in the feature matrix. 
Please choose the dataset and run the code. 
For the synthetic side information choose the value for the parameter alpha. 
This code uses the hyperparameters defined in the paper by default. 
Other values for the hyperparameters can be chosen. 
"""

from fun_test import fun_simulator
dataset = 'k-sbm' #'cora', 'citeseer', 'pubmed', 'k-sbm'
k = 5 #for cora, citeseer, and pubmed: None 
      #for k-sbm: 3,4,5
side_generator = 'extracted' #'extracted', 'synthetic'
classifier = None #None, 'NN', 'GBC', 'GCN'
alpha = 0.7 #for extracted side information: None
            #for synthetic side information: 0.3, 0.5, 0.7
radius = None
side = None #None, 'X', 'Ar'
seed = None
Eu = None
epochs = None
beta = None

out = fun_simulator(dataset=dataset, 
                    side_generator=side_generator, 
                    k=k, 
                    alpha=alpha, 
                    classifier=classifier, 
                    radius=radius, 
                    side=side, 
                    seed=seed, 
                    Eu=Eu, 
                    epochs=epochs, 
                    beta=beta)