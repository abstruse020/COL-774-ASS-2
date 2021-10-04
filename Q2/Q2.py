# -*- coding: utf-8 -*-
import cvxopt
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

#%%
### Variables
test_path = 'Q2/train.csv'
d = 5

def read_it(path):
    train_data = pd.read_csv(path, header=None)
    
    new_columns = {train_data.columns[-1]: 'labels'}
    train_data.rename(columns = new_columns, inplace=True)
    
    train_data = train_data[(train_data.labels == d) | (train_data.labels == d+1)]

    return train_data

train_data = read_it(test_path)


print('Filtered data shape',train_data.shape, '\nLast col after:', train_data.columns[-1])
print('uniq labels:', set(train_data.labels))
#%%
X_data = train_data.iloc[:10,0:-1]
X_data = X_data / 255
Y_data = train_data.iloc[:10,-1]
Y_data = (Y_data - 5.5)*2
Y_data = Y_data.tolist()

print(X_data.head()) 
print(Y_data.head())

#%%

class SVM_Linear:
    def __init__(self, X, Y, c):
        self.X = X
        self.Y = Y
        self.c = c
        self.m, self.n = X.shape
        self.aplhas = [0]*len(Y)
        
    def make_parameters(self):
        print("Making parameeters")
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                temp = self.X[i].T @ self.X[j]
                P[i,j] = self.Y[i] * self.Y[j] * temp
        q = np.array([1]*self.m)
        G = np.array([ [1]*self.m, [-1]*self.m ])
        h = np.array([self.c, 0.0])
        A = np.array([self.Y])
        b = np.array([0.0])
        
        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')
        return P,q,G,h,A,b
    
    def learn(self):
        P, q, G, h, A, b = self.make_parameters()
        sol = solvers.qp(P, q, G, h, A, b)
        print("Solving Complete")
        self.aplhas = sol['x']
        print('Primal objective:', sol['primal objective'])
        print('Alphas:', self.aplhas)
        return self.aplhas
    
    def calc_parameters(self):
        W = np.zeros(self.n)
        for i in range(self.m):
            W += self.aplhas[i] * self.Y[i] * self.X[i]
        b1_min = 10000000
        b2_max = 0
        for i in range(self.m):
            comp = W.T @ self.X[i]
            if self.Y[i] == 1:
                if comp < b1_min:
                    b1_min = comp
            if self.Y[i] == -1:
                if comp > b2_max:
                    b2_max = comp
        b = -1.0 * (b1_min + b2_max)/2
        return (W, b)
    
    def accuracy(Y_prediction, Y_label):
        s =0
        for y_hat, y in zip(Y_prediction, Y_label):
            if y_hat == y:
                s += 1
        return s*100/len(Y_label)
    
    def predict(self, X_test, Y_test = []):
        W, b = self.calc_parameters()
        Y_prediction = X_test @ W + b
        print(Y_prediction)
        return Y_prediction


#%%

svm = SVM_Linear(X_data, Y_data, 1.0)

svm.learn()








    
        
        
        