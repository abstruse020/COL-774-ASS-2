# -*- coding: utf-8 -*-
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

#%%
### Variables
train_path = '~/IITD/COL774-ML/Assignment2/Q2/train.csv'
test_path = '~/IITD/COL774-ML/Assignment2/Q2/test.csv'
d = 5

def read_it(path):
    train_data = pd.read_csv(path, header=None)
    
    new_columns = {train_data.columns[-1]: 'labels'}
    train_data.rename(columns = new_columns, inplace=True)
    
    train_data = train_data[(train_data.labels == d) | (train_data.labels == d+1)]

    return train_data

def preprocessing(data):
    X_data = data.iloc[:,0:-1]
    X_data = X_data / 255
    X_data = X_data.to_numpy()
    Y_data = data.iloc[:,-1]
    Y_data = (Y_data - 5.5)*2
    Y_data = Y_data.tolist()
    return X_data, Y_data

train_data = read_it(train_path)

#%%
X_data , Y_data = preprocessing(train_data)

print(X_data[:5]) 
print(Y_data[:5])

#%%

class SVM_Linear:
    def __init__(self, X, Y, c):
        self.X = X
        self.Y = Y
        self.c = c
        self.m, self.n = X.shape
        self.alphas = [0]*len(Y)
        
    def gaussian_kernel(self, x, z, sigma_sq = 10):
        x1 = np.linalg.norm(x - z)
        val = np.exp(-1* x1/(2* sigma_sq))
        return val
        
    def make_parameters(self, kernel = 'Linear'):
        print("Making parameeters")
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                temp = 1
                if kernel == 'Linear':
                    temp = self.X[i].T @ self.X[j]
                elif kernel == 'Gaussian':
                    temp = self.gaussian_kernel(self.X[i], self.X[j])
                P[i,j] = self.Y[i] * self.Y[j] * temp
        q = np.array([1]*self.m)
        G1 = np.diag([1]*self.m)
        G2 = np.diag([-1]*self.m)
        G = np.append(G1, G2, axis = 0)
        h = np.append([self.c]*self.m, [0.0]*self.m, axis =0)
        A = np.array([self.Y])
        b = np.array([0.0])
        
        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')
        return P,q,G,h,A,b
    
    def learn(self, kernel = 'Linear'):
        t1 = datetime.datetime.now()
        P, q, G, h, A, b = self.make_parameters(kernel)
        t2 = datetime.datetime.now()
        print('Time to make params:', t2 -t1)
        print('Solving Started->')
        sol = solvers.qp(P, q, G, h, A, b)
        self.alphas = sol['x']
        print('Primal objective:', sol['primal objective'])
        print('Alphas:\n', self.alphas[:10])
        return self.alphas
    
    def calc_parameters(self):
        W = np.zeros(self.n)
        for i in range(self.m):
            W += self.alphas[i] * self.Y[i] * self.X[i]
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
    
    def accuracy(self, Y_prediction, Y_label):
        s =0
        for y_hat, y in zip(Y_prediction, Y_label):
            if y_hat == y:
                s += 1
        return s*100/len(Y_label)
    
    def predict(self, X_test, Y_test = []):
        W, b = self.calc_parameters()
        Y_prediction = X_test @ W + b
        Y_prediction = [ 1 if y_hat>0 else -1 for y_hat in Y_prediction]
        print('Accuracy:',self.accuracy(Y_prediction, Y_test))
        return Y_prediction
    
    def predict_gaussian(self, X_test, Y_test = []):
        Y_prediction = np.zeros(X_test.shape[0])
        b1_min = 1000000
        b2_max = 0
        for j in range(self.m):
            comp = 0
            for i in range(self.m):    
                comp += self.alphas[i] *self.Y[i] * self.gaussian_kernel(self.X[i], self.X[j])
            if self.Y[j] == 1:
                if b1_min > comp:
                    b1_min = comp
            else:
                if b2_max < comp:
                    b2_max = comp
        b = -1* (b1_min + b2_max)/2
        
        for i in range(X_test.shape[0]):
            for j in range(self.m):
                Y_prediction[i] += self.alphas[j] * self.Y[j] * self.gaussian_kernel(self.X[j], X_test[i]) + b
        Y_prediction = [ 1 if y_hat>0 else -1 for y_hat in Y_prediction]
        if len(Y_test) == len(X_test):
            print('Accuracy:', self.accuracy(Y_prediction, Y_test))
        return Y_prediction
        
                    

#%%

svm = SVM_Linear(X_data, Y_data, 1.0)

alphas = svm.learn()

#%%

#print(max(alphas))
svm.predict(X_data[:], Y_data[:])

#%%

test_data = read_it(test_path)

X_test_data, Y_test_data = preprocessing(test_data)

#%%
test_predictions = svm.predict(X_test_data, Y_test_data)

#%%
####
# Things left -->
# 1. Calculate the support vectors 
# 2. Check if Epsilon_i is required or not ( as only c came and not epsilon)

svm_gaussian = SVM_Linear(X_data, Y_data, 1.0)

gau_alpha = list(svm_gaussian.learn('Gaussian'))

#%%
        
predict_gau = svm_gaussian.predict_gaussian(X_data, Y_data)

#%%
predict_test_gau = svm_gaussian.predict_gaussian(X_test_data, Y_test_data)
#%%

zero_count =0
for al in alphas:
    if al == 0:
        zero_count +=1
print(min(alphas))


#%%

#Part c -- SVM for Linear and Gaussian using LIBSVM

from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_predict

def Libsvm_linear(X_train, Y_train, X_test, Y_test):
    
    prob = svm_problem(Y_train, X_train.tolist())
    
    params = svm_parameter("-s 0 -c 1 -t 0")
    
    svm_lin = svm_train(prob, params)
    
    _, _, prediction = svm_predict(Y_test, X_test, svm_lin)
        
    return prediction

def Libsvm_gaussian(X_train, Y_train, X_test, Y_test, c, g):
    
    prob = svm_problem(Y_train, X_train.tolist())
    
    params = svm_parameter("-s 0 -c " + str(c)+" -t 2 -g " + str(g))
    
    svm_lin = svm_train(prob, params)
    
    _, _, prediction = svm_predict(Y_test, X_test, svm_lin)
        
    return prediction

#%%
# On Train data

Libsvm_linear(X_data, Y_data, X_data, Y_data)

_ = Libsvm_gaussian(X_data, Y_data, X_data, Y_data, 10, 0.01)

#%%
# On Test data

Libsvm_linear(X_data, Y_data, X_test_data, Y_test_data)

_ = Libsvm_gaussian(X_data, Y_data, X_test_data, Y_test_data, 10, 0.01)
