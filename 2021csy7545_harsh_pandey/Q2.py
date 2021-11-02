# -*- coding: utf-8 -*-
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import statistics
from math import floor
import sys


#%%


def read_it(path, filter_it = True):
    train_data = pd.read_csv(path, header=None)
    new_columns = {train_data.columns[-1]: 'labels'}
    train_data.rename(columns = new_columns, inplace=True)
    if filter_it:
        train_data = train_data[(train_data.labels == d) | (train_data.labels == d+1)]

    return train_data

# For 2 (B) reading data
def fancy_read(path):
    train_data = pd.read_csv(path, header=None)
    new_columns = {train_data.columns[-1]: 'labels'}
    train_data.rename(columns = new_columns, inplace=True)
    
    d_data = {}
    for i in range(10):
        d_data[i] = train_data[train_data.labels == i].iloc[:,0:-1]
        d_data[i] = d_data[i] / 255
        d_data[i] = d_data[i].to_numpy()
    
    return d_data

def preprocessing(data, process_y = True):
    X_data = data.iloc[:,0:-1]
    X_data = X_data / 255
    X_data = X_data.to_numpy()
    Y_data = data.iloc[:,-1]
    if process_y:
        Y_data = (Y_data - (d + 0.5))*2
    Y_data = Y_data.tolist()
    return X_data, Y_data


#%%

class SVM_class:
    def __init__(self, X, Y, c=1, gama = 0.05):
        self.X = X
        self.Y = Y
        self.c = c
        self.gama = gama
        self.m, self.n = X.shape
        self.alphas = [0]*len(Y)
        self.b_gau = None
    
    def gaussian_kernel(self, x, z, gama = 0.05):
        x1 = np.linalg.norm(x - z)**2
        val = np.exp(-1* x1 * gama)
        return val
        
    def make_parameters(self, kernel = 'Linear'):
        P = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                temp = 1
                if kernel == 'Linear':
                    temp = self.X[i].T @ self.X[j]
                elif kernel == 'Gaussian':
                    temp = self.gaussian_kernel(self.X[i], self.X[j], self.gama)
                P[i,j] = self.Y[i] * self.Y[j] * temp
        q = np.array([-1]*self.m)
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
    
    def learn(self, kernel = 'Linear', print_op = True):
        t1 = datetime.datetime.now()
        
        print('Making parameters...') if print_op else None
        P, q, G, h, A, b = self.make_parameters(kernel)
        
        print('Solving Started...') if print_op else None
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        
        t2 = datetime.datetime.now()
        print('Time to learn:', t2 -t1) if print_op else None
        
        self.alphas = sol['x']
        alpha_threshold = (min(self.alphas)  + max(self.alphas))/1000
        nSV = 0
        for i in range(len(self.alphas)):
            if self.alphas[i] < alpha_threshold:
                self.alphas[i] = 0
            else:
                nSV += 1
                
        print("nSV:", nSV, 'With threshold:', alpha_threshold) if print_op else None
        print('Primal objective:', sol['primal objective']) if print_op else None
        return self.alphas
    
    def calc_parameters(self):
        W = np.zeros(self.n)
        for i in range(self.m):
            W += self.alphas[i] * self.Y[i] * self.X[i]
        b1_min = 10000000
        b2_max = -10000000
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
    
    def predict(self, X_test, Y_test = [], print_op = True):
        W, b = self.calc_parameters()
        Y_prediction = X_test @ W + b
        Y_prediction = [ 1 if y_hat>0 else -1 for y_hat in Y_prediction]
        if print_op:
            print('Accuracy:',self.accuracy(Y_prediction, Y_test))
        return Y_prediction
    
    def predict_gaussian(self, X_test, Y_test = [], print_op= True):
        Y_prediction = np.zeros(X_test.shape[0])
        
        #Calculating b
        if self.b_gau == None:
            b1_min = 1000000
            b2_max = -1000000
            for j in range(self.m):
                comp = 0
                for i in range(self.m):
                    comp += self.alphas[i] *self.Y[i] * self.gaussian_kernel(self.X[i], self.X[j], self.gama)
                if self.Y[j] == 1 and b1_min > comp:
                    b1_min = comp
                elif self.Y[j] == -1 and b2_max < comp:
                    b2_max = comp
            self.b = -1.0* (b1_min + b2_max)/2
        b = self.b
        
        # For prediction using kernel (X@W + b)
        for i in range(X_test.shape[0]):
            for j in range(self.m):
                Y_prediction[i] += self.alphas[j] * self.Y[j] * self.gaussian_kernel(self.X[j], X_test[i], self.gama)
            Y_prediction[i] += b
            
        # specifying predicted values to class
        Y_prediction = [ 1 if y_hat >= 0.0 else -1 for y_hat in Y_prediction]
        if len(Y_test) == len(X_test) and print_op:
            print('Accuracy:', self.accuracy(Y_prediction, Y_test))
        return Y_prediction


#%%

#Part c -- SVM for Linear and Gaussian using LIBSVM

from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_predict

class Libsvm_class:
    def __init__(self, X_train, Y_train, c = 1, gama = 0.05):
        self.problem = svm_problem(Y_train, X_train.tolist())
        self.c = c
        self.gama = gama
        self.model = None
    
    def learn(self, kernel = 'Linear', print_op = True):
        param = None
        t1 = datetime.datetime.now()
        input_params =''
        if kernel == 'Linear':
            input_params = "-s 0 -c " + str(self.c) + " -t 0"
        elif kernel == 'Gaussian':
            input_params = "-s 0 -c " + str(self.c) + " -t 2 -g "+ str(self.gama)
        if not print_op:
            input_params += ' -q'
        param = svm_parameter(input_params)
        
        self.model = svm_train(self.problem, param)
        if print_op:
            print('Time to learn:', datetime.datetime.now() - t1)
        
    def predict(self, X, Y, print_op = True):
        pred = None
        if print_op:
            pred, _, _ = svm_predict(Y, X, self.model)
        else:
            pred, _, _ = svm_predict(Y, X, self.model, '-q')
        return pred


#%%

class One_vs_one_classifier:
    def __init__(self, dict_data):
        self.dict_data = dict_data
        self.models = {}
        self.svm_class = 'cvxopt'
        
    def learn(self, c= 1, gama= 0.05, svm_class= 'cvxopt', print_op = True):
        print("Learning models...") if print_op else None
        t1 = datetime.datetime.now()
        self.svm_class = svm_class
        for i in range(10):
            for j in range(i+1,10):
                X = np.append(self.dict_data[i], self.dict_data[j],  axis = 0)
                Y = np.append([1.0]*len(self.dict_data[i]), [-1.0]*len(self.dict_data[j]))
                if svm_class == 'cvxopt':
                    self.models[(i,j)] = SVM_class(X, Y, c, gama)
                elif svm_class == 'livsvm':
                    self.models[(i,j)] = Libsvm_class(X, Y, c, gama)
                self.models[(i,j)].learn('Gaussian',print_op = False)
        print("Time to learn:", datetime.datetime.now() - t1) if print_op else None
        return self.models
    
    def predict(self, X_test, Y_test, print_op = True):
        print("Doing Prediction...") if print_op else None
        t1 = datetime.datetime.now()
        predictions_mesh = []
        predictions = [0]*len(X_test)
        for i in range(10):
            for j in range(i+1, 10):
                pred = None
                if self.svm_class == 'cvxopt':
                    pred = self.models[(i,j)].predict_gaussian(X_test, Y_test, False)
                elif self.svm_class == 'livsvm':
                    pred = self.models[(i, j)].predict(X_test, Y_test, print_op = False)
                pred = [i if p == 1 else j for p in pred]
                predictions_mesh.append(pred)
        predictions_mesh = np.transpose(predictions_mesh)
        
        #Taking the highest freq value and highest val if tie
        for i in range(len(predictions)):
            predictions_mesh[i].sort()
            predictions[i] = statistics.mode(predictions_mesh[i][::-1])
        t2 = datetime.datetime.now()
        print("Time to predict:", t2 - t1) if print_op else None
        #print(predictions)
        acc = self.accuracy(predictions, Y_test)
        print("Accuracy:", acc) if print_op else None
        return predictions, acc
    
    def accuracy(self, Y_pred, Y_label):
        s =0
        for y_hat, y in zip(Y_pred, Y_label):
            if y_hat == y:
                s +=1
        return 100*s/len(Y_label)
    
    def confusion_matrix(self, predictions, labels, path):
        n = 10
        show_wrong_data = 10
        conf_matrix = [[0 for i in range(n)] for j in range(n)]
        print("Few wrongly classified datas:")
        for y_hat, y in zip(predictions, labels):
            conf_matrix[y][y_hat] +=1
            if y != y_hat and show_wrong_data !=0:
                print('actual:', y, '\t', 'predicted:', y_hat)
                show_wrong_data -= 1
        
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.9)
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[0])):
                ax.text(x=j, y=i,s=conf_matrix[i][j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig(path)
        return conf_matrix



#%%

def structure_x_y(X, Y):
    train_data = np.append(X, np.array(Y).reshape(X.shape[0],-1), axis = 1)
    train_data = pd.DataFrame(train_data)
    new_columns = {train_data.columns[-1]: 'labels'}
    train_data.rename(columns = new_columns, inplace=True)
    
    d_data = {}
    for i in range(10):
        d_data[i] = train_data[train_data.labels == i].iloc[:,0:-1]
        d_data[i] = d_data[i].to_numpy()
    
    return d_data

def validation_set(K, X_train, Y_train, X_test, Y_test):
    gama = 0.05
    c_s = [10**-5, 10**-3, 1, 5, 10]
    accuracy_c = [0.0]*len(c_s)
    accuracy_test = [0.0]*len(c_s)
    m = len(Y_train)
    X_set = [None]*K
    Y_set = [None]*K
    for k in range(K):
        X_set[k] = X_train[floor(k*m/K) : floor((k+1)*m/K)]
        Y_set[k] = Y_train[floor(k*m/K) : floor((k+1)*m/K)]
    
    for c,i in zip(c_s, range(len(c_s))):
        print('-------------For c=', c ,'-----------')
        acc_c = []
        acc_t = []
        for k in range(K):
            print('\tSet',k+1, end='\t')
            X = np.empty((0, X_train.shape[1]), float)
            Y = np.empty((0, 1), float)
            #making train data for learning except kth
            for j in range(K):
                if j == k:
                    continue
                X = np.append(X, X_set[j], axis = 0)
                Y = np.append(Y, Y_set[j])
            d_data = structure_x_y(X,Y)
            ooc_liv = One_vs_one_classifier(d_data)
            _ = ooc_liv.learn(c, gama, 'livsvm', print_op = False)
            #print('\t for train set:', end = '\n\t')
            pred_1, acc_1 = ooc_liv.predict(X, Y, print_op = False)
            # print("set acc =",acc)
            # print("\tprediction:",pred)
            pred_2, acc_2 = ooc_liv.predict(X_set[k], Y_set[k], print_op = False)
            print("\tValidation Accuracy:", acc_2, '\n')

            pred_3, acc_3 = ooc_liv.predict(X_test, Y_test, print_op = False)
            
            acc_c.append(acc_2)
            acc_t.append(acc_3)
        accuracy_c[i] = np.mean(acc_c)
        accuracy_test[i] = np.mean(acc_t)
        print("\tCross V. Accuracy (for c=",c ,") =",accuracy_c[i])
        print("\tTest Accuracy for (c=",c ,") =",accuracy_test[i])
    
    #print(accuracy_c)
    plt.plot(c_s, accuracy_c, 'g', label = 'Validation Accuracy')
    plt.plot(c_s, accuracy_test, 'r', label = 'Test Accuracy')
    plt.xticks(c_s)
    plt.xlabel('C values')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.savefig('acc_test_vs_valid.pnd')
    print(accuracy_c)
    print(accuracy_test)
    best_c_index = np.where(accuracy_c == np.amax(accuracy_c))[0]
    print("Best values of c is ", c_s[best_c_index[0]])
    return c_s[best_c_index[0]]

#%%
## Running k cross validation

# train_data_all = read_it(train_path, False)
# X_data_all, Y_data_all = preprocessing(train_data_all, False)

# validation_set(5, X_data_all[:1000], Y_data_all[:1000], X_test[:10], Y_test[:10])

    
#%%

def main(train_path, test_path, classify, part):
    
    ## Read and preprocess data
    
    train_data = read_it(train_path)
    X_data , Y_data = preprocessing(train_data) 
    
    test_data = read_it(test_path)
    X_test_data, Y_test_data = preprocessing(test_data)
    
    if classify == '0':
        if part == 'a':
            print("Linear SVM:")
            svm = SVM_class(X_data, Y_data, 1.0)
            alphas = svm.learn('Linear')
            
            print("Prediction:")
            print("Training data", end = " ")
            svm.predict(X_data[:], Y_data[:])
            print("Testing data", end = " ")
            test_predictions = svm.predict(X_test_data, Y_test_data)
            
        elif part == 'b':
            print("Gaussian SVM:")
            c = 1.0
            gama = 0.05
            svm_gaussian = SVM_class(X_data, Y_data, c, gama)
            gau_alpha = svm_gaussian.learn('Gaussian')
            
            print("Prediction:")
            print("Training data", end =" ")
            predict_gau = svm_gaussian.predict_gaussian(X_data, Y_data)
            print("Test data", end = " ")
            predict_test_gau = svm_gaussian.predict_gaussian(X_test_data, Y_test_data)
            
        elif part == 'c':
            print("\nLibsvm:")
            c =1.0
            gama = 0.05
            
            print('Linear')
            libsvm_linear = Libsvm_class(X_data, Y_data, c)
            libsvm_linear.learn('Linear')
            print('predictions on train and test')
            libsvm_linear.predict(X_data, Y_data)
            libsvm_linear.predict(X_test_data, Y_test_data)
            
            print('\nGaussian')
            libsvm_gaussian = Libsvm_class(X_data, Y_data, c, gama)
            libsvm_gaussian.learn('Gaussian')
            print('predictions on train and test ')
            libsvm_gaussian.predict(X_data, Y_data)
            libsvm_gaussian.predict(X_test_data, Y_test_data)
        else:
            print('Invalid argument')
            
    elif classify == '1':
        d_data = fancy_read(train_path)
        test_data = read_it(test_path, False)
        X_test, Y_test = preprocessing(test_data, False)
        
        if part == 'a':
            print("\n\nOne vs One Classifier for cvxopt")
            
            ooc = One_vs_one_classifier(d_data)
            models = ooc.learn(1.0, 0.05)
            
            pred, _ = ooc.predict(X_test, Y_test)

        elif part == 'b':
            print("\n\nOne vs One Classifier for livsvm")
            c = 1
            gama = 0.05
            
            ooc_liv = One_vs_one_classifier(d_data)
            models = ooc_liv.learn(c, gama, 'livsvm')
            
            pred, _ = ooc_liv.predict(X_test, Y_test)
            
        elif part == 'c':
            c = 1
            gama = 0.05
            
            ooc = One_vs_one_classifier(d_data)
            models = ooc.learn(1.0, 0.05)
            pred, _ = ooc.predict(X_test, Y_test)
            
            ooc.confusion_matrix(pred, Y_test, '2a_conf_matrix.png')
            
            ooc_liv = One_vs_one_classifier(d_data)
            models = ooc_liv.learn(c, gama, 'livsvm')
            pred, _ = ooc_liv.predict(X_test, Y_test)
            
            ooc_liv.confusion_matrix(pred, Y_test, '2b_conf_matrix.pnd')
            
        elif part == 'd':
            train_data_all = read_it(train_path, False)
            X_data_all, Y_data_all = preprocessing(train_data_all, False)
            
            validation_set(5, X_data_all, Y_data_all, X_test, Y_test)
            
        else:
            print('Invalid argument')
    else:
        print('Invalid argument')
    print("------------------- END ---------------------")
    return None
#%%
##Initializations ----
#train_path = sys.argv[1]
#test_path = sys.argv[2]
#classify = sys.argv[3]
#part = sys.argv[4]
# train_path = '~/IITD/COL774-ML/Assignment2/Q2/train.csv'
# test_path = '~/IITD/COL774-ML/Assignment2/Q2/test.csv'
# classify = '0'
# part = 'a'
d = 5

#%%
# main(train_path, test_path, classify, part)
# main(train_path, test_path, classify, 'b')
# main(train_path, test_path, classify, 'c')
# main(train_path, test_path, '1', 'a')
# main(train_path, test_path, '1', 'b')
# main(train_path, test_path, '1', 'c')
#main(train_path, test_path, '1', 'd')


#%%
if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    classify = sys.argv[3]
    part = sys.argv[4]
    main(train_path, test_path, classify, part)