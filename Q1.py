#!/usr/bin/env python
# coding: utf-8

# In[1]:

## Run This commented lines (3) to get stopwords and punkt from nltk
# import nltk
# nltk.download('punkt')
#nltk.download('stopwords')

import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from random import seed
from random import randint

# In[6]:


def read_it(path):
    data = {}
    with open(path) as f:
        i =0
        for l in f:
            data[i] = eval(l)
            i += 1
    return pd.DataFrame.from_dict(data, orient='index')


def tokenize(review_string, stop_words= None, table = None, ps = None):
    tokens = word_tokenize(review_string)
    tokens = [token.lower() for token in tokens]
    if table != None:
        tokens = [token.translate(table) for token in tokens]
    words = [token for token in tokens if token.isalpha() and len(token)>=1]
    if stop_words != None:
        words = [word for word in words if not word in stop_words]
    if ps != None:
        words = [ps.stem(word) for word in words if len(word) >= 1]
    return words

def tokenize_bigram(review_string, stop_words= None, table = None, ps = None):
    global count_files
    count_files+=1
    if count_files%500 == 0:
        print(count_files, end=" ")
    tokens = word_tokenize(review_string)
    tokens = [token.lower() for token in tokens]
    if table != None:
        tokens = [token.translate(table) for token in tokens]
    words = [token for token in tokens if token.isalpha() and len(token)>=1]
    if stop_words != None:
        words = [word for word in words if not word in stop_words]
    if ps != None:
        words = [ps.stem(word) for word in words]
    bigram = []
    prev_word = ''
    for word in words:
        if prev_word == '':
            prev_word = word
            continue
        bigram.append(prev_word +' '+ word)
        prev_word = word
    return bigram

def calc_docs_in_class(data):
    docs_in_class = {}
    for review, label in zip(data['reviewText'],data['overall']):
        if label not in docs_in_class:
            docs_in_class[label]=0
        docs_in_class[label] += 1
    return docs_in_class

def make_term_dict(data, stop_words = None, table = None, ps = None, bigram = False):
    #reviews_by_class = {}
    term_dict_by_class = {}
    docs_in_class = {}
    vocab = set()
    t1 = datetime.datetime.now()
    
    for review, label in zip(data['reviewText'],data['overall']):
        words = []
        if bigram:
            words = tokenize_bigram(review, stop_words, table, ps)
        else:
            words = tokenize(review, stop_words, table, ps)
        
        ##Counting the docs in respective class
        if label not in docs_in_class:
            docs_in_class[label]=0
        docs_in_class[label] += 1
        
        for word in words:
            if label not in term_dict_by_class:
                term_dict_by_class[label] = {}
            if word not in term_dict_by_class[label]:
                term_dict_by_class[label][word] = 0
            
            term_dict_by_class[label][word] += 1
        
        vocab = vocab | set(words)
        
    t2 = datetime.datetime.now()
    print('Time Taken to make Term Dictionary:', t2 - t1)    
    return term_dict_by_class, docs_in_class, vocab


class NaiveBayes:
    
    def __init__(self):
        self.theta = {}
        self.prior = {}
        self.class_labels = []
        self.freq_count = {}
        self.alpha = 1
    
    def model_parameters(self, term_dict_by_class, docs_in_class, alpha, vocab):
        theta = {}
        prior = {}
        self.alpha = alpha
        total_docs = 0
        #class_labels = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.class_labels = list(set(term_dict_by_class))
        self.class_labels.sort()
        print(self.class_labels)
        for k in self.class_labels:
            #Counting total documents
            total_docs += docs_in_class[k]
            
            #Calculating frequency of total terms in a class_label
            self.freq_count[k] = 0
            for term in vocab:
                if term in term_dict_by_class[k]:
                    self.freq_count[k] += term_dict_by_class[k][term]
        
        #Calculating Thetas
        for k in self.class_labels:
            theta[k] = {}
            for term in vocab:
                if term in term_dict_by_class[k]:
                    theta[k][term] = term_dict_by_class[k][term] + alpha
                    theta[k][term] = math.log(theta[k][term]) - math.log(self.freq_count[k] + alpha*len(vocab))
                else:
                    theta[k][term] = math.log(alpha) - math.log(self.freq_count[k] + alpha*len(vocab))
        
        #Calculating Priors
        for k in self.class_labels:
            prior[k] = math.log(docs_in_class[k]/total_docs)

        return theta, prior
    
    def learn(self, term_dict_by_class,docs_in_class, alpha, vocab):
        self.theta, self.prior = self.model_parameters(term_dict_by_class,docs_in_class, alpha, vocab)
        return self.theta, self.prior
    
    def calc_accuracy(self, predictions, labels):
        correct = 0
        for p,l in zip(predictions, labels):
            if( p == l):
                correct +=1
        return correct*100/len(predictions)
    
    def predict(self, vocab, reviews, labels=[], print_f1_score = False, stop_words = None, table = None, ps = None, bigram = False):
        table = str.maketrans('', '', string.punctuation)
        overall_ratings = []*len(reviews)
        result = [{}]*len(reviews)
        final_result = []
        for review,i in zip(reviews, range(len(reviews))):
            
            words = []
            if bigram:
                words = tokenize_bigram(review, stop_words, table, ps)
            else:
                words = tokenize(review, stop_words, table, ps)
            
            for k in self.class_labels:
                result[i][k] = self.prior[k]
                for word in words:
                    #Handling new words in test case
                    if word in self.theta[k]:
                        result[i][k] += self.theta[k][word]
                    else:
                        result[i][k] += math.log(self.alpha/(self.freq_count[k] + self.alpha*len(vocab)))
            
            max_result_val = result[i][1.0]
            final_class = 1.0
            for k in self.class_labels:
                if result[i][k] > max_result_val:
                    final_class = k
                    max_result_val = result[i][k]
            
            final_result.append(final_class)
        if len(reviews) == len(labels):
            print('Accuracy: ',self.calc_accuracy(final_result, labels),'%')
            #self.confusion_matrix(final_result, labels)
        if print_f1_score:
            self.f1_score(final_result, labels)
        return final_result, result
    
    def random_prediction(self, reviews, labels=[]):
        self.class_labels = [1.0, 2.0, 3.0, 4.0, 5.0]
        seed(1)
        final_result = []
        for review in reviews:
            rand_index = randint(0,4)
            final_result.append(self.class_labels[rand_index])
        if len(final_result) == len(labels):
            print('Accuracy:', self.calc_accuracy(final_result, labels))
            #self.f1_score(final_result, labels)
        return final_result
    
    def majority_prediction(self, reviews, labels=[], docs_in_class = []):
        max_freq = 0
        max_label = 0
        for d in docs_in_class:
            if docs_in_class[d] > max_freq:
                max_freq = docs_in_class[d]
                max_label = d
        final_result = [max_label]*len(reviews)
        print('Most Occurring label:', max_label)
        if len(final_result) == len(labels):
            print('Accuracy:', self.calc_accuracy(final_result, labels))
            #self.confusion_matrix(final_result, labels)
            #self.f1_score(final_result, labels)
        return final_result
    
    def confusion_matrix(self, predictions, labels, plot_it = False, path = 'conf_matrix'):
        n = len(self.class_labels)
        conf_matrix = [[0 for i in range(n)] for j in range(n)]
        for p, l in zip(predictions, labels):
            conf_matrix[int(l)- 1][int(p)- 1] +=1
        if not plot_it:
            return conf_matrix
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
    
    def f1_score(self, prediction, labels):
        n = 5
        precision = [0]*n
        recall = [0]*n
        f1_s = [0]*n
        conf_matrix = self.confusion_matrix(prediction, labels)
        for i in range(n):
            total_row = 0
            total_col = 0
            for j in range(n):
                total_row += conf_matrix[i][j]
                total_col += conf_matrix[j][i]
            if total_col != 0:
                precision[i] = conf_matrix[i][i]/total_col
            if total_row != 0:
                recall[i] = conf_matrix[i][i]/total_row
            if precision[i] != 0 or recall[i] != 0:
                f1_s[i] = 2 * precision[i]*recall[i]/(precision[i] + recall[i])
        print('F1 Score for classes:')
        macro_f1 = 0
        for i in range(n):
            print(self.class_labels[i], ':', f1_s[i])
            macro_f1 += f1_s[i]
        macro_f1 /= n
        print("Macro F1 Score: ",macro_f1)
        return f1_s
            


#%% 

def main(train_path, test_path, part):
    
    data = read_it(train_path)
    test_data = read_it(test_path)
    docs_in_class = calc_docs_in_class(data)
    print(data.shape)
    nb = NaiveBayes()
    if part == 'a':
        alpha = 1
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data)
        print(len(vocab))
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['reviewText'], data['overall'])
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'])
        
    elif part == 'b':
        print('For Random prediction')
        _ = nb.random_prediction(data['reviewText'], data['overall'])
        print('For Majority Prediction')
        _ = nb.majority_prediction(test_data['reviewText'], test_data['overall'], docs_in_class)
        
    elif part == 'c':
        alpha = 1
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data)
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'])
        
        nb.confusion_matrix(prediction_part_a_test, test_data['overall'], True, 'NB_conf_matrix_2a.png')
        
    elif part =='d':
        alpha = 1
        ps = PorterStemmer()
        stop_words = stopwords.words('english')
        table = str.maketrans('', '', string.punctuation)
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data, stop_words, table, ps)
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['reviewText'], data['overall'], False, stop_words, table, ps)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'], False, stop_words, table, ps)

    elif part == 'e':
        print('Raw Bigrams - ')
        alpha = 1
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data, None, None, None, True)
        for t in term_dict_by_class:
            print(t, end =" ")
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['reviewText'], data['overall'], True, None, None, None, True)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'], True, None, None, None, True)
        
        print('Bigrams with stemming, stopwords removed and punctuations removed')
        alpha = 1
        ps = PorterStemmer()
        stop_words = stopwords.words('english')
        table = str.maketrans('', '', string.punctuation)
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data, stop_words, table, ps, True)
        print(len(vocab))
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['reviewText'], data['overall'], True, stop_words, table, ps, True)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'], True, stop_words, table, ps, True)
        
    elif part == 'f':
        alpha = 1
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data)
        print(len(vocab))
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['reviewText'], data['overall'], True)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['reviewText'], test_data['overall'], True)
        
        
    elif part == 'g':
        alpha = 1
        term_dict_by_class, docs_in_class, vocab = make_term_dict(data)
        print(len(vocab))
        _, _ = nb.learn(term_dict_by_class, docs_in_class, alpha, vocab)
        
        _, _ = nb.predict(vocab, data['summary'], data['overall'], True)
        prediction_part_a_test, _ = nb.predict(vocab, test_data['summary'], test_data['overall'], True)
    else:
        print('Invalid Input')
    print('------------------ END ----------------')
    
#%%
#Initializations
path = './reviews_Digital_Music_5.json/Music_Review_train.json'
test_path = './reviews_Digital_Music_5.json/Music_Review_test.json'
part = 'a'
class_size = 4

count_files = 0
#%%
main(path, test_path, 'a')
main(path, test_path, 'b')
main(path, test_path, 'c')
main(path, test_path, 'd')
main(path, test_path, 'e')
main(path, test_path, 'f')
main(path, test_path, 'g')

#%%
# if __name__ == "__main__":
    # main()

