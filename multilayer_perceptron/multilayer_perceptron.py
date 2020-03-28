#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
        return sigmoid(z)*(1-sigmoid(z))


cost = []

def threshold(z):
    z = np.array([1 if z_ >= 0.5 else 0 for z_ in z])
    return z
    

class MultilayerPerceptron:
    '''
    Parameters:
        learning_rate (float): Learning rate for our model
        
        hidden_layers (list): Number of hidden layers and neurons in them. 
            Example: [2, 2] has two hidden layers, each has 2 neurons
            
        epoch: The number of passes through the entire training dataset. 
            We use batch size as entire dataset so 1 epoch = 1 iteration
    '''
    def __init__(self, learning_rate=0.1, hidden_layers=[2], n_epochs = 100):
    
        self.learning_rate = learning_rate 
        self.layers = hidden_layers
        self.n_epochs = n_epochs
        
    def init_weight(self, n_features):
        
        np.random.seed(42)
        
        self.layers.insert(0, n_features) # For input layer
        self.layers.append(1)             # For output layer. We only cover binary classification so there's only one node
        
        self.weights = [[]] # Weights start from one so w[0] = []
        self.bias = [[]]    # Same as weights
        
        for i in range(1, len(self.layers)):
            w = np.random.randn(self.layers[i-1], self.layers[i])
            b = np.ones((1, self.layers[i]))
            self.weights.append(w)
            self.bias.append(b)
            
    def fit(self, X, y):
        
        self.init_weight(X.shape[1])
        
        for epoch in range(self.n_epochs):
            
            Z, A = self.feedforward(X)
            dw, db = self.backprop(Z, A, y)
            
            _c = -1/X.shape[0] * np.sum(y * np.log(A[-1]) + (1-y) * np.log(1 - A[-1]))
            
            if epoch % 10 == 0:
                print('Cost: ', _c)
                
            cost.append(_c)
            
            for i in range(1, len(self.layers)):
                
                self.weights[i] -= self.learning_rate * dw[i]
                self.bias[i] -= self.learning_rate * db[i]
            
            
        
    def feedforward(self, X):
        
        A = [X]  # Output after activation 
        Z = [[]] # Weighted sum
        
        for i in range(1, len(self.layers)):
            
            _x = A[-1]
            _x = np.dot(_x, self.weights[i]) + self.bias[i]
            _a = sigmoid(_x)
            
            Z.append(_x)
            A.append(_a)
    
        return (Z, A)
    
    def backprop(self, Z, A, y):
        
        y_pred = A[-1]
        
        dz = [[] for i in range(len(self.layers))]
        db = dz.copy()
        dw = dz.copy()
        
        
        for i in range(len(self.layers)-1, 0, -1):
            
            if i == len(self.layers)-1:
                dz[i] = y_pred - y
            else:
                dz[i] = np.dot(dz[i+1], self.weights[i+1].T) * sigmoid_derivative(Z[i])
                
            dw[i] = np.dot(A[i-1].T, dz[i])
            db[i] = np.sum(dz[i])
                
            
            
        return (dw, db)
    
    def predict(self, X):
        
        A = self.feedforward(X)[1]
        
        return threshold(A[-1])
        
        
            
            
        
df = pd.read_csv('heart.csv')


df['sex'] = df['sex'].astype("object") 
df['cp'] = df['cp'].astype("object") 
df['fbs'] = df['fbs'].astype("object") 
df['thal'] = df['thal'].astype("object")
df['restecg'] = df['restecg'].astype("object")
df['slope'] = df['slope'].astype("object")
df['exang'] = df['exang'].astype("object")
df['thal'] = df['thal'].astype("object")



df = pd.get_dummies(df, drop_first=True)


def normalization(col):
    res = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return res


df['trestbps'] = normalization('trestbps')
df['age']= normalization('age')
df['chol'] = normalization('chol')
df['thalach'] = normalization('thalach')
df['oldpeak'] = normalization('oldpeak')
df['ca'] = normalization('ca')


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df_X = df.drop(columns = ['target'])
df_y = df['target']


X = df_X.to_numpy()
y = df_y.to_numpy().reshape(-1, 1)


np.random.seed(42)



X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size = 0.3)



model = MultilayerPerceptron(learning_rate=0.01, hidden_layers=[16,8], n_epochs = 500)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)


print('Accuracy test: ', accuracy_score(y_pred, y_test.flatten())*100, '%')





