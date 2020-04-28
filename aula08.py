# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:00:12 2019
"""
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

i=0
iris = load_iris() #carregamento da database
X = iris.data
y = iris.target
print(iris)

#Separação em sets de treino e teste (10% para teste e 90% para treino)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


#chamada do Perceptron para 3 classes 
pcp=Perceptron(tol=1e-3,random_state=0)
pcp.fit(X_train,y_train)
print('Score de Separação - 3 classes - Perceptron:',pcp.score(X_test,y_test))

#chamada do MLP para 3 classes
mlp=MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10, random_state=21,tol=0.000000001)
mlp.fit(X_train,y_train)
print('Score de Separaçao - 3 classes - MLP:',mlp.score(X_test,y_test))

#retirada da classe 2 (Virginica) do dataset
while (y[i]!=2):
    i=i+1
X=X[:i]
y=y[:i]

#Separação em sets de treino e teste (10% para teste e 90% para treino)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#chamada do Perceptron para 2 classes
pcp=Perceptron(tol=1e-3,random_state=0)
pcp.fit(X_train,y_train)
print('Score de Separação - 2 classes - Perceptron:',pcp.score(X_test,y_test))




