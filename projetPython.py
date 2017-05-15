# -*- coding: utf-8 -*-

import os
os.chdir('/home/karim/Documents/malek/mooc-python/projet')
import pandas as pd 
import numpy as np
import random
random.seed(9001)
tab = pd.read_csv('data_rating.csv',names=['userId', 'foodId', 'rating'])
len(tab)

useri,frequsers=np.unique(tab.userId,return_counts=True)#useri les id des users, frequsers les freq de chaque user
itemi,freqitems=np.unique(tab.foodId,return_counts=True)#itemi les id des alimentss, freqitem les freq de chaque aliment
n_users=len(useri)
n_items=len(itemi)
print("le nombre des utilisateurs est :"+ str(n_users) + " Et le nombre des aliments est: "+ str(n_items))
#le nombre des utilisateurs est :943 Et le nombre des aliments est: 1682

indice_user = pd.DataFrame()
indice_user["indice"]=range(1,len(useri)+1)
#range sert à générer une séquence de nombre range(start,stop)
indice_user["useri"]=useri


indice_item = pd.DataFrame()
indice_item["indice"]=range(1,len(itemi)+1)
indice_item["itemi"]=itemi
           
           
#créer user_ID_new et Item_ID_new

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(tab[["userId","foodId","rating"]], test_size=0.25,random_state=123)

sparsity=round(1.0-len(tab)/float(n_users*n_items),3)
print 'The sparsity level of our data base is ' +  str(sparsity*100) + '%'

 #La mise en place du modèle K-Nearst Neighbor:                                                       
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import mean_squared_error
from math import sqrt
L=[]
for i in xrange(2000,5000,500):
    k=knn(n_neighbors=i)
    k.fit(train_data[['userId', 'foodId']],train_data[['rating']])
    pred=k.predict(test_data[['userId', 'foodId']])
    L.append(sqrt(mean_squared_error(pred, test_data[['rating']])))

np.argmin(L)
                                                      
#La mise en place du modèle:                                                      
train_data_matrix = np.zeros((n_users, n_items))#matrice nulle de longuer tous les users et tous les items
for line in train_data.itertuples():#parcourire la ligne col par col
    train_data_matrix[line[1]-1, line[2]-1] = line[3] 

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    
#calcule de la cos similarity : (construction du modèle)
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
user_similarity1 = pairwise_distances(train_data_matrix, metric='cityblock')
item_similarity1 = pairwise_distances(train_data_matrix.T, metric='cityblock')

def predict(ratings, similarity, type='user'):#prend
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)#mean pour chauqe utilisateur (type = float)
#np.newaxis pour convertir mean_user_rating de array de float en array d'array pour l'utiliser avec ratings
#puis on a normalisé la var ratings (rating - E)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) #(type === array comme la var rating)
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)]) 
        
    x = np.zeros((n_users, n_items))
    for i in range(0,n_items):
        a=max(pred[:,i])
        b=min(pred[:,i])
        c=0
        d=5
        for j in range(0,n_users):
            x[j,i]=(pred[:,i][j]-(a-c))*d/(b-a+c)
    
    return x

#la prédiction avec les differents modèles:
item_prediction = predict(test_data_matrix, item_similarity, type='item')
user_prediction = predict(test_data_matrix, user_similarity, type='user')
item_prediction1 = predict(test_data_matrix, item_similarity1, type='item')
user_prediction1 = predict(test_data_matrix, user_similarity1, type='user')

#comparaison
#la creation de la fonction qui calcule le RMSE:
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth): #Root Mean Squared Error
    prediction = prediction[ground_truth.nonzero()].flatten() 
    #.flatten() fusionne les elts des array en un array
    #on attribue a prediction, les résultats des prédictions où on connait le vrais rating cad:
    #prediction: tous nos prédictions sur test; ground_truth.nonzero():les vrais résultats qu'on a dans test
    #on va mettre dans prediction les valeurs qu'on a prédit pour les elts qu'on adéja.
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()#pareil dans ground truth
    return sqrt(mean_squared_error(prediction, ground_truth))

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))
print 'User-based1 CF RMSE: ' + str(rmse(user_prediction1, test_data_matrix))
print 'Item-based1 CF RMSE: ' + str(rmse(item_prediction1, test_data_matrix))

#comparaison des prediction de modele  user avec l'oeil:
R = pd.DataFrame(test_data_matrix)
R_pred=pd.DataFrame(predict(test_data_matrix, item_similarity1, type='item'))
# Compare true ratings of item 17 with predictions
ratings = pd.DataFrame(data=R.T.loc[16,R.T.loc[16,:] > 0]).head(n=10)
ratings['Prediction'] = R_pred.T.loc[16,R.T.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
ratings



# Model-based Collaborative Filtering

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#Obtenir les composantes de SVD à partir de la matrice User-Item du train. On choisit une valeur de k=20.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)

# Multiplication des 3 matrices avec np.dot pour obtenir la matrice User_Item estimée.
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

#la normalisation de X_pred vu qu'elle retourne des données qui sont pas bien distribué dans [0,5]
import math
x = np.zeros((n_users, n_items))
for i in range(0,n_items):
    a=max(X_pred[:,i])
    b=min(X_pred[:,i])
    c=0
    d=5
    for j in range(0,n_users):
        x[j,i]=(X_pred[:,i][j]-(a-c))*d/(b-a+c)
        if math.isnan(x[j,i]): x[j,i]=0
                     
                     
# Calcul de performance avec RMSE entre la matrice estimée et la matrice du test
print 'RMSE: ' + str(rmse(x, test_data_matrix))



