#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:08:44 2019

@author: eric
"""

# importation des package d'affichage des matrice d'exploitations de notre dataset et des calcul en machine learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importation de notre dataset

dataset = pd.read_csv("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/dataset/data.csv")

# separation de notre jeu de donnée en variable prédictive et en variable prédicte
X = dataset.iloc[:, :-1].values


# separation de la variable a prédire

Y = dataset.iloc[:,-1].values


# Pour le traitement des données manquantes

from sklearn.impute import SimpleImputer


# creation de l'object imputer qui nous permettra de faire le salle boulot
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)


# affectation a notre object imputer de nos données afin qu'il puisse effectuer les traitements
# on utilise a cet effet la methode fit() qui prend en argument notre jeux de donnée
#on lui donne seulement les colone dans laquelle nous avons des données manquantes

imputer = imputer.fit(X[:, 1:3])

# apres la fin du calcul ne pas oublier de remplacer les valeurs calculer dans notre dataset

X[:, 1:3] = imputer.transform(X[:, 1:3])

# a present puisque nos algorihtme de machine learning utilise uniquement des valeurs numériques 
# pour effectuer les travaux alors il est important pour nous de transformer ici les variables
# cathégorielle que nous avons en variable continue c'est a dire des nombres pour cella on aura:
# cella se trouve dans le paquet LabelEncodder de sklearn

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


#transformation des valeurs cathégorielle des variables prédictives en valeurs continue 
# indépendente
ct = ColumnTransformer([('one_hot_encoder', 
                         OneHotEncoder(), [0])],
                         remainder='passthrough')
X =np.array(ct.fit_transform(X), dtype=np.float)

#tranformation des valeurs cathégorielle de la variable a predire en valeurs continue
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
 

# separation de notre jeux de donnée en training set et en dataTest
# ici on utilise encore les element de la bibliothèque sklearn
# tu peux toujour faire le crtl+i pour voir la documentation d'une fonction quer tu veux utiliser
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# normalisation et standardisation pour le changement d'echelle
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

# ici on calcule la moyene et l'ecart typoe qui nous permettrons de standardiser nos data
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# on ne standardise pas la variable predite car nous avons seulement deux valeurs mais dans un cas de regression 
# 
