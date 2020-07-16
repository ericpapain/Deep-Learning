#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:34:59 2020

@author: eric
"""


import cv2
import numpy as np
import os
import pickle 
from sklearn.neighbors import NearestNeighbors
from comon import features, build_histogram

                
path_imgg = ('image/456.jpg')
nb_near_img_search =10


extractor = cv2.xfeatures2d.SIFT_create(200)


def recherche(path_imgg,nb_near_img_search):
    #image de test
    data_path_test = os.path.normpath(path_imgg)
    #lecture de l'image
    img_test = cv2.imread(data_path_test)
    #passage de l'image en niveau de gris
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    #utilisation des descripteur SIFT
    keypoint, descriptor = features(img_test, extractor)
    
    #histogramme de l'image correspondante
    kmeans = pickle.load(open('serial/new_kmean','rb'))
    histogram = build_histogram(descriptor, kmeans,150)
    
    #recherche des k plus proche voisin
    neighbor = NearestNeighbors(n_neighbors = int(nb_near_img_search))
   
    preprocessed_image = pickle.load(open('serial/new_histo','rb'))
    
    #entrainement pour le knn
    neighbor.fit(preprocessed_image)
    
    X_train = [histogram]
    dist, result = neighbor.kneighbors(X_train)
    
    result = np.reshape(result,(-1,1))
    
    return result


resultat = recherche(path_imgg,nb_near_img_search)

print(resultat)
    