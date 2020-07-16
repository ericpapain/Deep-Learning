#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:17:52 2020

@author: eric
"""

import cv2
import numpy as np
import os
import glob
import pickle 
from sklearn.cluster import KMeans
from comon import extraction_features, features

#############################################
#############################################



extractor = cv2.xfeatures2d.SIFT_create(200)

def build_histogram(descriptor_list, cluster_alg, normal_number):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        #normalisation de l'histogramme en utilisant le rapport par rapport au nombre de mots visuels par image
        histogram[i] += 1/normal_number
    return histogram

#construction histogramme générale
preprocessed_image = []

data_path = os.path.join('image','*g')
files_img = glob.glob(data_path)



descriptors,keypoints=extraction_features(files_img,200)
#sauvegarde des descripteurs
pickle.dump(descriptors,open('serial/new_descriptors','wb'))
#recherche des centroides de l'images
kmeans = KMeans(n_clusters = 150)
kmeans.fit(descriptors)
#sauvegarde des Kmean entrainé
pickle.dump(kmeans,open('serial/new_kmean','wb'))
#histogramme de tous mes images 
for image in files_img:
      image = cv2.imread(image)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      keypoint, descriptor = features(image, extractor)
      if (descriptor is not None):
          histogram = build_histogram(descriptor, kmeans, 150)
          preprocessed_image.append(histogram)
#sauvegarde des histogramme          
pickle.dump(preprocessed_image,open('serial/new_histo','wb'))







