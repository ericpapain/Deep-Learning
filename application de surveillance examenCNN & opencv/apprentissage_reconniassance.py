#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 03:39:55 2019

@author: eric
"""

import cv2
import os
import numpy as np
import pickle

image_dir='data/photos'

current_id=0
label_ids={}
x_train=[]
y_labels=[]

for root, dirs, files in os.walk(image_dir):
    """ on verifie s'il ya des image"""
    if len(files):
        
        """recuperation du nom du repertoire qui correspond au nom de l'acteur"""
        
        label=root.split("/")[-1]
        """parcours de tous le tableau files et recuperation des png"""
        for file in files:
            if file.endswith("png"):
                path=os.path.join(root, file)
                if not label in label_ids:
                    label_ids[label]=current_id
                    current_id+=1
                id_=label_ids[label]
                
                #redimensionnement de l'image pour permettre a notre modèle d'être plus compétent
                image=cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (150, 150))
              
                #verification de la netteter dans les image et fixage du seuil de selection des images a enregistrer dans la dataset
                #le laplacien permet de voir si une photos est floue ou non
                fm=cv2.Laplacian(image, cv2.CV_64F).var()
                if fm<89:
                    print("Photo exclue:", path, fm)
                else:
                    x_train.append(image)
                    y_labels.append(id_)
                    

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

x_train=np.array(x_train)
y_labels=np.array(y_labels)

#fonction de reconnaissance d'image
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, y_labels)
recognizer.save("trainner.yml") 