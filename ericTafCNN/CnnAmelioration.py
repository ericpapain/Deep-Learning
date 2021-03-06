#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:59:02 2019

@author: eric
"""

#Partie 1 -

"""la premier etape ici est d'importer les librairie qui nous aiderons dans
l'importation et le traitements sur nos differentes images."""
""" initialisation ANNs"""
from keras.models import Sequential  
"""operation de convolution"""
from keras.layers import Convolution2D
"""Pooling reduction image""" 
from keras.layers import MaxPooling2D
"""flattenign pour applatir pour entrer ANN""" 
from keras.layers import Flatten 
""" pour ajouter des couche cachée et connecter"""
from keras.layers import Dense 

from keras.layers import Dropout

# initialisation du CNN de neurone a convolution comme les ANNs
classifier = Sequential()

# step 1: convolution ajout de la couche de convolution
"""
    - dans cette partie pour la creation de notre couche de convolution nous 
devons definir dans cette etape le nombre de feature detector que nous allons 
utiliser elle correspond en meme temps au nombre de features maps que nous allonsq
creer car pour chaque features detector correspond un features maps donné
    - filters= dimensioanlité espace de sortie === nombre de feature detector c'est dire de filtre
    comme remarque ici si nous avons une deuxieme couche de convolution, alors le nombre de filtre dois 
    doubler normalement c'est a dire 64 dans autre ccas ainsi de suite. cella tu peux expliquer
    
    -kernel_size= elle correspond a la taille de la matricfe de notre filters
    sa pouvais etre de la forme [3, 3] ou [3,5...]
    
    -strides= taille de deplacement de pixel 1 ou 2 quand on effectue l'operation de convolution
   
    -inpur_shape= permet de definir la taille de nos image a lire(forcer les image a adopter le meme format) et le second 
    argument 3 permet de dire que nous manipulons des images couleurs RGB
    
    -activation= pour ajouter de la non lineariter dans le modele
    permet de remplacer toutes les valeurs négative par des 0.
    -relu correspond a la fonction redresseur comme fonction d'activation
"""
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             input_shape=(200, 200, 3),
                             activation = "relu"))



# step 2: Pooling
"""
elle consiste a prendre la feauture maps que nous avons obtenue juste avant 
l'etape de convolution et on va prendre les case 2/2 on construit comme sa jusqu'aobtenir 
un plus petit resultat
    
- pool_size=permet de definir la taille de notre matrice de selection du maximun
"""
classifier.add(MaxPooling2D(pool_size=(2,2)))


# ajout de la nouvelle couche de convolution faut pas oublier son pooling
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
                             activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))



# a present pour melanger les deux couche de convolution on dois multiplier par 64 filtre a present 32*2

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
                             activation = "relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
                             activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))



# step 3: Flattening
"""
    -phase d'applatissage pour obtenir des input pour notre ANNs
    
elle se fait a la fin pour permettre de renseigner de bonne information au neurone
"""
classifier.add(Flatten())

# step 4: ANNs completement connecté
"""
    - Dense = permet d'ajouter une couche de neurone caché
        -units= nombre de neurone qui appartiennent a la couche
            dans le cas des reseaux de neurone artificielle 
            on a dis que nous pouvions prendre le nombre de variable ici nous ne poiuvons 
            pas definir normalement 
            alors dans notre cas on aura bcp de features faut prendre les nombre 
            puissance de 2 sa marche tres bien
        -activation= represente la fonction d'activation pour cette couche
        - relu est tres utiliser pour sa particularité d'etre stricte 
        soit elle laisse passer le signal ou non
"""
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(rate=0.2))


# definiton de la couche de sortie de notre reseau de neurone a convolution
"""
 - pour la couche de sortir puisque 
 nous somme tjr dans le contexte de classification 
 alors nous utilisons la fonction sigmoid sinon on aurais utiliser dans un cadre c
 catégorielle la fonction softmax et nous avons juste besoin de 1 neurpne
"""
classifier.add(Dense(units=1, activation="sigmoid"))



# etape de compilation de notre reeseau de neurone.
"""
    - optimizer= correspond a l'algorithme de macine learning a utiliser pour la classification
        adam correspond au stochastique de merde
    -loss= represente la fonction de cout binary_cross.. pour la classification et categorical_cros... pour la regression
    -metrics= "accuracy" 
    
"""
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])


#########################################################
# Entrainement de notre réseaux de neurone a convolution#
#########################################################

"""
    - faut aller lire dans la documentation de keras a keras documentation
    - augmentation d'image : permet d'eviter le surentrainement sur le jeux de donné il permet de 
    modifier le jeux de donnée de toutes les formes et de transformer les images et nous permettra d'avoir beaucoup plus 
    d'image differente
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        '/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 2 - Convolutional_Neural_Networks/ericTafCNN/dataset/training_set',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 2 - Convolutional_Neural_Networks/ericTafCNN/dataset/test_set',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary')

"""
    pour obtenir le nombre de validation_steps, on divise le nombre de donnée
    du dataset par le nombre de batch_size::::: 2000/32....
    - pour le training_set  = on divise par le nombre d'observation de notre 
    training set par le nombre de batch_size se qui donne 8000/32=250
    - pour le validation_test= ici on effectue le meme processus pour le training
    set mais on prend par contre l'echantillon de test a cette fois 2000/32 = 62.5 ===63
    -nous avons mentionner lors de la construction des ANNs cella permet d'evaluer le reseau au fur et a mesure qu'on l'entraine
    pour ne pas l'evaluer a la fin de l'apprentissage en meme temps 
    ici on fais tous a la fois comme le k-cross... evaluation et ajustement de paramètre
"""

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=50,
        validation_data=test_set,
        validation_steps=63)


#  dANS CETTE NOUVELLE PHASE NOUS ALLONS PASSER A LA PRÉDICTION D'ANIMAUX CHIEN OU CHAT
"""
    - ici il ne s'agit pas de manipuler des matrice mais plutot des image alors nous
    devons les importers dans l'endroit ou il se trouve grace a des bibliotheque de keras 
    - ensuite penser a dimenssionner notre images a la taille voulu
    - et lancer la prediction comme avec les ANNs
"""

import numpy as np
from keras.preprocessing import image

# importation de notre image en spécifiant la taille qui correspond forcement a celle de l'entrainement

test_image = image.load_img("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 2 - Convolutional_Neural_Networks/ericTafCNN/dataset/single_prediction/cat_or_dog_2.jpg",
                            target_size=(200, 200))

# ajout d'une quatrieme dimenssion a notre image a l'indice 0 pour permettre l'evaluation par notre CNN
# axis permet de spécifier l'index du groupe 
# car nous avons dans notre cas le premier groupe si nous avons plusieurs groupe on peut les positionner de la meme facon

test_image = np.expand_dims(test_image, axis=0)

# transformation de notre image en array un tableaux d'element
# test_image = image.img_to_array(test_image)

# prediction sur notre image chargé

result = classifier.predict(test_image)

# maintenant il nous faut spécifier a quoi correspond chaque prédiction 0 et 1
training_set.class_indices

# on peut maintenant mettre le resultat dans une variable et afficher

if result[0][0]==1:
    prediction = "c'est un chien"
else:
    prediction = "c'est un chat"
    
    
    
"""
POUR AMELIORER UN MODÈLE ON PEUT :
        - Changer la taille de l'image
        - ajouter plusieurs couche de convolution
        - ajouter de nouvelle couche de reseaux de neurone et pour eviter 
        de tomber dans les cas de surapprentissage alors ajouter le drop-out 
        pour  definir le taux d'apprentissage qui permet de ne pas construire un réseaux de neurone
        qui apprend trop il permet de désactiver les neurones qui apprenent trop.
        - tous ses éléments permettent d'améliorer les performance de notre modèle
        et eviter le surapprentissage lorsque le taux d'apprentissage sur de nouvelle 
        donnée est tres inférieur a celle de donnée d'entrainement.
"""