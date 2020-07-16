#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 01:23:09 2019

@author: eric
"""

# Artificial Neural Network

# Installing Theano 
# qui est un module de calcul numérique tres perfomant qui travaille avec numpy

# Installing Tensorflow 
# module de calcul numerique pouvant utiliser le cpu et gpu 
# permet de recoder les reseaux de neurone de notre facon

# Intalling Keras
# nous permet de creer des reseaux de neurone avec quelques modules c'est un framework

#################################################################################
# Partie 1 : Préparation de données

# importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# chargement des données
dataset = pd.read_csv("/home/eric/Documents/analyseDonnee/deeplearnig/deeplearning-master/Artificial_Neural_Networks/ericTaf/dataset/Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,-1].values

# notre jeux de donnée a des variables indépendantes catégorielle il faut donc les transformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# maintenant nous avons trois colone et il ne faut pas tous les garders sinon on risque d'avoir
# de petit probleme a cet effet, on peut la supprimer comme suis:
# alors quand on aura 0 et 0 on saura alors qu'il s'agit intuitivement de l'autre pays
X = X[:,1:]


#separation du jeu de données en train et test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# Changement de l'échelle normalisation et standardisation data scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#####################################################################################

# PARTIE 2 : CONSTRUCTION DU RESEAU DE NEURONE

# importation des modules de keras

import keras

# c'est le module qui nous permettra d'initailaiser le reseau de neurone
from keras.models import Sequential

# module qui nous permettra de creer les differentes couches du reseau de neurone
from keras.layers import Dense

# importation du module dropOut pour eviter notre reseau de tomber dans une situation de surapprntissage overffiting
from keras.layers import Dropout

# la premier etape pour creer notre reseau de neurone c'est de l'initialiser ces differente couches

# on utilisera le module sequential


# Initialisation du réseaux de neurones
classifier = Sequential()



# remarque pour les couche d'entrer la fonction redresseur est mieux pour les anns
# pour les couche de sortie la fonction sigmoide est bonne car donne une probabilité et non pas une solution directe nous permet de tirer des conclusion


# Ajout des couches dans notre réseaux de neurones ( Dense() permet de definr une nouvelle couche du reseaux a ajouter par la methode add de notre classifier)
# activation="relu" pour definir la fonction d'activation sur cette couche : "relu" veut dire fonction reddresseur
# input_dim=11 permet d'initialiser pour la premiere fois notre reseau de neurone ici 11 represente le nombre de parametre de notre entrer X(3:13) sa fait 11 parametre  et on soustrais les valeurs non utiliser

classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))

# definition sur chaque couche du taux d'apprentissage  sur chaque couche de ANNs avec notre dropout
classifier.add(Dropout(rate=0.1))


# Ajout d'une deuxième couche cachée de reseaux de neurone
classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform"))
# definition sur chaque couche du taux d'apprentissage  sur chaque couche de ANNs avec notre dropout
classifier.add(Dropout(rate=0.1))


# Ajout de la couche de sortie 
# ici c'est les paramètres qui changerons. 
# on a un seul nerone car il doit renvoyer un seul resultat binaire 
# et la fonction d'activation sigmoide car nous allons utiliser des probabilité

# unit: correspond au nombre de neurone
# puisque nous avons uniquement une seule variable a prédire alors on aura besoin dans notre cas d'un seul neurone
# soit le client doit quitter la banque soit le client va rester dans la banque c'est pourkoi on a un seul neurone
# remarque si on avais trois catégorie de sortie au lieu de deux, alors  a cet effet on devais 
# avoir trois neurone de sortie. par exemple on cherche a classifier les clients on aimerais
# savoir de  quels pays il proviennent dans se moment on aura trois variable de sortie qui vaudrais un pays pour chaque variable
# et on changeras aussi la fonction d'activation et on utilisera la fonction softmax() qui correspond toujours a une forme de fonction d'activation
# sigmoide mais qui accepte des ne correspond plus forcement a la fonction de classification de deux parametres mais de plusieurs parametre
# donc il faut retenir la fonction softmax() pour la couche de sortie

# classifier.add(Dense(units=3, activation="softmax", kernel_initializer="uniform"))
classifier.add(Dense(units=1, activation="sigmoid", 
                     kernel_initializer="uniform"))


# Compilation du réseau de neurone
# optimizer ="adam" permet de specifier que nous utilisons dans 
# se cas un algorihtme de regression stochastique comme algorihtme d'entrainement de notre couche de neurone

# loss = "binary_crossentropy" elle represente la definition de notre fonction de cout pour l"='evaluation du model
# de la meme facon si on avais plus de 2 catégorie de sortie, on devais pas utiliser la 
# fonction de cout 'binary_crossentropy' mais plutot 'categorical_crossentropy' pour l'evaluation de la fonction de cout
# metrics : permettra de messurer la performance du modèle 'accuracy'.
classifier.compile(optimizer="adam",loss = "binary_crossentropy",
                   metrics = ["accuracy"])


# Entrainement du réseaux de neurone
# on va utiliser la methode fit() pour entrainer le reseau de neurone
# batch_size : permet de specifier le lot d'observation apres qw on passe a une erreur

classifier.fit(X_train,y_train, batch_size= 10, epochs =3) 


# Prédiction sur le jeux de test c'est a dire un jeux de donnée qu'il n'as jamais vu
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)



###########################################################
#                       Travaux Pratiques - Instructions
# Utilisez notre réseau de neurones pour prédire si le client suivant va ou non quitter la banque dans les 6 mois :
# Pays : France
# Score de crédit : 600
# Genre : Masculin
# Âge : 40 ans
# Durée depuis entrée dans la banque : 3 ans
# Balance : 60000 €
# Nombre de produits : 2
# Carte de crédit ? Oui
# Membre actif ? : Oui
# Salaire estimé : 50000 €
# Devrait-on dire au revoir à ce client ?
###########################################################
# le format de donnée doit etre exactement formater comme celui du jeux de test
# sc_X.. nous permettra de le faire dans se cas .
# de la meme facon comme france vaut 0, 0 alors si nous avons par exemple 4 pays on utilise la loi 0, 0, 0 2^⁸(0000)(0001)(0010).... pour chaque nouvelle valeur on aura une representation personnele 
# jai bien compris c'est pourquoi pour la france on a mis 0, 0 popur l'observation car on a regarder dans le tableau des pays et si c'etais allemagne on aurais mis 0, 1

new_predict = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_predict= (new_predict>0.5)

# Alors quand nous allons dans le tableau de prediction on constate bien évidement que notre valeurs prédicte par le réseaux est false donc il ne quitera pas la banque





# Matrice de confusion pour évaluer les prédictions du modèle conçu et donc la perfoemance
# il permet de calculer le nombre de bonne reponse et les mauvaise reponse de prédiction 
# cette focntion de confusion permet de gerer les valeurs de 0 ou 1 du coup nous conbvertison y_pred en boolean
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



###########################################
# EVALUATION DU RESEAUX DE NEURONE
"""Dans cette section nous allons utiliser deux fonction 
de keras et de sklearn de machine learning """
"""la fonction que nous aimerions utiliser fais partir du module de sklearn de machine
learning or notre modele est dans keras nous allons alors utiliser 
uen fonction de keras qui permet de lier keras et le module de sklearn de machine
"""
# importation des modules de macchine learning et de framework keras 
# cette importation nous permettra de faire le pont en sklearn et keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# construction de notre classifier ici la seuel etape qu'on ne met pas c'est l'entrainement

def build_classifier():
    # on reconstruit le reseau de neurone sans lancer l'entrainement
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer="uniform", input_dim=11))

    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer="uniform"))

    classifier.add(Dense(units=1, activation="sigmoid", 
                         kernel_initializer="uniform"))

    classifier.compile(optimizer="adam",loss = "binary_crossentropy",
                       metrics = ["accuracy"])

    return classifier

# K-fold cros validation
# modification de notre object afin qu'il soit adaptable a notre fonction de sklearn
classifier = KerasClassifier(build_fn = build_classifier, batch_size= 10, epochs =3) 

# definition de la recupération de notre précision en utilisant la validation croiser
# cv =10 correspond au nombre de division de notre datatrain k=10 (k-cross)
precisions = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

# calcul de la moyenne de prediction et de l'ecart type
moyenne = precisions.mean()
ecart_type = precisions.std()


#################################################################
# PARTIE 4

"""Ajustement des paramètres avec le DropOut meme procedure comme avec k-fold cross"""


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# construction de notre classifier ici la seuel etape qu'on ne met pas c'est l'entrainement

def build_classifier(optimizer):
    # on reconstruit le reseau de neurone sans lancer l'entrainement
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer="uniform", input_dim=11))

    classifier.add(Dense(units=6, activation="relu", 
                         kernel_initializer="uniform"))

    classifier.add(Dense(units=1, activation="sigmoid", 
                         kernel_initializer="uniform"))

# ici on change le parametre de optimizer pour qu'il prennet  en compte a chaque fois le choix fait par notre algorithme
    classifier.compile(optimizer=optimizer,
                       loss = "binary_crossentropy",
                       metrics = ["accuracy"])   
   

    return classifier

# K-fold cros validation
# modification de notre object afin qu'il soit adaptable a notre fonction de sklearn
# , batch_size= 10, epochs =3 dans cette section on supprime sa car c'est eux on veut optimiser
# en fonction des paramètre du tableau de dictionnaire si dessous, cella permettra a 
# algorithme de faire le meilleur choix    
classifier = KerasClassifier(build_fn = build_classifier)
parameters =  {"batch_size":[25, 32],
               "epochs":[10, 20],
               "optimizer":["adam", "rmsprop"]}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_parameters = grid_search.best_score_
