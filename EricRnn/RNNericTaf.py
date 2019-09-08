# RESEAU DE NEURONE RECURRENT
"""
nous allons essayer de prédire le cout de l'action google pour voir 
l'evolution de l'action et prédire l'allure de l'action 
dans les mois prochain

le marcher booursier
"""



# PARTIE 1 - preparation des données

# importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# chargement jeux d'entrainement
dataset_train = pd.read_csv("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 3 - Recurrent_Neural_Networks/EricRnn/dataset/Google_Stock_Price_Train.csv")

training_set = dataset_train[["Open"]].values


# Changement de l'échelle normalisation data scaling pour mettre les donnée dans le meme echelle
"""
    pour les RNNs pour l'etape de changement d'echelle c'est la normalisation qui est priser avec
    le minmaxscaling un peu comme 
    
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

feature_range= permet de definir la plage de normalisation entre (0 et 1).
"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)



# creation d'une structure avec 60 timesstepset 1 sortie

"""
60 timestep : veut dire que chaque entrer ou bien pour predire chaque jours de bourse il dois 
regarder les valeurs de l'action dans les 60 dernier jours et en fonction de ces 60 
jours précédents il dois prédire la valeur du jours suivants.

le modèle regarde le comportement dans les 60 dernier jours de bourses
celle correspond au 3 derniers mois de bourses

X_train=correspond au données d'entrainement c'est a dire au 60 jours de bourses
y_train=correspondras au sortie de notre reseaux c'est a dire a la prédiction du reseau.

"""

# principe prendre les 60 dernier jours dans le passé pour essayer de prédire le jour suivant.
 
x_train =[]
y_train =[]

for i in range(60,1258):
    x_train.append(training_set_scaled[(i-60):i])
    y_train.append(training_set_scaled[i,0])

#se rappeler que numpy travaille juste avec des array duc oup convertir notre liste en array
    
x_train = np.array(x_train)
y_train = np.array(y_train)


# Reshaping ou ajout d'une nouvelle dimenssion dans nos donnée
"""
chaque ligne dans notre jeux de donnée correspond a un jour sur les ligne et sur les colone on a le temps
ppour avoir une matrice de profondeur de trois dimension c'est pourquoi on utilise la fonction reshape
lire la documentation de reshape dans la documentation de keras
"""
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))


"""
alors plus tard si on veut ameliorer nos prédiction on peut ajouter de nouvelle donnée
de nouvelle variable pour améliorer notre prédiction
"""


##################################
# PARTIE 2 - Construction du RNN #
##################################

# importation des modules de keras

import keras

# c'est le module qui nous permettra d'initailaiser le reseau de neurone
from keras.models import Sequential

# module qui nous permettra de creer les differentes couches du reseau de neurone
from keras.layers import Dense

#importation de la librarie pour la gestion des LSTM

from keras.layers import LSTM

# importation du module dropOut pour eviter notre reseau de tomber dans une situation de surapprntissage overffiting
from keras.layers import Dropout


#initialisation du reseau de neurone
"""
cette fois on fait de la regression car il 
s'agit de predire la valeur de quelque chose une valeurs continue
"""
regressor = Sequential()


#ajout de nouvelle couche caché de LSTM+Dropout
"""
    - unit = correspond au nombre de neurone pour cette couche en particulié
    - return_sequences = on a besoin de le spécifier pour empiler plusieurs couche de LSTM ensemble
cella est un peut comme l'accumulation des couche de convolution .
on met a true pour dire que c'est bon
    - input_shape = un peu pour definir la taille de se qu'on va lui donner en entreer
comme dans les convolutions avec la definition de la taille des images.
     - Dropout : permet de diminuer les risque de surentrainement.elle permet de desactiver les reseaux de maniere un peu aléatoire afin que les neurone ne devie
     enne pas trop fort entre eux et cella permet de corriger le surentrainement
"""
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

regressor.add(Dropout(0.2))


#ajout de la 2 ème couche caché de LSTM+Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#ajout de la 3 ème couche caché de LSTM+Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#ajout de la 3.1 ème couche caché de LSTM+Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#ajout de la 3.2 ème couche caché de LSTM+Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

#ajout de la 4 ème couche caché de LSTM+Dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

#ajout de la couche de sortie
"""
ici on a un seul neurone en sortie car nous allons predire plutot la valeur de sortie d'une valeur qui est le jour suivant
"""
regressor.add(Dense(units=1))


# compilation du reseau de neurone : place fction de regression, activation, algo de machine learning
"""
pardon il faut lire la doc de keras
mean_squared_error = ici est mieux car nous ne sommes plus dans un cadre de classification 
du coup on n'utiliseras plus le binary_error .....
    -batch_size permet de specifier sur combien d'observation on va faire la mise a jour des poids
    cella permet de spécifier apres combien d'observatroin on feras la mise a jours direct des poids du reseaux
"""
regressor.compile(optimizer="adam", loss="mean_squared_error")


# entrainement du réseau
regressor.fit(x_train, y_train, epochs=10, batch_size=32)


##########################################
# PARTIE 3 - Prédiction et visualisation #
##########################################
"""
ici maintenant n'oublions pas que nous avons dis pour predire le jours suivant
on devrais s'inspirer des 60 dernier jours 
alors dans se cas nous aurons besoin de jumeller alors le jeux de donnée d'entrainemtn
avec le jeux de donée de test n'oublions pas aussi les echelles
- solution on va concatener tous les deux jeux de test
-on va changer l'echelle seulement pour le jeux de donnée d'entree
"""

#Données de 2017
dataset_test = pd.read_csv("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 3 - Recurrent_Neural_Networks/EricRnn/dataset/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values


#Prédiction pour 2017
"""
    - axis=0 : pour spécifier que la concatenation se feras sur les ligne .
    si on voulais une concaténation sur les colone on mettrais axis=1 en python c'est sa 
     les lignes commence a l'indice 0 pour les ligne et 1 pour les colone
"""

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)

"""
    a present on construit la donnée qu'il faut nourir notre reseau regarde les valeur s dans les variable
    on utilise plus fitTransform car si on le fais le reseau perd la facon donc il a transformer les donner
    du coup il faut utiliser l'objet transfort pour avoir la meme transformation sur le jeu de test
"""
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#on recupere les 20 jours a prédire.
x_test =[]

for i in range(60,80):
    x_test.append(training_set_scaled[(i-60):i])    
x_test = np.array(x_test)

#ajout de la troisieme dimenssion
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

# place a la prédiction
predicted_stock_price = regressor.predict(x_test)

"""
puisqu'on a fait des transformation pour avoir des valeur entres 0 et 1
alors nous voulons afficher les valeurs predicte alors nous allons  faire la transformation inverse
    - cela se fait avec la fonction "inverse_transform"
"""

predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualisation des résultats
plt.plot(real_stock_price, color="red", 
         label="prix réel de l'action Google")

plt.plot(real_stock_price, color="green", 
         label="prix prédit de l'action Google")

plt.title("Prédictions de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()