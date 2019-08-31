# Préparation des données 

# Préparation des packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# chargement des données
dataset = pd.read_csv("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/dataset/data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

#separation du jeu de données en train et test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Changement des éhelle normalisation et Standardisartion par l'ecart type et la moyenne
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)