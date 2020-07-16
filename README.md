# Description de chaque projet

## moteur de recherche d'image dans la base de donnée de Wang

    > Enoncé 
La recherche d’images par le contenu est un domaine de recherche très actif depuis plusieurs
années maintenant. Avec l’explosion de l’imagerie numérique, nous nous retrouvons souvent
avec d’énormes bases d’images et la recherche d’une image particulière est un problème difficile à résoudre.
Dans ce projet, Nous :
- implémentons un petit système de recherche d’images par le contenu avec des descripteurs SIFT. L’idée est de calculer des caractéristiques pour chaque image en entrée et de rechercher les images ayant les caractéristiques les plus semblables.
- La base d’images utilisée est constituée de 1000 images séparé en 100 classes et chaque classe ayant 100 éléments et tous classé par concept (Afrique, elephant, ...)
    > Technologies, outils et divers...
- Python
- K-means
- SIFT
- BoVW
- KNN
- Tensorflow
- Keras
- Theano
- sklearn

- dataset utilisé  http://wang.ist.psu.edu/docs/related/
- résultats ( sont disponible dans le rapport en pdf dans le dossier du projet)

## TP2 Indexation de contenu numérique (moteur de recherche d'images)

    > Enoncé : 
Plusieurs indices peuvent être utilisés pour indexer correctement les images. Parmi ces objets, on compte différents types d’objets (visages, animaux, véhicules, etc.). Nous allons ici
nous intéresser à la reconnaissance de panneaux routiers, basée sur de la classification supervisée.
- Créer un parser pour la base de test.
- Paramétrer et entraîner trois méthodes de classification supervisées sur la
base d’apprentissage.
- Tester les modèls sur la base de test.
- Analyser en profondeur les résultats de chacune des méthodes, et les comparer.
    > Technologies, outils et divers...
- Python
- forêts aléatoires
- SVM 
- CNN
- Tensorflow
- Keras
- Theano
- dataset utilisé  https://www.dropbox.com/s/tspzsocabq86ygi/Database.zip?dl=0
- résultats ( sont disponible dans le rapport en pdf dans le dossier du projet)

## EricSom

    > Enoncé : 
Les banques, les grandes structures financières et tous les hommes en particuliés sont victime de nos jours des fraudes de tous genres. dans se petit TP, nous utilisons les cartes auto adaptative (SOMs) pour faire de la prédiction de fraude dans une banque. a cet effet comme d'habitude, nous avons une serie d'information envoyé par la banque dans un fichier csv que nous allons utiliser pour l'entrainement et les test de notre modèle SOMs construit. Ainsi, nous allons :
- implémenté un modèl de détection de fraude utilisant un apprentissage non supervisé basé sur les Cartes auto-adaptative
- Le dataset utilisée est nommé Credit_Card_Applications.csv disponible a la racine du dossier EricSom
- le code complet est * SOMcarteAutoAdaptativeEERIC.py * .
    > Technologies et outils et divers...
- Python
- carte auto adaptative (SOMs)
- Tensorflow
- Keras
- Theano
- sklearn

## EricRnn

    > Enoncé : 
Nous implementons un petit modèl basé sur les RNN(réseau de neurone récurrent) pour essayer de prédire le cout de l'action google pour voir 
l'evolution de l'action et prédire l'allure de l'action  dans les mois prochain (marché boursier)

- implémenté un modèle de prédiction des coût des actions dans le marché boursier en utilisant les RNNs
- Le dataset utilisée est dataset disponible a la racine du dossier EricSom.
- le code complet est * EricSom * .
    > Technologies, outils et divers...
- Python
- LSTM
- Réseaux de Neurones Réccurent (RNNs)
- Tensorflow
- Keras
- Theano
- sklearn

## EricTaf
    > Enoncé: 
Nous implementons un petit modèl basé sur les ANN(réseau de neurone Artificiel) pour essayer de prédire le départ des clients d'une banque (predire pourquoi les clients quitte la banque) en exploitant les informations des clients. nous allons : 

- implémenté un modèl de prédiction de départ d'un client ou s'il va rester dans la banque en fonction des informations du client. nous utilisons les ANNs
- Le dataset utilisée est Churn_Modeling.csv disponible a la racine du dossier EricSom.
- le code complet est * annEric.py * .
    > Technologies, outils et divers...
- Python
- ANNs
- Tensorflow
- Keras
- Theano
- sklearn

## ericTafCNN

    > Enoncé : 
Nous implementons un petit modèle basé sur les ANN(réseau de neurone Artificiel) pour essayer de prédire le départ des clients d'une banque (predire pourquoi les clients quitte la banque) en exploitant les informations des clients. nous allons : 

- implémenté un modèl de classification des images en utilisants les CNNs
- Le dataset utilisée est Churn_Modeling.csv disponible a la racine du dossier EricSom
- le code complet est * cnnEric.py * .
- le code complet amélioré est * CnnAmelioration.py * .
    > Technologies, outils et divers...
- Python
- CNNs
- Tensorflow
- Keras
- Theano
- sklearn

## Template_ANN_Construction
   > Enoncé : 
nous produisons ici un template pour le traitement des données et l'organisation de la structure générale des réseaux de neurone ANNs

## application de surveillance examen

Nous implementons ici une application basé sur la reconnaissance des formes, la vision par ordinateur qui permettras de faire une surveillance en temps réel des étudiants pendant les examens, de faire un contrôle de présence avec des caméras. nous allons : 

- implémenté un modèl de classification utilisant les CNNs qui vont permettre a chaque capture d'une image, de renvoyer si la personne est reconnu par notre système ou non.
- implémenter un second model utilisant les descripteur de HAAR.xml pour la detection et la reconnaissance
- utiliser des bibliothèque disponible dans opencv pour faire la capture en temps réel des images et les renvoyer a nos différents modèles (CNNs et CascadeClassified).

- Le dataset utilisée est constitué des image de classe de chaque étudiant qui sont exploiter dans une vidéo de quelques minutes de chaque étudiant
- le code complet est * cnnEric.py * .
- le code complet amélioré est * CnnAmelioration.py * .
    > Technologies, outils et divers...
- Python
- CNNs
- Tensorflow
- Keras
- Theano
- OpenCV
- Haar
- DQ-NN
- sklearn
- Plus d'explication sur le code et les résultats ( sont disponible dans le rapport en pdf dans le dossier du projet)

## Vsualisation des réseaux de neurone 
    > Enoncé : 
Nous implementons ici une application basé sur la reconnaissance des formes, la vision par ordinateur qui permettras de faire une surveillance en temps réel des étudiants pendant les examens, de faire un contrôle de présence avec des caméras. nous allons : 

- nous utiliserons GToolkit pour proposer un tutoriel d’apprentissage et conciliation code et visualisation pour les acteurs non-experts en intelligence artificielle.
- consisterait à utiliser les réseaux de neurones comme exemple pour la plateforme GToolkit et le moteur de visualisation Roassal basé sur Pharo afin d'améliorer la compréhension par l'exploration des réseaux de neurones profond.
- utiliser des bibliothèque disponible dans opencv pour faire la capture en temps réel des images et les renvoyer a nos différents modèles (CNNs et CascadeClassified).

- Le dataset utilisée est disposible dans le repository de Pharo
- le code complet est expliqué dans le rapport 19_mezatio.pdf.
    > Technologies, outils et divers...
- Java
- Pharo
- Tensorflow
- GToolskits
- Roassal
- Plus d'explication sur le code et les résultats ( sont disponible dans le rapport en pdf dans le dossier du projet)

## Générateur d'images avec GANs

    > Enoncé : 
Nous implementons ici les réseaux de neurones antagonistes générative qui sont des réseau permettant de generer des images a partir d'une image quelconque. 
. nous allons : 

- Implementer un Discriminateur qui contient des couches de CNNs entrainé sur un ensemble de photos et ou d'image et ensuite son rôle serais de jouer a la police pour les nouvelles images générer de façon aléatoire par un générateur.
- Implémenter un générateur qui commence par une image aléatoire bruité et a chaque fois essaie de le comparer avec l'image dispo en BD et c'est au discriminateur de dire a cahque instant que l'image générer est fausse afin de permettre au modèle par le phénomène de back propagation d'améliorer les paramètre du générateur et en même temps du discriminateur ( une sorte de jeux : le générateur essai de présenter des fausses monaies au dicriminateur qui lui connais les vrai moanaies et à chaque fois le discriminateur dis si la monai est bonne ou pas et ceci entraine le générateur à allez améliorer sa monaie truqué et de revenirainsi de suite on fin par obtenir des images qui se rapproche des vrai image.
- utiliser des bibliothèque disponible dans opencv pour faire la capture en temps réel des images et les renvoyer a nos différents modèles (CNNs et CascadeClassified).

- Le dataset nous avons utilisé l'ensemble des images de nos camarades pour le projet de surveillance.

    > Technologies, outils et divers...
- Python
- CNNs
- Tensorflow
- Keras
- Theano
- sklearn

