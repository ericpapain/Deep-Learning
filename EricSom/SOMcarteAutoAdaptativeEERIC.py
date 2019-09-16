##########################
# CARTES AUTO-ADAPTATIVES#
##########################


# importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# chargement des données
dataset = pd.read_csv("/home/eric/Documents/analyseDonnee/udemyInoussaMerci/deeplearnig/deeplearning-master/Part 4 - Self_Organizing_Maps/EricSom/Credit_Card_Applications.csv")

# separation du jeux de données
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# transformation des données pour l'intervale 0,1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

#entrainement du Soms ou encore carte adaptative.
"""
   ici on a deux option posible,
   - la premiere option c'est de le faire à partir de zero
   - la deuxieme option est d'utiliser un code deja taper par un programmeur il s'agit de 
   minisom qui dois toujours etre dans le paquet que nous allons utiliser 
   donc le code que nous utilisons sont souvent a l'exterieur du coup dans notre 
   cas par exemple nous avons minisom qui est deja dans le meme dossier que mon fichier 
   de programmation.
   
   -x et y represente la taille de la carte auto adaptative 10*10 qui correspond a 100 neurone
   -input_len correspond au nombre de colone qui vaut le nombre de colone de notre dataset qui est 15 pour nous
   
"""

"""
nous avons eu tous les soucis du monde dans cette partie du coup je prefere copier le contenue 
du fichier Minisom dnas le code source directement
"""

# le code permet d'obetenir le resultat suivant : 
from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn


"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function=None, random_seed=None):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            decay_function, function that reduces learning_rate and sigma at each iteration
                            default function: lambda x,current_iteration,max_iter: x/(1+current_iteration/max_iter)
            random_seed, random seed to use.
        """
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self.random_generator = random.RandomState(random_seed)
        else:
            self.random_generator = random.RandomState(random_seed)
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = self.random_generator.rand(x,y,input_len)*2-1 # random initialization
        for i in range(x):
            for j in range(y):
                self.weights[i,j] = self.weights[i,j] / fast_norm(self.weights[i,j]) # normalization
        self.activation_map = zeros((x,y))
        self.neigx = arange(x)
        self.neigy = arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = subtract(x, self.weights) # x - w
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  # || x - w ||
            it.iternext()

    def activate(self, x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def gaussian(self, c, sigma):
        """ Returns a Gaussian centered in c """
        d = 2*pi*sigma*sigma
        ax = exp(-power(self.neigx-c[0], 2)/d)
        ay = exp(-power(self.neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def diff_gaussian(self, c, sigma):
        """ Mexican hat centered in c (unused) """
        xx, yy = meshgrid(self.neigx, self.neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def winner(self, x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        eta = self._decay_function(self.learning_rate, t, self.T)
        sig = self._decay_function(self.sigma, t, self.T) # sigma and learning rate decrease with the same rule
        g = self.neighborhood(win, sig)*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self, data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[self.random_generator.randint(len(data))]
            self.weights[it.multi_index] = self.weights[it.multi_index]/fast_norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_iteration):
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            rand_i = self.random_generator.randint(len(data)) # pick a random sample
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data, num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2  # keeps the learning rate nearly constant for the last half of the iterations

    def distance_map(self):
        """ Returns the distance map of the weights.
            Each cell is the normalised sum of the distances between a neuron and its neighbours.
        """
        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += fast_norm(self.weights[ii, jj, :]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        for x in data:
            error += fast_norm(x-self.weights[self.winner(x)])
        return error/len(data)

    def win_map(self, data):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

### unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal


class TestMinisom:
    def setup_method(self, method):
        self.som = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                assert_almost_equal(1.0, linalg.norm(self.som.weights[i,j]))  # checking weights normalization
        self.som.weights = zeros((5, 5))  # fake weights
        self.som.weights[2, 3] = 5.0
        self.som.weights[1, 1] = 2.0

    def test_decay_function(self):
        assert self.som._decay_function(1., 2., 3.) == 1./(1.+2./3.)

    def test_fast_norm(self):
        assert fast_norm(array([1, 3])) == sqrt(1+9)

    def test_gaussian(self):
        bell = self.som.gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_win_map(self):
        winners = self.som.win_map([5.0, 2.0])
        assert winners[(2, 3)][0] == 5.0
        assert winners[(1, 1)][0] == 2.0

    def test_activation_reponse(self):
        response = self.som.activation_response([5.0, 2.0])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_quantization_error(self):
        self.som.quantization_error([5, 2]) == 0.0
        self.som.quantization_error([4, 1]) == 0.5

    def test_quantization(self):
        q = self.som.quantization(array([4, 2]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        assert_array_almost_equal(som1.weights, som2.weights)  # same initialization
        data = random.rand(100,2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data,10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data,10)
        assert_array_almost_equal(som1.weights,som2.weights)  # same state after training

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
        for w in som.weights:
            assert_array_equal(w[0], array([1.0, .0]))
            
            

###############################################################################################
# revenons a nos mouton paour l'entrainement de notre carte auto adaptative
###############################################################################################
            #entrainement du Soms ou encore carte adaptative.
"""
   ici on a deux option posible,
   - la premiere option c'est de le faire à partir de zero
   - la deuxieme option est d'utiliser un code deja taper par un programmeur il s'agit de 
   minisom qui dois toujours etre dans le paquet que nous allons utiliser 
   donc le code que nous utilisons sont souvent a l'exterieur du coup dans notre 
   cas par exemple nous avons minisom qui est deja dans le meme dossier que mon fichier 
   de programmation.
   
   -x et y represente la taille de la carte auto adaptative 10*10 qui correspond a 100 neurones
   -input_len correspond au nombre de colone qui vaut le nombre de colone de notre dataset qui est 15 pour nous
   
"""
            
som = MiniSom(x = 10, y = 10, input_len = 15)

# initialisation des poids se fait manuellement avec une methode qui existe on va juste l'envoyer notre dataset.

som.random_weights_init(x)

#entrainement de notre carte auto adaptative cette fois on utiliseras la fonction train-random

som.train_random(x, num_iteration=100)

# Visualisation des résultats

"""
    notre carte est entrainer maintenant pour visualiser il nous faux une carte de 10*10 se que nous visualisons 
    c'est la distance MID (DISTANCE MINIMALE INTER NEURONALE) minimale interneronal. cet a dire
    pour chaque neurone de notre carte 10*10 on va calculer la distance entre se neurone d'intéret et 
    tous les neurones qui sont dans son voisinage. on va donc faire la moyenne de toutes ses distance
    et sa sa va correspondre a  notre distance MID rechercher.
    
    - si le MID est éléver, cella signifie que tous les neurones qui sont dans le voisinage du neurone
    d'interet son eloigner et du coup on conclu que notre neurone est un peu bizare et donc ne correspond pas a la norme
    et se neurone seras qualifier de neurone fraude
    
    - si le MID est faible alors cella veut dire que les neurones dans se voisinages sont plutot rapproché
     
    et nous allons choisi le neurone donc le MID est plutot élever
     a se sujet on va utiliser les couleurs pour representer tous cella le blanc pour les valeurs maximale
     et le noir pour les valeur minimale
     
     - nous allons utiliser de nouvelle fonction d'une nouvelle librairie pylab car plus customiser
"""

from pylab import bone, pcolor, colorbar, plot, show

#initialisation du graphe
bone()

# affichage des neurones de sortie sur la carte il s'agit de la distance MID et transformation des valeur en couleur
"""
pour le calcul de la distance MID on a besoin d'utiliser la methode distance_map qui permet de la calculer 
elle prend en paramètre notre jeu de donnée puisqu'elle va nous donner une matrice on va la transposer c'est a dire une rotation de 90°
"""
pcolor(som.distance_map().T)

# a present on va installer une petite légende pour se rassurer de la coerence entre les couleurs comme nous le  souhaitons
colorbar()


"""
    dans la suite nous allons marquer pour les clients accepter en vert et les client refuser en roug
    on utiliseras le parametre class que nous avons mis dans y.
    nous allons creer des markers et les afficher sur le meme graphe
    on va faire une boucle sur tous client et verifier s'il sont accepter ou non
    nous utiliserons le type énumérate.
"""

markers = ["o", "s"]
colors = ["r","g"]

for i, x in enumerate(x):
    """
        la fonction winner est une fonction qui existe deja et qui nous permet de selectionner a 
        chaque fois le neurone gagnant en fonction du MID calculer.
        donc a chaque fois la matrice w recuperer x correspond a une instance de notre dataset
        --pour mettre le marqueur au centre du carrer on ajouteras +0.5
    """
    w = som.winner(x)
    
    """
        positionnement de notre marqueur dans la figure avant affichage.
    """
    
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
show()


# detection de la fraude
"""
    on va creer un dictionnaire pour ranger les résultats obtenue
    les neurone frauduleux son ceux la qui on la fois rouge et vert donc le neurone donne des resultat bizare
"""

mappings = som.win_map(x)



# liste des clients frauduleux
"""
    en effet dans cette section, nous allons voire d'apres le graphe que nous avons obtenue des
    zone blanche normalement mais que nous verosn plusieur de valeur de decision par les neurone 
    alors on pouras conclure qu'il ya fraude et nous allons essayer d'afficher les fraudeurs de ces partie 
    lire et lire necore refaire et refaire encore cette partie
    
    c'est un systeme de coordonnée alors voila se que sa nous donne pour notre graphe on veut connaitre les clie
    les clients qui sont dans une zone de marquage donc les couleurs ne correspondent pas.
"""
#axis =0 veut dire concatenation en ligne et non en colone si on veut concatrener les tableau en colone on mettra axis=1
frauds = np.concatenate((mappings[(1,3)], 
                        mappings[(1,4)], 
                        mappings[(2,4)], 
                        mappings[(5,2)], 
                        mappings[(7,1)]),
                        axis=0)

#transformation pour avoir nos valeurs normale comme au depart
x_test = np.array(x_test)
frauds = frauds.reshape(1, -1)
frauds = np.reshape(frauds, (frauds.shape[0], frauds.shape[1],1))
frauds = sc.inverse_transform(frauds)

"""
cette liste peut deja etre envoyer a la banque pour signaler les fraudeur.
"""