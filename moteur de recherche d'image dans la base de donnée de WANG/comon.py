#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 05:30:06 2020

@author: eric
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:45:54 2020

@author: eric
"""


import cv2
import numpy as np
import os
import glob
import pickle 
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


extractor = cv2.xfeatures2d.SIFT_create(200)



def extraction_features(cheminBanqueImage,nbre_feature):
    ##
    tab_descripteur=[]
    
    data_path = os.path.join(cheminBanqueImage,'*g')
    files = glob.glob(data_path)    
    dict_descripteur = {}
    sift = cv2.xfeatures2d.SIFT_create(nbre_feature)
    features=[]
    sift_vectors={}
    
    
    for filename in files:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = sift.detectAndCompute(gray,None)
     
        #ajout de tous mes element dans le dictionnaire labeliser
        dict_descripteur[filename]=descriptor   
        
        #liste  classer par ordre d'arriver
        tab_descripteur.extend(descriptor)
     
        #ajout a la suite pile
        features.append(descriptor)
        
        #creation des vecteur de sift qu'on recupera comme dans le tutos
        sift_vectors[filename] = features
        
        #recupération des element comme dans le tutoriel
        all_bovw_feature=sift_vectors
    
    return tab_descripteur,all_bovw_feature


def serialisation_feat(all_bovw_feature, tab_descripteur):
    #serialisation des données
    pickle.dump(all_bovw_feature,open('serial/serialisation_dictionnaire_descipteur','wb'))
    pickle.dump(tab_descripteur,open('serial/serialisation_tab_descripteur','wb'))
    #pickle.dump(k_model,open('serial/k_mean_modele','wb'))
    #pickle.dump(visual_words,open('serial/centroide','wb'))
    #pickle.dump(bovw_train,open('serial/centroide','wb'))

    #desérialisation
    #var = pickle.load(open('serialisation_dictionnaire_descipteu','rb'))
    
def kmeans(k, tab_descripteur):
    k_model = KMeans(n_clusters = k, n_init=30)
    k_model = k_model.fit(tab_descripteur)
    visual_words = k_model.cluster_centers_ 
    
    return visual_words,k_model 



def histogramme_images(all_bovw_feature,nbreCluster,k_model):
    #histogramme de l'image representer
    dict_feature={}
    label_train =k_model.labels_
    i=0
    print(label_train)
    for key,tab_descripteur in all_bovw_feature.items():
        histo_list = []
        print(key)
       # for img in tab_descripteur:
        histo = np.zeros(nbreCluster)
            #idx = k_model.predict([feat])
            # 15 ==correspond au nobre de mots visuels par image
        histo[label_train[i]] += 1/100 
        i+=1
        histo_list.append(histo)
        dict_feature[key]=histo_list
    return dict_feature
   
    

def traitement_img_enter(path_imgg,cluster_size,k_model):
   
        ##
    tab_descripteur=[]
    
    data_path = os.path.join(path_imgg,'*g')
    files = glob.glob(data_path)
    i=0
    
    dict_descripteur = {}
    sift = cv2.xfeatures2d.SIFT_create(100)
    features=[]
    sift_vectors={}
    
    
    for filename in files:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = sift.detectAndCompute(gray,None)
     
        #ajout de tous mes element dans le dictionnaire labeliser
        dict_descripteur[filename]=descriptor   
        
        #liste  classer par ordre d'arriver
        tab_descripteur.extend(descriptor)
     
        #ajout a la suite pile
        features.append(descriptor)
        
        #creation des vecteur de sift qu'on recupera comme dans le tutos
        sift_vectors[filename] = features
        
        #recupération des element comme dans le tutoriel
        all_bovw_feature=sift_vectors
        descriptor_list =tab_descripteur
       
        i=i+1
        if(i==1):
            break
    
    #histogramme de l'image representer
    dict_feature={}
    label_train =k_model.labels_
    i=0
    print(label_train)
    for key,tab_descripteur in all_bovw_feature.items():
        histo_list = []
        print(key)
       # for img in tab_descripteur:
        histo = np.zeros(cluster_size)
            #idx = k_model.predict([feat])
            # 100 ==correspond au nobre de mots visuels par image
        histo[label_train[i]] += 1/100 
        i+=1
        histo_list.append(histo)
        dict_feature[key]=histo_list
  
    return dict_feature



def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

#calcul de la precision de recherche pour chaque image
def analyse_result(dic_neigbor):
    dic_faux_positif={}
    dico_moyenne={}
    dic_precision_concept={}
    
    bad_afr_plage=0
    bad_afr_monument=0
    bad_afr_bus=0
    bad_afr_dinosaure=0
    bad_afr_elepht=0
    bad_afr_fleur=0
    bad_afr_cheval=0
    bad_afr_montagne=0
    bad_afr_nouriture=0
    
    bad_pla_afrique=0
    bad_pla_monument=0
    bad_pla_bus=0
    bad_pla_dinosaure=0
    bad_pla_elepht=0
    bad_pla_fleur=0
    bad_pla_cheval=0
    bad_pla_montagne=0
    bad_pla_nouriture=0
    
    bad_monu_afrique=0
    bad_monu_plage=0
    bad_monu_bus=0
    bad_monu_dinosaure=0
    bad_monu_elepht=0
    bad_monu_fleur=0
    bad_monu_cheval=0
    bad_monu_montagne=0
    bad_monu_nouriture=0
    
    
    bad_bus_afrique=0
    bad_bus_plage=0
    bad_bus_monument=0
    bad_bus_dinosaure=0
    bad_bus_elepht=0
    bad_bus_fleur=0
    bad_bus_cheval=0
    bad_bus_montagne=0
    bad_bus_nouriture=0
    
    bad_din_afrique=0
    bad_din_plage=0
    bad_din_monument=0
    bad_din_bus=0
    bad_din_elepht=0
    bad_din_fleur=0
    bad_din_cheval=0
    bad_din_montagne=0
    bad_din_nouriture=0
    
    bad_ele_afrique=0
    bad_ele_plage=0
    bad_ele_monument=0
    bad_ele_bus=0
    bad_ele_dinosaure=0
    bad_ele_fleur=0
    bad_ele_cheval=0
    bad_ele_montagne=0
    bad_ele_nouriture=0
    
    bad_fle_afrique=0
    bad_fle_plage=0
    bad_fle_monument=0
    bad_fle_bus=0
    bad_fle_dinosaure=0
    bad_fle_elepht=0
    bad_fle_cheval=0
    bad_fle_montagne=0
    bad_fle_nouriture=0
    
    bad_che_afrique=0
    bad_che_plage=0
    bad_che_monument=0
    bad_che_bus=0
    bad_che_dinosaure=0
    bad_che_elepht=0
    bad_che_fleur=0
    bad_che_montagne=0
    bad_che_nouriture=0
    
    bad_mon_afrique=0
    bad_mon_plage=0
    bad_mon_monument=0
    bad_mon_bus=0
    bad_mon_dinosaure=0
    bad_mon_elepht=0
    bad_mon_fleur=0
    bad_mon_cheval=0
    bad_mon_nouriture=0
    
    bad_nou_afrique=0
    bad_nou_plage=0
    bad_nou_monument=0
    bad_nou_bus=0
    bad_nou_dinosaure=0
    bad_nou_elepht=0
    bad_nou_fleur=0
    bad_nou_cheval=0
    bad_nou_montagne=0
        
    for key, val in dic_neigbor.items():
        bon=0
        #print(key)
        for item in val:
            item=int(item)
            #print(item)
            if((key<=99)and(item<=99)):
                #print('Afrique')
                bon=bon+1
            elif((key<=99)and(item>99)and(item<=199)):
                bad_afr_plage=bad_afr_plage+1
            elif((key<=99)and(item>199)and(item<=299)):
                bad_afr_monument=bad_afr_monument+1
            elif((key<=99)and(item>299)and(item<=399)):
                bad_afr_bus=bad_afr_bus+1
            elif((key<=99)and(item>399)and(item<=499)):
                bad_afr_dinosaure=bad_afr_dinosaure+1
            elif((key<=99)and(item>499)and(item<=599)):
                bad_afr_elepht=bad_afr_elepht+1
            elif((key<=99)and(item>599)and(item<=699)):
                bad_afr_fleur=bad_afr_fleur+1
            elif((key<=99)and(item>699)and(item<=799)):
                bad_afr_cheval=bad_afr_cheval+1
            elif((key<=99)and(item>799)and(item<=899)):
                bad_afr_montagne=bad_afr_montagne+1
            elif((key<=99)and(item>899)and(item<=999)):
                bad_afr_nouriture=bad_afr_nouriture+1
      #######          
            elif((key>99)and(key<=199)and(item>99)and(item<=199)):
                #print('plage')
                bon=bon+1
            elif((key>99)and(key<=199)and(item<=99)):
                bad_pla_afrique += 1
            elif((key>99)and(key<=199)and(item>199)and(item<=299)):
                bad_pla_monument += 1
            elif((key>99)and(key<=199)and(item>299)and(item<=399)):
                bad_pla_bus += 1
            elif((key>99)and(key<=199)and(item>399)and(item<=499)):
                bad_pla_dinosaure += 1
            elif((key>99)and(key<=199)and(item>499)and(item<=599)):
                bad_pla_elepht+= 1
            elif((key>99)and(key<=199)and(item>599)and(item<=699)):
                bad_pla_fleur += 1
            elif((key>99)and(key<=199)and(item>699)and(item<=799)):
                bad_pla_cheval += 1
            elif((key>99)and(key<=199)and(item>799)and(item<=899)):
                bad_pla_montagne += 1
            elif((key>99)and(key<=199)and(item>899)and(item<=999)):
                bad_pla_nouriture += 1
    
      #######            
            elif((key>199)and(key<=299)and(item>199)and(item<=299)):
                #print('monument')
                bon=bon+1
            elif((key>199)and(key<=299)and(item<=99)):
                bad_monu_afrique += 1
            elif((key>199)and(key<=299)and(item>99)and(item<=199)):
                bad_monu_plage += 1
            elif((key>199)and(key<=299)and(item>299)and(item<=399)):
                bad_monu_bus += 1
            elif((key>199)and(key<=299)and(item>399)and(item<=499)):
                bad_monu_dinosaure += 1
            elif((key>199)and(key<=299)and(item>499)and(item<=599)):
                bad_monu_elepht += 1
            elif((key>199)and(key<=299)and(item>599)and(item<=699)):
                bad_monu_fleur += 1
            elif((key>199)and(key<=299)and(item>699)and(item<=799)):
                bad_monu_cheval += 1
            elif((key>199)and(key<=299)and(item>799)and(item<=899)):
                bad_monu_montagne += 1
            elif((key>199)and(key<=299)and(item>899)and(item<=999)):
                bad_monu_nouriture += 1
                
        #######          
            elif((key>299)and(key<=399)and(item>299)and(item<=399)):
                #print('bus')
                bon=bon+1
            elif((key>299)and(key<=399)and(item<=99)):
                bad_bus_afrique += 1
            elif((key>299)and(key<=399)and(item>99)and(item<=199)):
                bad_bus_plage += 1
            elif((key>299)and(key<=399)and(item>199)and(item<=299)):
                bad_bus_monument += 1
            elif((key>299)and(key<=399)and(item>399)and(item<=499)):
                bad_bus_dinosaure += 1
            elif((key>299)and(key<=399)and(item>499)and(item<=599)):
                bad_bus_elepht += 1
            elif((key>299)and(key<=399)and(item>599)and(item<=699)):
                bad_bus_fleur += 1
            elif((key>299)and(key<=399)and(item>699)and(item<=799)):
                bad_bus_cheval += 1
            elif((key>299)and(key<=399)and(item>799)and(item<=899)):
                bad_bus_montagne += 1
            elif((key>299)and(key<=399)and(item>899)and(item<=999)):
                bad_bus_nouriture += 1
                
       #######           
            elif((key>399)and(key<=499)and(item>399)and(item<=499)):
                #print('dinosaure')
                bon=bon+1
            elif((key>399)and(key<=499)and(item>99)and(item<=199)):
                bad_din_plage += 1
            elif((key>399)and(key<=499)and(item>199)and(item<=299)):
                bad_din_monument += 1
            elif((key>399)and(key<=499)and(item>299)and(item<=399)):
                bad_din_bus += 1
            elif((key>399)and(key<=499)and(item<=99)):
                bad_din_afrique += 1
            elif((key>399)and(key<=499)and(item>499)and(item<=599)):
                bad_din_elepht += 1
            elif((key>399)and(key<=499)and(item>599)and(item<=699)):
                bad_din_fleur += 1
            elif((key>399)and(key<=499)and(item>699)and(item<=799)):
                bad_din_cheval += 1
            elif((key>399)and(key<=499)and(item>799)and(item<=899)):
                bad_din_montagne += 1
            elif((key>399)and(key<=499)and(item>899)and(item<=999)):
                bad_din_nouriture += 1
                
       #######           
            elif((key>499)and(key<=599)and(item>499)and(item<=599)):
                #print('éléphant')
                bon=bon+1
            elif((key>499)and(key<=599)and(item>99)and(item<=199)):
                bad_ele_plage += 1
            elif((key>499)and(key<=599)and(item>199)and(item<=299)):
                bad_ele_monument += 1
            elif((key>499)and(key<=599)and(item>299)and(item<=399)):
                bad_ele_bus += 1
            elif((key>499)and(key<=599)and(item>399)and(item<=499)):
                bad_ele_dinosaure += 1
            elif((key>499)and(key<=599)and(item<=99)):
                bad_ele_afrique += 1
            elif((key>499)and(key<=599)and(item>599)and(item<=699)):
                bad_ele_fleur += 1
            elif((key>499)and(key<=599)and(item>699)and(item<=799)):
                bad_ele_cheval += 1
            elif((key>499)and(key<=599)and(item>799)and(item<=899)):
                bad_ele_montagne += 1
            elif((key>499)and(key<=599)and(item>899)and(item<=999)):
                bad_ele_nouriture += 1
                
       #######           
            elif((key>599)and(key<=699)and(item>599)and(item<=699)):
                #print('fleur')
                bon=bon+1
            elif((key>599)and(key<=699)and(item>99)and(item<=199)):
                bad_fle_plage += 1
            elif((key>599)and(key<=699)and(item>199)and(item<=299)):
                bad_fle_monument += 1
            elif((key>599)and(key<=699)and(item>299)and(item<=399)):
                bad_fle_bus += 1
            elif((key>599)and(key<=699)and(item>399)and(item<=499)):
                bad_fle_dinosaure += 1
            elif((key>599)and(key<=699)and(item>499)and(item<=599)):
                bad_fle_elepht += 1
            elif((key>599)and(key<=699)and(item<=99)):
                bad_fle_afrique += 1
            elif((key>599)and(key<=699)and(item>699)and(item<=799)):
                bad_fle_cheval += 1
            elif((key>599)and(key<=699)and(item>799)and(item<=899)):
                bad_fle_montagne += 1
            elif((key>599)and(key<=699)and(item>899)and(item<=999)):
                bad_fle_nouriture += 1
                
      #######            
            elif((key>699)and(key<=799)and(item>699)and(item<=799)):
                #print('cheval')
                bon=bon+1
            elif((key>699)and(key<=799)and(item>99)and(item<=199)):
                bad_che_plage += 1
            elif((key>699)and(key<=799)and(item>199)and(item<=299)):
                bad_che_monument += 1
            elif((key>699)and(key<=799)and(item>299)and(item<=399)):
                bad_che_bus += 1
            elif((key>699)and(key<=799)and(item>399)and(item<=499)):
                bad_che_dinosaure += 1
            elif((key>699)and(key<=799)and(item>499)and(item<=599)):
                bad_che_elepht += 1
            elif((key>699)and(key<=799)and(item>599)and(item<=699)):
                bad_che_fleur += 1
            elif((key>699)and(key<=799)and(item<=99)):
                bad_che_afrique += 1
            elif((key>699)and(key<=799)and(item>799)and(item<=899)):
                bad_che_montagne += 1
            elif((key>699)and(key<=799)and(item>899)and(item<=999)):
                bad_che_nouriture += 1
                
       #######           
            elif((key>799)and(key<=899)and(item>799)and(item<=899)):
                #print('montagne')
                bon=bon+1
            elif((key>799)and(key<=899)and(item>99)and(item<=199)):
                bad_mon_plage += 1
            elif((key>799)and(key<=899)and(item>199)and(item<=299)):
                bad_mon_monument += 1
            elif((key>799)and(key<=899)and(item>299)and(item<=399)):
                bad_mon_bus += 1
            elif((key>799)and(key<=899)and(item>399)and(item<=499)):
                bad_mon_dinosaure += 1
            elif((key>799)and(key<=899)and(item>499)and(item<=599)):
                bad_mon_elepht += 1
            elif((key>799)and(key<=899)and(item>599)and(item<=699)):
                bad_mon_fleur += 1
            elif((key>799)and(key<=899)and(item>699)and(item<=799)):
                bad_mon_cheval += 1
            elif((key>799)and(key<=899)and(item<=99)):
                bad_mon_afrique += 1
            elif((key>799)and(key<=899)and(item>799)and(item<=999)):
                bad_mon_nouriture += 1
                
         #######         
            elif((key>899)and(key<=999)and(item>899)and(item<=999)):
                #print('nourriture')
                bon=bon+1
            elif((key>899)and(key<=999)and(item>99)and(item<=199)):
                bad_nou_plage += 1
            elif((key>899)and(key<=999)and(item>199)and(item<=299)):
                bad_nou_monument += 1
            elif((key>899)and(key<=999)and(item>299)and(item<=399)):
                bad_nou_bus += 1
            elif((key>899)and(key<=999)and(item>399)and(item<=499)):
                bad_nou_dinosaure += 1
            elif((key>899)and(key<=999)and(item>499)and(item<=599)):
                bad_nou_elepht += 1
            elif((key>899)and(key<=999)and(item>599)and(item<=699)):
                bad_nou_fleur += 1
            elif((key>899)and(key<=999)and(item>699)and(item<=799)):
                bad_nou_cheval += 1
            elif((key>899)and(key<=999)and(item>799)and(item<=899)):
                bad_nou_montagne += 1
            elif((key>899)and(key<=999)and(item<=99)):
                bad_nou_afrique += 1
            
            #ajout au dico des precision de chaque image
            precision=bon
            dic_precision_concept[('image:',key)]=precision
        
            #dico des dico
            dico_moyenne[key]=(precision*100)*10
            
    dic_faux_positif['bad_afr_plage']=bad_afr_plage
    dic_faux_positif['bad_afr_monument']=bad_afr_monument
    dic_faux_positif['bad_afr_bus']=bad_afr_bus
    dic_faux_positif['bad_afr_dinosaure']=bad_afr_dinosaure
    dic_faux_positif['bad_afr_elepht']=bad_afr_elepht
    dic_faux_positif['bad_afr_fleur']=bad_afr_fleur
    dic_faux_positif['bad_afr_cheval']=bad_afr_cheval
    dic_faux_positif['bad_afr_montagne']=bad_afr_montagne
    dic_faux_positif['bad_afr_nouriture']=bad_afr_nouriture
    
    dic_faux_positif['bad_pla_afrique']=bad_pla_afrique
    dic_faux_positif['bad_pla_monument']=bad_pla_monument
    dic_faux_positif['bad_pla_bus']=bad_pla_bus
    dic_faux_positif['bad_pla_dinosaure']=bad_pla_dinosaure
    dic_faux_positif['bad_pla_elepht']=bad_pla_elepht
    dic_faux_positif['bad_pla_fleur']=bad_pla_fleur
    dic_faux_positif['bad_pla_cheval']=bad_pla_cheval
    dic_faux_positif['bad_pla_montagne']=bad_pla_montagne
    dic_faux_positif['bad_pla_nouriture']=bad_pla_nouriture
    
    dic_faux_positif['bad_monu_afrique']=bad_monu_afrique
    dic_faux_positif['bad_monu_plage']=bad_monu_plage
    dic_faux_positif['bad_monu_bus']=bad_monu_bus
    dic_faux_positif['bad_monu_dinosaure']=bad_monu_dinosaure
    dic_faux_positif['bad_monu_elepht']=bad_monu_elepht
    dic_faux_positif['bad_monu_fleur']=bad_monu_fleur
    dic_faux_positif['bad_monu_cheval']=bad_monu_cheval
    dic_faux_positif['bad_monu_montagne']=bad_monu_montagne
    dic_faux_positif['bad_monu_nouriture']=bad_monu_nouriture
    
    
    dic_faux_positif['bad_bus_afrique']=bad_bus_afrique
    dic_faux_positif['bad_bus_plage']=bad_bus_plage
    dic_faux_positif['bad_bus_monument']=bad_bus_monument
    dic_faux_positif['bad_bus_dinosaure']=bad_bus_dinosaure
    dic_faux_positif['bad_bus_elepht']=bad_bus_elepht
    dic_faux_positif['bad_bus_fleur']=bad_bus_fleur
    dic_faux_positif['bad_bus_cheval']=bad_bus_cheval
    dic_faux_positif['bad_bus_montagne']=bad_bus_montagne
    dic_faux_positif['bad_bus_nouriture']=bad_bus_nouriture
    
    dic_faux_positif['bad_din_afrique']=bad_din_afrique
    dic_faux_positif['bad_din_plage']=bad_din_plage
    dic_faux_positif['bad_din_monument']=bad_din_monument
    dic_faux_positif['bad_din_bus']=bad_din_bus
    dic_faux_positif['bad_din_elepht']=bad_din_elepht
    dic_faux_positif['bad_din_fleur']=bad_din_fleur
    dic_faux_positif['bad_din_cheval']=bad_din_cheval
    dic_faux_positif['bad_din_montagne']=bad_din_montagne
    dic_faux_positif['bad_din_nouriture']=bad_din_nouriture
    
    dic_faux_positif['bad_ele_afrique']=bad_ele_afrique
    dic_faux_positif['bad_ele_plage']=bad_ele_plage
    dic_faux_positif['bad_ele_monument']=bad_ele_monument
    dic_faux_positif['bad_ele_bus']=bad_ele_bus
    dic_faux_positif['bad_ele_dinosaure']=bad_ele_dinosaure
    dic_faux_positif['bad_ele_fleur']=bad_ele_fleur
    dic_faux_positif['bad_ele_cheval']=bad_ele_cheval
    dic_faux_positif['bad_ele_montagne']=bad_ele_montagne
    dic_faux_positif['bad_ele_nouriture']=bad_ele_nouriture
    
    dic_faux_positif['bad_fle_afrique']=bad_fle_afrique
    dic_faux_positif['bad_fle_plage']=bad_fle_plage
    dic_faux_positif['bad_fle_monument']=bad_fle_monument
    dic_faux_positif['bad_fle_bus']=bad_fle_bus
    dic_faux_positif['bad_fle_dinosaure']=bad_fle_dinosaure
    dic_faux_positif['bad_fle_elepht']=bad_fle_elepht
    dic_faux_positif['bad_fle_cheval']=bad_fle_cheval
    dic_faux_positif['bad_fle_montagne']=bad_fle_montagne
    dic_faux_positif['bad_fle_nouriture']=bad_fle_nouriture
    
    dic_faux_positif['bad_che_afrique']=bad_che_afrique
    dic_faux_positif['bad_che_plage']=bad_che_plage
    dic_faux_positif['bad_che_monument']=bad_che_monument
    dic_faux_positif['bad_che_bus']=bad_che_bus
    dic_faux_positif['bad_che_dinosaure']=bad_che_dinosaure
    dic_faux_positif['bad_che_elepht']=bad_che_elepht
    dic_faux_positif['bad_che_fleur']=bad_che_fleur
    dic_faux_positif['bad_che_montagne']=bad_che_montagne
    dic_faux_positif['bad_che_nouriture']=bad_che_nouriture
    
    dic_faux_positif['bad_mon_afrique']=bad_mon_afrique
    dic_faux_positif['bad_mon_plage']=bad_mon_plage
    dic_faux_positif['bad_mon_monument']=bad_mon_monument
    dic_faux_positif['bad_mon_bus']=bad_mon_bus
    dic_faux_positif['bad_mon_dinosaure']=bad_mon_dinosaure
    dic_faux_positif['bad_mon_elepht']=bad_mon_elepht
    dic_faux_positif['bad_mon_fleur']=bad_mon_fleur
    dic_faux_positif['bad_mon_cheval']=bad_mon_cheval
    dic_faux_positif['bad_mon_nouriture']=bad_mon_nouriture
    
    dic_faux_positif['bad_nou_afrique']=bad_nou_afrique
    dic_faux_positif['bad_nou_plage']=bad_nou_plage
    dic_faux_positif['bad_nou_monument']=bad_nou_monument
    dic_faux_positif['bad_nou_bus']=bad_nou_bus
    dic_faux_positif['bad_nou_dinosaure']=bad_nou_dinosaure
    dic_faux_positif['bad_nou_elepht']=bad_nou_elepht
    dic_faux_positif['bad_nou_fleur']=bad_nou_fleur
    dic_faux_positif['bad_nou_cheval']=bad_nou_cheval
    dic_faux_positif['bad_nou_montagne']=bad_nou_montagne

    return dic_precision_concept, dico_moyenne,dic_faux_positif


#creation des variable pour le calcul des precision moyenne
def precision_moyenne(dico_moyenne):
    Afrique=0
    plage=0
    monument=0
    bus=0
    dinosaure=0
    éléphant=0
    fleur=0
    cheval=0
    montagne=0
    nourriture=0
    #calcul des precision moyenne   
       
    for key, val in dico_moyenne.items():
        val_i=(val/10)
    
        if((key<=99)):
                #print('Afrique')
            Afrique=Afrique+val_i
        elif((key>99)and(key<=199)):
                #print('plage')
            plage=plage+val_i
        elif((key>199)and(key<=299)):
                #print('monument')
            monument=monument+val_i
        elif((key>299)and(key<=399)):
                #print('bus')
            bus=bus+val_i
        elif((key>399)and(key<=499)):
                #print('dinosaure')
            dinosaure=dinosaure+val_i
        elif((key>499)and(key<=599)):
                #print('éléphant')
            éléphant=éléphant+val_i
        elif((key>599)and(key<=699)):
                #print('fleur')
            fleur=fleur+val_i
        elif((key>699)and(key<=799)):
                #print('cheval')
            cheval=cheval+val_i
        elif((key>799)and(key<=899)):
                #print('montagne')
            montagne=montagne+val_i
        elif((key>899)and(key<=999)):
                #print('nourriture')
            nourriture=nourriture+val_i
    return Afrique, plage, monument, bus, dinosaure, éléphant, fleur, cheval, montagne, nourriture


#lecture des images et recherches des différents voisins correspondant
def dico_voisin_each_image(files_img,preprocessed_image,kmeans):
    dic_neigbor={}
    dico_neighbor={}
    for image in files_img:
        image_lab=image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoint, descriptor = features(image, extractor)
              
        #histogramme de l'image correspondante
        histogram = build_histogram(descriptor, kmeans,150)
            
        #recherche des k plus proche voisin
        neighbor = NearestNeighbors(n_neighbors = 10)
            
        #entrainement pour le knn
        neighbor.fit(preprocessed_image)
            
        X_train = [histogram]
        dist, result = neighbor.kneighbors(X_train)
        
        result=np.reshape(result,(-1,1))
    
        #sauvegarde dans le dictionnaire de voisin
        file=image_lab[:-4]
        file=file[6:]
        file=int(file)
        dic_neigbor[file]=result
        dico_neighbor['image:',file]=result
        
    return dic_neigbor
