#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:28:03 2019

@author: eric
"""

import cv2
import operator

face_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_frontalface_alt2.xml')
profile_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_profileface.xml')

#lancement de notre camera
cap=cv2.VideoCapture(0)

#recuperation de la largeur de la webcam sa nous servira pour rogner les images
width=int(cap.get(3))
marge=70

id=0
while True:
    #frame: contient l'image
    ret, frame=cap.read()
    tab_face=[]
    
    #mesure de temps d'execution algo  de notre image avec getTickcount()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #face contient une liste de quadruplet que nous allons récuperer
    #minSize : permet de suprimer les rectangle ayant une valeur innferieur a 5,5
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(5, 5))
    
    #x,y: cordonnée; w,h: hauteur et largeur;
    for x, y, w, h in face:
        cv2.imwrite('data/fichier_image/photos_camarade_classe/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        #cv2.rectangle(frame,(x,y),(x+w, y+h), (255, 0, 0), 2)
        tab_face.append([x, y, x+w, y+h])
        id+=1
    face=profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    
    for x, y, w, h in face:
        cv2.imwrite('data/fichier_image/photos_camarade_classe/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        tab_face.append([x, y, x+w, y+h])
        id+=1
    gray2=cv2.flip(gray, 1)
    face=profile_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=4)
    
    for x, y, w, h in face:
        cv2.imwrite('data/fichier_image/photos_camarade_classe/p-{:d}.png'.format(id), frame[y:y+h, x:x+w])
        tab_face.append([width-x, y, width-(x+w), y+h])
        id+=1
    tab_face=sorted(tab_face, key=operator.itemgetter(0, 1))
    index=0
   
    for x, y, x2, y2 in tab_face:
        if not index or (x-tab_face[index-1][0]>marge or y-tab_face[index-1][1]>marge):
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
        index+=1
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(3):
            ret, frame=cap.read()
    
    #affichage du temps d'execution
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    
    #affichage du text dans la video
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', frame)
cap.release()
cv2.destroyAllWindows()