#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:37:29 2019

@author: eric
"""

import cv2
import pickle
import numpy as np
import common as c

face_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_frontalface_alt2.xml')
profile_cascade=cv2.CascadeClassifier('data/fichier_xml/haarcascade_profileface.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()

#importation et lecture de notre modèle formé
recognizer.read("trainner.yml")
id_image=0
color_info=(255, 255, 255)
color_ko=(0, 0, 255)
color_ok=(0, 255, 0)

with open("labels.pickle", "rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k, v in og_labels.items()}

#cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture('data/fichier_video/presence_1.mp4')
cap=cv2.VideoCapture('data/fichier_video/classe_6_classe/inoussa1.mp4')
while True:
    ret, frame=cap.read()
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(60, 60))
    for (x, y, w, h) in faces:
        #changement de la taille des photos
        roi_gray=cv2.resize(gray[y:y+h, x:x+w], (150, 150))
        #id_, conf=recognizer.predict(cv2.resize(roi_gray, (c.min_size, c.min_size)))
        id_, conf=recognizer.predict(roi_gray)
        if conf<=70:
             color=color_ok
             name=labels[id_]
        else:
            color=color_ko
            name="Inconnu"
        label=name+" "+'{:5.2f}'.format(conf)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
    cv2.imshow('video de la classe', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100):
            ret, frame=cap.read()

cv2.destroyAllWindows()
print("Fin de la video")
