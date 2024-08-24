# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:40:55 2022

@author: mubashir khan
"""

#import the required libraries
import cv2
import numpy as np
import os
from sys import exit


# function to detect face from image
def face_detection(image_to_detect):
    #converting the image to grayscale since its required for eigen and fisher faces
    image_to_detect_gray = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
    
    # load the pretrained model for face detection
    # lbpcascade is recommended for LBPH
    # haarcascade is recommended for Eigenface and Fisherface haarcascade_frontalface_default.xml
    # download lpbcascade from https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface.xml
    face_detection_classifier = cv2.CascadeClassifier(r'C:\Users\mubashir khan\face_AI\model\lbpcascade_frontalface.xml')
    # can also use lbpcascade_frontalface.xml
    
    # detect all face locations in the image using classifier
    all_face_locations = face_detection_classifier.detectMultiScale(image_to_detect_gray)
    
    # if no faces are detected
    if (len(all_face_locations) == 0):
        return None, None
    
    #splitting the tuple to get four face positions
    x,y,width,height = all_face_locations[0]
    
    #calculating face coordinates
    face_coordinates = image_to_detect_gray[y:y+width, x:x+height]
    
    #training and testing images should be of same size for eigen and fisher faces
    #for LBPH its optional
    face_coordinates = cv2.resize(face_coordinates,(500,500))
    
    #return the face detected and face location
    return face_coordinates, all_face_locations[0]


# function to prepare training data
def prepare_training_data(images_dir, label_index):
    
    #list to hold all the faces and label indexes
    faces_coordinates = []
    labels_index = []
    
    #get the image names from the given directory
    images = os.listdir(images_dir)
    
    for image in images:
        image_path = images_dir + "/" + image
        training_image = cv2.imread(image_path)
        #display the current image taken for training
        cv2.imshow("Training in progress for "+names[label_index], cv2.resize(training_image,(700,700)))
        cv2.waitKey(100)
        
        #detect face using the method for detection
        face_coordinates, box_coordinates = face_detection(training_image)
        
        if face_coordinates is not None:
            # add the returned face to the list of faces
            faces_coordinates.append(face_coordinates)
            labels_index.append(label_index)
    
    return faces_coordinates, labels_index
/*
###### preprocessing ##############
names = []

#names.append("mahmuna khan")
#face_coordinates_mahmuna, labels_index_mahmuna = prepare_training_data(r"C:\Users\mubashir khan\face_AI\images\training\mahmuna",0)
    
names.append("aarez khan")
face_coordinates_aarez, labels_index_aarez = prepare_training_data(r"C:\Users\mubashir khan\face_AI\images\training\aarez",0)

names.append("taapse panu")
face_coordinates_taapse, labels_index_taapse = prepare_training_data(r"C:\Users\mubashir khan\face_AI\images\training\taapse",1)

names.append("donald trump")
face_coordinates_trump, labels_index_trump = prepare_training_data(r"C:\Users\mubashir khan\face_AI\images\training\trump",2)
   
face_coordinates =  face_coordinates_aarez +face_coordinates_taapse +face_coordinates_trump
labels_index =  labels_index_aarez + labels_index_taapse +labels_index_trump

#print total number of faces and names
print("Total faces:", len(face_coordinates))
print("Total names:", len(names))


######## training ###############

#create the instance of face recognizer
face_classifier = cv2.face.LBPHFaceRecognizer_create()
#cv2.face.EigenFaceRecognizer_create()
#cv2.face.FisherFaceRecognizer_create()
#cv2.face.LBPHFaceRecognizer_create()

face_classifier.train(face_coordinates,np.array(labels_index))



#########saving model for later use ###########
face_classifier.save(r"C:\Users\mubashir khan\face_AI\model\face_model.yml")

###load model###
import yaml

with open(r"C:\Users\mubashir khan\face_AI\model\face_model.yml") as f:
    
    face_classifier = yaml.load(f, Loader=yaml.FullLoader)
    #print(data)



######## prediction ##############
image_to_classify = cv2.imread(r"C:\Users\mubashir khan\face_AI\images\testing\1.jpg")

#make a copy of the image
image_to_classify_copy = image_to_classify.copy()

#get the face from the image
face_coordinates_classify, box_locations = face_detection(image_to_classify_copy)  

#if no faces are returned
if face_coordinates_classify is None:
    print("There are no faces in the image to classify")
    exit()
    
#if not none, we have predict the face
name_index, distance = face_classifier.predict(face_coordinates_classify)
name = names[name_index]
distance = abs(distance)

#draw bounding box and text for the face detected
(x,y,w,h) = box_locations
cv2.rectangle(image_to_classify,(x,y),(x+w, y+h),(0,255,0),3)
cv2.putText(image_to_classify,name,(x,y-5),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),2)

#show the image in window
cv2.imshow("Prediction "+name, cv2.resize(image_to_classify, (700,700)))
cv2.waitKey(0)
cv2.destroyAllWindows()
