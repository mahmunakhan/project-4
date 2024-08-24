# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:05:39 2022

@author: mubashir khan
"""

#import the required libraries
import cv2
import os
from openpyxl import Workbook
from datetime import datetime


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
    ################git###########
   
    
    
    #training and testing images should be of same size for eigen and fisher faces
    #for LBPH its optional
    #face_coordinates = cv2.resize(face_coordinates,(700,700))
    
    #return the face detected and face location
    return face_coordinates, all_face_locations[0]



###########################################################################################
def record_attendence(name):
    with open('Attendance.csv', 'r+') as file:
        # Read lines in csv file, except first line
        lines_in_file = file.read().splitlines()[1:]
        # Store only names
        names_in_file=list(map(lambda line : line.split(',')[0], lines_in_file))

        if not name in names_in_file:
            # Create datetime object
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_weekday = now.strftime("%A")
            current_month = now.strftime("%B")
            current_day_of_month = now.strftime("%d")

            # Write time and day details
            file.writelines(f"{name},{current_weekday},{current_month},{current_day_of_month},{current_time}\n")
            text_display = f"{name}, your attendence is recorded"
            print(text_display)
            
            
            
            
'''
####marking attendance###
# Load present date and time

def markAttendance(name):
    def markAttendance(name):
       with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')                                                                                             
                                                                                                        
 '''           

names = []
#names.append("Mahmuna Khan")
names.append("Aarez Khan")
names.append("Taapse Panu")
names.append("Donald Trump")


#create the instance of face recognizer
face_classifier = cv2.face.LBPHFaceRecognizer_create()
#cv2.face.EigenFaceRecognizer_create()
#########read the pretrained model ###########
face_classifier.read(r"C:\Users\mubashir khan\face_AI\model\face_model.yml")


#get the video
#video_stream = cv2.VideoCapture('images/testing/modi.mp4')
video_stream = cv2.VideoCapture(0)

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,image_to_classify = video_stream.read()
    
    ######## prediction ##############
    
    #make a copy of the image
    image_to_classify_copy = image_to_classify.copy()
    
    #get the face from the image
    face_coordinates_classify, box_locations = face_detection(image_to_classify_copy) 
   
    
   ####################################################
    
    
    
    #if no faces are returned
    if face_coordinates_classify is not None:
    
        #if not none, we have predict the face
        name_index, distance = face_classifier.predict(face_coordinates_classify)
        name = names[name_index]
        distance = abs(distance)
        #draw bounding box and text for the face detected
        (x,y,w,h) = box_locations
        cv2.rectangle(image_to_classify,(x,y),(x+w, y+h),(0,255,0),2)
        cv2.putText(image_to_classify,name,(x,y-5),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,0),2)
    
        
  
            
       # markAttendance(name)
        record_attendence(name)
    
    
    #show the image in window
    cv2.imshow("Prediction", cv2.resize(image_to_classify, (700,700)))
    #cv2.waitKey(10)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#release the stream and cam
#close all opencv windows open
video_stream.release()
cv2.destroyAllWindows()  

