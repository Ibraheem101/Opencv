import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

people = []
for i in os.listdir('C:/Users/User/OneDrive/Pictures/train'):
    people.append(i)

haar_cascade = cv2.CascadeClassifier('C:/Users/User/OneDrive/Programming books/Opencv/haar_face.xml')


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

DIR = 'C:/Users/User/OneDrive/Pictures/test'

for person in ['Abdulsamad', 'Ibraheem']:
    path =  os.path.join(DIR, person)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img_arr = cv2.imread(img_path)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h,x:x+w]

            label, confidence = face_recognizer.predict(faces_roi)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv2.putText(img_arr, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv2.rectangle(img_arr, (x,y), (x+w,y+h), (0,255,0), thickness=2)

        cv2.imshow(f'{img}', img_arr)

cv2.waitKey()