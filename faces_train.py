import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

people = []
for i in os.listdir('C:/Users/User/OneDrive/Pictures/train'):
    people.append(i)
print(people)
DIR = 'C:/Users/User/OneDrive/Pictures/train'
haar_cascade = cv2.CascadeClassifier('C:/Users/User/OneDrive/Programming books/Opencv/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 13)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features)
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

## Train face recognizer
face_recognizer.train(features, labels)

