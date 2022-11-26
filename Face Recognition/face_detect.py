import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

haar_cascades = cv2.CascadeClassifier('C:/Users/User/OneDrive/Programming books/Opencv/haar_face.xml')

capture = cv2.VideoCapture(0)
while True:
    is_true, frame = capture.read()
    face_rect = haar_cascades.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5)
    for (x, y, w, h) in face_rect:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), thickness=2)
        blank_mask = np.zeros(frame.shape[:2], dtype = 'uint8')
        mask = cv2.rectangle(blank_mask, (x, y), (x+w, y+h), 255, thickness=-1)

      
    masked_image = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow('frame', masked_image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

