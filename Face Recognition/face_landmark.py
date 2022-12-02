import os
import cv2
import dlib
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    is_true, frame = capture.read()
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()