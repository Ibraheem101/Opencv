# Pysource
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Trackbars
# Creating a trackbar on a video frame
def nothing(x):
    pass

cv2.namedWindow('frame')
cv2.createTrackbar('test', 'frame', 0, 100, nothing)

capture = cv2.VideoCapture(0)
while True:
    isTrue, frame = capture.read()

    pos = cv2.getTrackbarPos('test', 'frame')
    cv2.putText(frame, str(pos), (19, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0)), 

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
cv2.waitKey()