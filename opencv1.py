import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Rescaling imagees and video frames

def rescaleFrame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

########### Read Image
#################################
# img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg", cv2.IMREAD_GRAYSCALE)
# resized_img = rescaleFrame(img)

# cv2.imshow('Arafims', img)
# cv2.imshow('Resized Arafims', resized_img)
# cv2.waitKey() 
#################################

########### Read Videos
################################
# capture = cv2.VideoCapture("C:/Users/User/Videos/4K Video Downloader/NEW LOWEST LANDING Wizzair Airbus A321neo Landing at Skiathos Airport   JSI Plane Spotting [4K].mp4")

# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescaleFrame(frame, scale=.2)


#     #cv2.imshow('Video', frame)
#     cv2.imshow('Video Resized', frame_resized)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()
################################

########### Draw on images and Videos
##########################################
# blank = np.zeros((500, 500, 3), dtype = 'uint8')
# cv2.imshow('blank', blank)

## Color a part of the image
# blank[20:70, 20:70] = 0,255,0
# cv2.imshow('New', blank)

## Draw a rectangle
# cv2.rectangle(blank, (0,0), (200, 200), (255,0,0), thickness=1) #Specify thickness = -1 or cv2.
# cv2.imshow('Rectangle', blank)

## Draw a circle
# cv2.circle(blank, (125, 125), 40, (0,0,255), thickness = -1)
# cv2.imshow('Circle', blank)

## Draw a line
# cv2.line(blank, (0,0), (125, 125), (255,0,0), thickness=3)
# cv2.imshow('Line', blank)

## Write text
# cv2.putText(blank, 'Hey there!', (200, 200), cv2.FONT_HERSHEY_TRIPLEX 1.0, (255,255,255), thickness=2)
# cv2.imshow('Text', blank)

# cv2.waitKey()
######################################

#### Basic functions
#############################
img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg")

## Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)

## Gaussian Blur
blur = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

## Canny Edge Detector
canny = cv2.Canny(blur, 175, 175)
cv2.imshow('Canny', canny)

cv2.waitKey()
