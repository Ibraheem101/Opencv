import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg")

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
# img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg")

# ## Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray)

# ## Gaussian Blur
# blur = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
# cv2.imshow('Blur', blur)

# ## Canny Edge Detector
# canny = cv2.Canny(blur, 175, 175)
# cv2.imshow('Canny', canny)

# ## Cropping
# crop = img[:500, :500]
# cv2.imshow('Crop', crop)

# cv2.waitKey()
#######################################

########### Image Transformations
####################################
## Translation
def translate(img, tx, ty):
    transMat = np.float32([[1, 0, tx], [0, 1, ty]])
    width = img.shape[1] 
    height = img.shape[0]
    dimensions = (width, height)
    return cv2.warpAffine(img, transMat, dimensions)

translated = translate(img, 100, 100) # Right and down by 100 px
translated2 = translate(img, -100, 100) # Left and down by 100 px

# cv2.imshow('Translated', translated2)

## Rotation
def rotate(img, angle, rotation_point = None):
    (height, width) = img.shape[:2] # We don't need the number of channels
    dimensions = (width, height)
    if rotation_point == None:
        rotation_point = (width // 2, height // 2)
    
    rotationMat = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
    return cv2.warpAffine(img, rotationMat, dimensions)

# rotated = rotate(img, 50, (20, 20))
# cv2.imshow('Rotated', rotated)

# ## Resizing
# resized = cv2.resize(img, (500,500), interpolation=cv2.INTER_AREA)
# cv2.imshow('Resized', resized)

# ## Flipping
# flipped = cv2.flip(img, -1)
# cv2.imshow('Flipped', flipped)


########### Contour Detection
####################################
cv2.imshow('Arafims', img)

# blank = np.zeros(img.shape, dtype='uint8')
# # cv2.imshow('Blank', blank)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow('Gray', gray)

# blur = cv2.GaussianBlur(gray, (3,3), cv2.BORDER_DEFAULT)
# cv2.imshow('Blur', blur)

# canny = cv2.Canny(blur, 125, 175)
# cv2.imshow('Canny Edges', canny)

# contours, heirarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(img, contours, -1, (0,0,255))
# cv2.imshow('Contours Drawn', img)

# # Print number of contours
# print("Number of contours: " + str(len(contours)))
cv2.waitKey()
#########################

#### Colour spaces
########################
# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow('Lab', img_lab)

# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV', img_hsv)

# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2)
# cv2.imshow('Lab', img_lab)

cv2.waitKey()
#######################

#### Color channels
#########################

blank = np.zeros(img.shape[:2], dtype='uint8')
b, g, r = cv2.split(img)

blue = cv2.merge([b, blank, blank])
green = cv2.merge([blank, g, blank])
red = cv2.merge([blank, blank, r])

## Display channels in grayscale
# cv2.imshow('Blue', b)
# cv2.imshow('Green', g)
# cv2.imshow('Red', r)

## View channels in color
cv2.imshow('Blue', blue)
cv2.imshow('Green', green)
cv2.imshow('Red', red)

# merged = cv2.merge([b, g, r])
# cv2.imshow('Merged', merged)
cv2.waitKey()
