import cv2
import numpy as np
import pandas as pd
import imutils
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg")
spectrum = cv2.imread("C:/Users/User/OneDrive/Pictures/spectrum.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("C:/Users/User/OneDrive/Pictures/Ibraheemk.jpg")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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

rotated = rotate(img, 50, (20, 20))
# cv2.imshow('Rotated', rotated)

## Rotating an image while preserving boundaries
rotated2 = imutils.rotate_bound(img, 50)
# cv2.imshow('Rotate Bound', rotated2)

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
# cv2.waitKey()
#########################

#### Colour spaces
########################
# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow('Lab', img_lab)

# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV', img_hsv)

# img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2)
# cv2.imshow('Lab', img_lab)

# cv2.waitKey()
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
# cv2.imshow('Blue', blue)
# cv2.imshow('Green', green)
# cv2.imshow('Red', red)

# merged = cv2.merge([b, g, r])
# cv2.imshow('Merged', merged)
# cv2.waitKey()

#### Blurring
############################

## Gaussian blur
# gaussian = cv2.GaussianBlur(img, (3, 3), 0)
# cv2.imshow('Gaussian', gaussian)

# ## Average blur
# average = cv2.blur(img, (3, 3))
# cv2.imshow('Average', average)

# ## Median blur
# median = cv2.medianBlur(img, 3)
# cv2.imshow('Median', median)

# bilateral = cv2.bilateralFilter(img, 10, 15, 15)
# cv2.imshow('Bilateral', bilateral)

# cv2.waitKey()
################################

#### Bitwise Operations
##############################

# blank = np.zeros((500, 500), dtype='uint8')
# blank2 = np.zeros((500, 500), dtype='uint8')

# # cv2.line(blank, (250, 30), (30, 470), 255)
# # cv2.line(blank, (250, 30), (470, 470), 255)
# # cv2.line(blank, (30, 470), (470, 470), 255)

# # Fill triangle
# pts = np.array([[250, 30], [30, 470], [470, 470]])
# cv2.fillPoly(blank, [pts], color=255)

# cv2.circle(blank2, (250, 250), 200, 255, -1)

# cv2.imshow('Blank', blank)
# cv2.imshow('Blank2', blank2)

# ## Bitwise AND
# And = cv2.bitwise_and(blank, blank2)
# cv2.imshow('AND', And)

# ## Bitwise OR
# Or = cv2.bitwise_or(blank, blank2)
# cv2.imshow('OR', Or)

# ## Bitwise XOR
# xor = cv2.bitwise_xor(blank, blank2)
# cv2.imshow('XOR', xor)

# ## Bitwise NOT
# Not = cv2.bitwise_not(blank)
# cv2.imshow('Not', Not)
###############################

#### Masking
###############################
# blank_mask = np.zeros(img.shape[:2], dtype = 'uint8')
# mask = cv2.circle(blank_mask, (250, 250), 200, 255, -1)

# masked_image = cv2.bitwise_and(img, img, mask = mask)
# cv2.imshow('Masked Image', masked_image)

#### Histograms
##############################
## Gray Histogram
gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixels')
plt.xlim(0, 256)
plt.plot(gray_hist)
# plt.show()

# ## Color Histogram
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
	color_hist = cv2.calcHist([img], [i], None, [256], [0, 256])
	plt.title('Color Histogram')
	plt.xlabel('Bins')
	plt.ylabel('Pixels')
	plt.xlim(0, 256)
	plt.plot(color_hist, color = color)
# plt.show()

#######################################

#### Thresholding/ Binarizing
#############################
## Simple thresholding
threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('Simple threshold', thresh)

threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('Simple threshold inverse', thresh)
threshold, thresh_inv = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('Simple threshold inverse', thresh_inv)

## Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
										cv2.THRESH_BINARY, 11, 3)
# cv2.imshow('Adaptive Threshold', adaptive_thresh)

adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
										cv2.THRESH_BINARY, 11, 3)
# cv2.imshow('Adaptive Threshold Gaussian', adaptive_thresh_gaussian)

#####################################

#### Edge detection
##############################
## Laplacian edge
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
# cv2.imshow('Laplacian', lap)

## Sobel edge detector
# Computes in x and y directions
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# cv2.imshow('Sobelx', sobelx)
# cv2.imshow('Sobely', sobely)

sobel = cv2.bitwise_or(sobely, sobelx)
# cv2.imshow('Sobel combined', sobel)
##########################################

#### Face Detection
####################################
## Haar Cascades
person = cv2.imread("C:/Users/User/OneDrive/Pictures/Ibraheemk.jpg")
people = cv2.imread("C:/Users/User/OneDrive/Pictures/people.jpg")
people2 = cv2.imread("C:/Users/User/OneDrive/Pictures/nikah.jpg")

gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
gray_people = cv2.cvtColor(people, cv2.COLOR_BGR2GRAY)


haar_cascade = cv2.CascadeClassifier('C:/Users/User/OneDrive/Programming books/Opencv/haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(people2, scaleFactor = 1.1, minNeighbors = 5)

print(len(faces_rect))
for (x, y, w, h) in faces_rect:
	cv2.rectangle(people2, (x, y), (x+w, y+h), (0,0,255), thickness=2)

cv2.imshow('Faces', people2)
##################################

#### Color detection
############################
# Define list of boundaries
# boundaries = [
# 	([5, 5, 255], [5, 78, 255]),
# 	([5, 255, 182], [195, 255, 5]),
# 	([255, 195, 5], [255, 5, 126])
# ]
# blank2 = np.zeros((500, 500, 3), dtype='uint8')
# blank2[:] = (0,0,255)
# cv2.imshow('red', blank2)

# for (lower, upper) in boundaries:
# 	# create NumPy arrays from the boundaries
# 	lower = np.array(lower)
# 	upper = np.array(upper)
# 	# find the colors within the specified boundaries and apply
# 	# the mask
# 	mask = cv2.inRange(img2, lower, upper)
# 	output = cv2.bitwise_and(img2, img2, mask = mask)
# 	# show the images
# 	# cv2.imshow("images", np.hstack((img2, output)))
# 	# cv2.waitKey()








cv2.waitKey()
