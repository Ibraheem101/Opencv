# Pysource
## Trackbars
## Object detection using HSV Color space and trackbars
## Perspective Transformations
## Affine Transformations
## Edge detection
## Line Detection with Hough Transform

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Trackbars
# Creating a trackbar on a video frame
def nothing(x):
    pass

# cv2.namedWindow('frame')
# cv2.createTrackbar('test', 'frame', 0, 100, nothing)

# capture = cv2.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()

#     pos = cv2.getTrackbarPos('test', 'frame')
#     cv2.putText(frame, str(pos), (19, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0)), 

#     # cv2.imshow('frame', frame)

    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     break

# capture.release()
# cv2.destroyAllWindows()

#### Object detection using HSV Color space and trackbars

# cv2.namedWindow('frame')
# cv2.createTrackbar('H', 'frame', 0, 180, nothing)
# cv2.createTrackbar('S', 'frame', 0, 255, nothing)
# cv2.createTrackbar('V', 'frame', 0, 255, nothing)



# capture2 = cv2.VideoCapture(0)
# while True:
#     isTrue, frame = capture2.read()
#     # cv2.imshow('frame', frame)

#     blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     #Let's 
#     h_position = cv2.getTrackbarPos('H', 'frame')
#     s_position = cv2.getTrackbarPos('S', 'frame')
#     v_position = cv2.getTrackbarPos('V', 'frame')
#     hsv_values = np.array([h_position, s_position, v_position])

#     lower = np.array([h_position, 0, 0])
#     upper = hsv_values
#     mask = cv2.inRange(hsv, lower, upper)
    
#     # cv2.imshow('hsv', hsv)
#     # cv2.imshow('mask', mask)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# capture2.release()
# capture2.destroyAllWindows()



#### Perspective Transformations
perspective = cv2.imread("C:/Users/User/OneDrive/Pictures/perspective.png")
cv2.circle(perspective, (191, 17), 4, (0, 0, 255), -1)
cv2.circle(perspective, (207, 17), 4, (0, 0, 255), -1)
cv2.circle(perspective, (16, 126), 4, (0, 0, 255), -1)
cv2.circle(perspective, (380, 126), 4, (0, 0, 255), -1)

pts1 = np.float32([[191, 17], [207, 17], [16, 126], [380, 126]])
pts2 = np.float32([[0, 0], [400, 0], [0, 700], [400, 700]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(perspective, matrix, (400, 700))

# cv2.imshow('Result', result)
# print(perspective.shape)



#### Affine transformation
grid = cv2.imread("C:/Users/User/OneDrive/Pictures/grid.png")
cv2.circle(grid, (55, 75), 6, (0,0,255), -1)
cv2.circle(grid, (410, 75), 6, (0,0,255), -1)
cv2.circle(grid, (55, 580), 6, (0,0,255), -1)

pts1 = np.float32([[55, 75], [410, 75], [55, 580]])
pts2 = np.float32([[55, 75], [410, 75], [200, 580]])

aff_matrix = cv2.getAffineTransform(pts1, pts2)
aff_result = cv2.warpAffine(grid, aff_matrix, (grid.shape[1], grid.shape[0]))
# cv2.imshow('Affine', aff_result)

# cv2.imshow('grid', grid)
print(grid.shape)


#### Edge Detection
bumblebee = cv2.imread("C:/Users/User/OneDrive/Pictures/Bumblebee.jpg")
bumblebee_resized = cv2.resize(bumblebee, (1400, 800))
canny = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(bumblebee_resized, cv2.COLOR_BGR2GRAY), (3, 3), cv2.BORDER_DEFAULT), 50, 75)
# cv2.imshow('Canny', canny)


#### Line Detection with Hough Transform
line = cv2.imread("C:/Users/User/OneDrive/Pictures/broken line.png")
gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
canny_line = cv2.Canny(gray, 50, 75)
# cv2.imshow('line', canny_line)

lines = cv2.HoughLinesP(canny_line, 1, np.pi/180, 5, maxLineGap = 80)
# print(lines)

for linex in lines:
    x1, y1, x2, y2 = linex[0]
    cv2.line(line, (x1, y1), (x2, y2), (0, 0, 255), 4)
# cv2.imshow('line', line)

## Detect lanes in a video
video = cv2.VideoCapture("C:/Users/User/Videos/4K Video Downloader/Sheikh Zayed Road   Dubai UAE.mp4")
while True:
    _, frame = video.read()
    # cv2.imshow('frame', frame)

    low_lane = np.array([190, 188, 192])
    max_lane = np.array([209, 205, 199])

    mask = cv2.inRange(frame, low_lane, max_lane)
    lane_edge = cv2.Canny(mask, 50, 75)
    cv2.imshow('edge', lane_edge)

    lines = cv2.HoughLinesP(lane_edge, 1, np.pi/180, 35, maxLineGap=30)
    if lines is not None:
        for linex in lines:
            x1, y1, x2, y2 = linex[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
    # cv2.imshow('frame', frame)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
    
lane = cv2.imread("C:/Users/User/OneDrive/Pictures/lane.png")
# cv2.imshow('lane', lane)












cv2.waitKey()