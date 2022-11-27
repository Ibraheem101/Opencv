# Pysource
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

cv2.imshow('Result', result)

print(perspective.shape)










cv2.waitKey()