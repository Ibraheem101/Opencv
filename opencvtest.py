import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/User/OneDrive/Pictures/Arafims.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.ceil(np.random.randn(609, 1368) * 0.2)
noise_img = img_gray + noise

#cv2.waitKey()

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

noisy('gauss',img)