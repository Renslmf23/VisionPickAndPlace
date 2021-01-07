import cv2
import HelperFunctions as HP
import numpy as np
from math import *

img_orig = cv2.imread("Disney.jpg")
img = img_orig.copy()
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
morph = cv2.GaussianBlur(img, (5,5), 0)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
# morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#
# # take morphological gradient
# gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
# # split the gradient image into channels
# image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
#
# channel_height, channel_width, _ = image_channels[0].shape
#
# # apply Otsu threshold to each channel
# for i in range(0, 3):
#     _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#     image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
# # merge the channels
# image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
# img = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)
#
# HP.show_window(img)
# der_x = img.copy()
# for x in range(1, img.shape[0]-1):
#     for y in range(1, img.shape[1]-1):
#         der_x[x][y] = int((int(img[x+1][y]) - int(img[x-1][y])) / 2 + 128)
#
# der_y = img.copy()
# for y in range(1, img.shape[1]-1):
#     for x in range(1, img.shape[0]-1):
#         der_y[x][y] = int((int(img[x][y+1]) - int(img[x][y-1])) / 2 + 128)
#
# img = (der_x + der_y) / 2
#
# HP.show_window(img)


sobelx = cv2.Sobel(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=5)
HP.show_window(sobely)