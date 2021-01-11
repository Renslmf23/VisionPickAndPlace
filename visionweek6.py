import cv2
import HelperFunctions as HP
import numpy as np
from math import *
import time

img_orig = cv2.imread("Disney.jpg", cv2.IMREAD_COLOR)
img = img_orig.copy()
# # #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
morph = cv2.GaussianBlur(img, (5,5), 0)
#
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# take morphological gradient
gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
# split the gradient image into channels
image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

channel_height, channel_width, _ = image_channels[0].shape

# apply Otsu threshold to each channel
for i in range(0, 3):
    _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
# merge the channels
image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
img = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dilate = cv2.dilate(threshold, kernel)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(dilate, kernel, iterations=1)
HP.show_window(erosion)
der_x = img.copy()
for x in range(1, img.shape[0]-1):
    for y in range(1, img.shape[1]-1):
        der_x[x][y] = int((int(img[x+1][y]) - int(img[x-1][y])) / 2 + 128)

der_y = img.copy()
for y in range(1, img.shape[1]-1):
    for x in range(1, img.shape[0]-1):
        der_y[x][y] = int((int(img[x][y+1]) - int(img[x][y-1])) / 2 + 128)

deriv = (der_x + der_y) / 2

HP.show_window(deriv)

der = img_orig.copy()
for x in range(1, img.shape[0]-1):
    for y in range(1, img.shape[1]-1):
        der[x][y] = (int((int(img[x, y+1]) - int(img[x, y-1])) / 2 + 128) + int((int(img[x+1, y]) - int(img[x-1, y])) / 2 + 128))/2

HP.show_window(der)


# sobelx = cv2.Sobel(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=5)
# HP.show_window(sobely)

# strc_elm = cv2.imread("structuring_element.png")
# _, threshold_kernel = cv2.threshold(strc_elm.copy(), 210, 255, cv2.THRESH_BINARY_INV)
# _, threshold = cv2.threshold(img_orig.copy(), 210, 255, cv2.THRESH_BINARY_INV)
#
# kernel = threshold_kernel[:,:,0]
#
# img_orig = cv2.dilate(threshold, kernel, iterations=1)
#
# HP.show_window(img_orig)

# rows, cols = img.shape[:2]
# _, threshold = cv2.threshold(img_orig.copy(), 210, 255, cv2.THRESH_BINARY_INV)
# kernel_size, mid_point = 5, 3
#
# # # for i in range(7):
# # #     threshold = np.reshape([0 if not np.all(threshold[pos[0]:pos[0]+kernel_size, pos[1]:pos[1]+kernel_size] == 255) else x for pos, x in np.ndenumerate(threshold)],
# # #                (rows, cols)).astype(np.uint8)
# # #
# # # print(time.time() - start_time)
# # # HP.show_window(threshold, True)
# #
# for i in range(7):
#     eroded = threshold.copy()
#     for x in range(0, rows-kernel_size):
#         for y in range(0, cols-kernel_size):
#             if not np.all(threshold[x:x+kernel_size, y:y+kernel_size] == 255):
#                 eroded[x + mid_point, y + mid_point] = 0
#     threshold = eroded
# HP.show_window(threshold)
#
# for i in range(7):
#     eroded = threshold.copy()
#     for x in range(0, rows-kernel_size):
#         for y in range(0, cols-kernel_size):
#             if threshold[x + mid_point, y + mid_point] == 255:
#                 eroded[x:x + kernel_size, y:y+kernel_size] = 255
#     threshold = eroded
#
# HP.show_window(threshold)
