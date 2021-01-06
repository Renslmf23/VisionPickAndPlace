import cv2
import numpy as np

img = cv2.imread("Disney.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#_, threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
edged = cv2.Canny(img, 50, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dilate = cv2.dilate(edged, kernel)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(dilate, kernel, iterations=1)
cv2.imshow("img", erosion)
cv2.waitKey(0)