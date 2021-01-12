import cv2
import numpy as np
import random

# Lees een foto in (grijswaarden)
img = cv2.imread('Cheshire.JPG', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape[:2]


img = np.reshape([255 if random.random() < 0.01 else x for pos, x in np.ndenumerate(img)], (rows, cols)).astype(np.uint8)

img = np.reshape([0 if random.random() < 0.02 else x for pos, x in np.ndenumerate(img)], (rows, cols)).astype(np.uint8)

cv2.imshow('grijsw met png', img)
cv2.imwrite('ruis jpg.jpg', img)
cv2.waitKey(0)

img = cv2.imread('ruis jpg.jpg', cv2.IMREAD_GRAYSCALE)

def get_median(in_img, x, y):
    median_radius = 3
    begin_x = (x - median_radius if x - median_radius > 0 else 0)
    end_x = (x + median_radius if x + median_radius < rows - 1 else rows - 1)
    begin_y = (y - median_radius if y - median_radius > 0 else 0)
    end_y = (y + median_radius if y + median_radius < cols - 1 else cols - 1)
    return np.median(in_img[begin_x:end_x, begin_y:end_y])


img = np.reshape([get_median(img, pos[0], pos[1]) if (x > 252 or x < 3) else x for pos, x in np.ndenumerate(img)], (rows, cols)).astype(np.uint8)

cv2.imshow('Willekeurige grijswaarden', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('gefilterde jpg.jpg', img)
