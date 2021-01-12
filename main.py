import numpy as np
import cv2

# Lees een foto in (grijswaarden)
img = cv2.imread('Kroatie_grijs.jpg', 0)
# mg = cv2.resize(img, (800, 600))

rows, cols = img.shape[:2]

for i in range(rows):
    for j in range(cols):
        k = img[i, j]
        img[i, j] = (k // 32) * 32

# opdracht 2 a + b
print("hoogte: " + str(rows) + ". Breedte: " + str(cols))
print(len(img[img>127]))

# opdracht 3a
img_inv = 255 - img

#opdracht 3b
thresh = 127
_, binair = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

#opdracht 4a, b, c
helderder = img.copy()
helderder += 10
factor = 0.5
helderder += (helderder * ((255-helderder) / 255) * factor).astype('uint8')
helderder -= ((helderder - np.mean(helderder)) * 0.5).astype('uint8')

#opdracht 5
img_limit = img.copy()

img_limit[img_limit < 30] = 30
img_limit[img_limit > 200] = 200

def show_window(image):
    cv2.imshow('plaatje', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_window(img)
show_window(img_limit)
cv2.imwrite("kroatie_limit.jpg", img_limit)
# show_window(binair)
# show_window(helderder)
# show_window(helderder_mult)


# cv2.imwrite('Kroatie_out.jpg', img)