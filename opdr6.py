import numpy as np
import cv2


plaatje_index = 0

def show_window(image, noDestroy = False):
    global plaatje_index
    cv2.imshow('plaatje' + str(plaatje_index), image)
    plaatje_index+=1
    cv2.waitKey(0)
    if not noDestroy:
        cv2.destroyAllWindows()


img = cv2.imread('Kroatie_grijs.jpg', 0)

helderder = img.copy()
helderder[helderder<=245] += 10 # avoid overflow
show_window(helderder, True)
factor = 0.5
helderder += (helderder * ((255-helderder) / 255) * factor).astype('uint8')
show_window(helderder, True)
helderder -= ((helderder - np.mean(helderder)) * 0.5).astype('uint8')

img = helderder
show_window(img)

min_gray, max_gray = img.min(), img.max()
print(max_gray)
print(min_gray)

rows, cols = img.shape[:2]

# for i in range(rows):
#     for j in range(cols):
#         k = img[i, j]
#         img[i, j] = (k - min_gray) * (255 / (max_gray - min_gray))
#

img = (img - img.min()) * (255 // (img.max() - img.min()))

show_window(img)



