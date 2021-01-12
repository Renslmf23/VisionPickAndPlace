import cv2
import numpy as np

threshold = 159985

img_in = cv2.imread("full.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template_full.jpg", cv2.IMREAD_GRAYSCALE)
img_in = cv2.Canny(img_in, 50, 200)
template = cv2.Canny(template, 50, 200)

img = img_in.copy()
# Apply template Matching
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
print(max_val)
if max_val > threshold:
    cv2.rectangle(img, max_loc, (max_loc[0] + 125, max_loc[1] + 125), 255, 2)
    cv2.imshow(" mathced",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("bakje matched")