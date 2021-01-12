import cv2


#---------------------LOAD IMAGES------------------------#
img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
img_Base = cv2.imread("test.jpg", cv2.IMREAD_COLOR)

img_rectangle = cv2.threshold(cv2.imread("vierkant.png", cv2.IMREAD_GRAYSCALE), 200, 255, cv2.THRESH_BINARY_INV) # read and threshold the rectangle image
img_triangle = cv2.threshold(cv2.imread("driehoek.png", cv2.IMREAD_GRAYSCALE), 200, 255, cv2.THRESH_BINARY_INV) # read and threshold the triangle image
img_diamond = cv2.threshold(cv2.imread("diamant.png", cv2.IMREAD_GRAYSCALE), 200, 255, cv2.THRESH_BINARY_INV) # read and threshold the diamond image


#-------------------APPLY MODIFIERS----------------------#

edged_rect = cv2.Canny(img_rectangle, 30, 200)
edged_tri = cv2.Canny(img_triangle, 30, 200)
edged_diam = cv2.Canny(img_diamond, 30, 200)

contours_rect, _ = cv2.findContours(edged_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_tri, _ = cv2.findContours(edged_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_diam, _ = cv2.findContours(edged_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find Canny edges
edged = cv2.Canny(threshold, 30, 200)
cv2.imshow("canny", edged)


# Finding Contours
contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    match = cv2.matchShapes(contour, contours_rect, 1, 0.0)
    if(match < 0.01):
        print("contour is rectangle")
        cv2.drawContour(img_Base, contour, -1, (255, 0, 0), 2)

    match = cv2.matchShapes(contour, contours_diam, 1, 0.0)
    if(match < 0.01):
        print("contour is diamond")
        cv2.drawContour(img_Base, contour, -1, (0, 255, 0), 2)

    match = cv2.matchShapes(contour, contours_tri, 1, 0.0)
    if(match < 0.01):
        print("contour is triangle")
        cv2.drawContour(img_Base, contour, -1, (0, 0, 255), 2)


cv2.imshow('image', img_Base)
cv2.imshow('thres', threshold)
cv2.waitKey()
