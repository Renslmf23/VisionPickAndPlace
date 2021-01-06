import numpy as np
import cv2
from math import *
import Colors
import random
from HelperFunctions import *

# constants

min_contour_area = 1000
min_match_triangle = 0.15
min_match_rectangle = 0.15
min_match_diamond = 0.2
canny_thresh_min = 50
canny_thresh_max = 200

threshold_small, threshold_med = 10000, 18000

# ---------------------LOAD IMAGES------------------------#
_, img_rectangle = cv2.threshold(cv2.imread("vierkant.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                                 cv2.THRESH_BINARY_INV)  # read and threshold the rectangle image
_, img_triangle = cv2.threshold(cv2.imread("driehoek.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                                cv2.THRESH_BINARY_INV)  # read and threshold the triangle image
_, img_diamond = cv2.threshold(cv2.imread("diamant.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                               cv2.THRESH_BINARY_INV)  # read and threshold the diamond image


# -------------------APPLY MODIFIERS----------------------#
edged_rect = cv2.Canny(img_rectangle, 30, 200)
edged_tri = cv2.Canny(img_triangle, 30, 200)
edged_diam = cv2.Canny(img_diamond, 30, 200)

contours_rect, _ = cv2.findContours(edged_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_tri, _ = cv2.findContours(edged_tri.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_diam, _ = cv2.findContours(edged_diam.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find Canny edges



def find_contours(angle=None):
    '''Find the location and orientation of the pieces'''
    img = cv2.imread("renderrr.png", cv2.IMREAD_COLOR)
    # img = create_test_image(angle)
    img_Base = img.copy()

    prepped = prep_image(img, canny_thresh_min, canny_thresh_max)
    # Finding Contours
    contours, hierarchy = cv2.findContours(prepped,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        cv2.drawContours(img_Base, contour, -1, Colors.get_random_color(), 3)
        match = cv2.matchShapes(contour, contours_rect[0], 1, 0.0)
        # check if piece is rectangle:
        if match < min_match_rectangle:
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)
            box = np.int0(box)
            cv2.putText(img_Base, str(int(rectangle[2])), tuple([int(rectangle[0][0]), int(rectangle[0][1])]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.drawContours(img_Base, [box], -1, (255, 0, 0), 2)
            pieces.append(Piece(tuple([int(rectangle[0][0]), int(rectangle[0][1])]), int(angle), Shape.rect)) # add piece to database

        match = cv2.matchShapes(contour, contours_diam[0], 1, 0.0)
        # check if piece is diamond:
        if match < min_match_diamond:
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)
            box = np.int0(box)
            angle = -rectangle[2]
            if rectangle[1][0] < rectangle[1][1]:
                angle += 90
            cv2.drawContours(img_Base, [box], -1, (255, 0, 0), 2)
            cv2.putText(img_Base, str(int(angle)), tuple([int(rectangle[0][0]), int(rectangle[0][1])]),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.drawContours(img_Base, contour, -1, (0, 255, 0), 2)
            pieces.append(Piece(tuple([int(rectangle[0][0]), int(rectangle[0][1])]), int(angle), Shape.diamond)) # add piece to database

        match = cv2.matchShapes(contour, contours_tri[0], 1, 0.0)
        # check if piece is triangle:
        if match < min_match_triangle:
            shape = Shape.large_tri
            if cv2.contourArea(contour) < threshold_small:
                shape = Shape.small_tri
            elif cv2.contourArea(contour) < threshold_med:
                shape = Shape.med_tri

            _, triangle = cv2.minEnclosingTriangle(contour)
            triangle = np.int0(triangle)
            img_Base, loc, rot = get_base_triangle(triangle, img_Base) # get the triangle data
            pieces.append(Piece(loc, rot, shape)) # add piece to database

    show_window(img_Base, True)
    for piece in pieces:
        print(str(piece.location) + ", " + piece.shape.name)

#for x in range(8):
find_contours()
cv2.destroyAllWindows()


#wit, groen, blauw, paars, geel, rood, oranje