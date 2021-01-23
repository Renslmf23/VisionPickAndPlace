import numpy as np
import cv2
from math import *
import Colors
import random
from HelperFunctions import *
import snap7.client as c
from snap7.util import *
from snap7.snap7types import *

# constants

min_contour_area = 1000
min_match_triangle = 0.15
min_match_rectangle = 0.15
min_match_diamond = 0.2
canny_thresh_min = 50
canny_thresh_max = 200

threshold_small, threshold_med = 8000, 13000


# Variables
send_to_plc = True

# ---------------------LOAD IMAGES------------------------#
_, img_rectangle = cv2.threshold(cv2.imread("reference/vierkant.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                                 cv2.THRESH_BINARY_INV)  # read and threshold the rectangle image
_, img_triangle = cv2.threshold(cv2.imread("reference/driehoek.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                                cv2.THRESH_BINARY_INV)  # read and threshold the triangle image
_, img_diamond = cv2.threshold(cv2.imread("reference/diamant.png", cv2.IMREAD_GRAYSCALE), 200, 255,
                               cv2.THRESH_BINARY_INV)  # read and threshold the diamond image


# -------------------APPLY MODIFIERS----------------------#
edged_rect = cv2.Canny(img_rectangle, 30, 200)
edged_tri = cv2.Canny(img_triangle, 30, 200)
edged_diam = cv2.Canny(img_diamond, 30, 200)

contours_rect, _ = cv2.findContours(edged_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_tri, _ = cv2.findContours(edged_tri.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_diam, _ = cv2.findContours(edged_diam.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# -------------------MEMORY LOCATIONS----------------------#
begin_stukjes = 54
stukjes_offset = 28


#Color thresholds




def find_contours(img, angle=None):
    '''Find the location and orientation of the pieces'''
    #img = cv2.imread("capture0.jpg", cv2.IMREAD_COLOR)
    # img = create_test_image(angle)
    img_Base = img.copy()
    res, unwarped_img = MarkerHandler.unwarp(img)
    if not res:
        print("unwarp failed, using regular image...")
        unwarped_img = img
    else:
        img_Base = unwarped_img

    prepped = prep_image(unwarped_img, canny_thresh_min, canny_thresh_max)
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
            angle = -rectangle[2]

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
            #cv2.drawContours(img_Base, [box], -1, (255, 0, 0), 2)
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

    show_window(img_Base)
    for piece in pieces:
        print(str(piece.location) + ", " + piece.shape.name)
    pieces.sort(key=sort_by_enum)
    if send_to_plc:
        for i in range(len(pieces)):
            WriteMemory(begin_stukjes + stukjes_offset * i, pieces[i].location[0], S7WLReal)
            WriteMemory(begin_stukjes + 4 + stukjes_offset * i, pieces[i].location[1], S7WLReal)
            WriteMemory(begin_stukjes + 12 + stukjes_offset * i, pieces[i].rotation, S7WLWord)


def sort_by_enum(e):
    return int(e.shape.value)


def WriteMemory(byte, datatype, value):
    result = plc.read_area(areas['DB'], 1, byte, datatype)

    if datatype == S7WLByte or datatype == S7WLWord:
        set_int(result, 0, value)
    elif datatype == S7WLReal:
        set_real(result, 0, value)
    elif datatype == S7WLDWord:
        set_dword(result, 0, value)
    elif datatype == S7WLWord:
        set_int(result, 0, value)
    plc.write_area(areas['DB'], 1, byte, result)


if __name__ == "__main__":

    plc = c.Client()
    #plc.connect('192.168.0.1', 0, 1)
    cap = cv2.VideoCapture(3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #1280 * 720
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    index = 0
    while(True):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("video", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e'):
                cv2.imwrite("capture" + str(index) + ".jpg", frame)
                index+=1
                find_contours(frame)
    cap.release()
    cv2.destroyAllWindows()


#wit, groen, blauw, paars, geel, rood, oranje