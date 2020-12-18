import numpy as np
import cv2
from math import *
import Colors
import random

# constants

min_contour_area = 1000
min_match_triangle = 0.15
min_match_rectangle = 0.15
min_match_diamond = 0.2
canny_thresh_min = 50
canny_thresh_max = 200

def create_test_image(angle=None):
    if angle is None:
        angle = random.randint(-180, 180)
    angle += 90
    sizex, sizey = 400, 400
    triangle_size = 100
    img_new = np.zeros([sizex, sizey,3],dtype=np.uint8)
    img_new.fill(255)
    point1 = [int(sizex/2 + (sin(radians(angle)) * triangle_size)), int(sizey/2 + (cos(radians(angle)) * triangle_size))]
    point2 = [int(sizex/2 - (sin(radians(angle)) * triangle_size)), int(sizey/2 - (cos(radians(angle)) * triangle_size))]
    point3 = [int(sizex/2 + (sin(radians(angle - 90)) * triangle_size)), int(sizey/2 + (cos(radians(angle - 90)) * triangle_size))]
    cv2.line(img_new, tuple(point1), tuple(point2), Colors.black, 2)
    cv2.line(img_new, tuple(point1), tuple(point3), Colors.black, 2)
    cv2.line(img_new, tuple(point3), tuple(point2), Colors.black, 2)

    return img_new


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


plaatje_index = 0


def show_window(image, noDestroy=False):
    '''Custom function to display image'''
    global plaatje_index
    cv2.imshow('plaatje' + str(plaatje_index), image)
    plaatje_index += 1
    cv2.waitKey(0)
    if not noDestroy:
        cv2.destroyAllWindows()


def get_base_triangle(m_triangle, img_Base):
    '''Finds the longest edge in the given triangle'''
    # maths: longest edge is base, calc rotation
    edge1 = Edge(m_triangle[0][0], m_triangle[1][0])
    edge2 = Edge(m_triangle[0][0], m_triangle[2][0])
    edge3 = Edge(m_triangle[2][0], m_triangle[1][0])
    base = edge3
    dist_point = m_triangle[0][0]
    if edge1.get_length() > edge2.get_length() and edge1.get_length() > edge3.get_length():
        base = edge1
        dist_point = m_triangle[2][0]
    elif edge2.get_length() > edge1.get_length() and edge2.get_length() > edge3.get_length():
        base = edge2
        dist_point = m_triangle[1][0]
    cv2.line(img_Base, tuple(base.point1), tuple(base.point2), Colors.get_random_color(), 2)
    angle = -base.get_angle()
    if base.get_middle()[0] + 3 > dist_point[0] > base.get_middle()[0] - 3:
        if dist_point[1] > base.get_middle()[1]:
            angle += 180
    elif base.get_middle()[0] < dist_point[0]:
        angle += 180
    elif base.get_middle()[1] < dist_point[1]:
        angle -= 180

    if angle < -180:
        angle += 180
    if angle > 180:
        angle -= 180

    triangle_center = Edge(base.get_middle(), dist_point).get_middle()
    cv2.putText(img_Base, str(int(angle)), tuple(triangle_center), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return img_Base


class Edge:
    point1 = []
    point2 = []

    def get_length(self):
        return sqrt((self.point1[0] - self.point2[0]) ** 2 + (self.point1[1] - self.point2[1]) ** 2)

    def get_angle(self, reference=None):
        reference_points = [[0, 0], [100, 0]]
        if reference is not None:
            reference_points = [reference.point1, reference.point2]
        if self.point2[0] - self.point1[0] == 0:
            return 90
        return degrees(atan(self.get_slope(reference_points) - self.get_slope(self)) / (1 + self.get_slope(reference_points) * self.get_slope(self)))

    def get_slope(self, ref_edge):
        if type(ref_edge) is Edge:
            return (ref_edge.point2[1] - ref_edge.point1[1]) / (ref_edge.point2[0] - ref_edge.point1[0])
        else:
            return (ref_edge[1][1] - ref_edge[0][1]) / (ref_edge[1][0] - ref_edge[0][0])

    def get_middle(self):
        return [int(self.point1[0] + ((self.point2[0] - self.point1[0]) / 2)), int(self.point1[1] + ((self.point2[1] - self.point1[1]) / 2))]

    def __init__(self, pnt1, pnt2):
        self.point1 = pnt1
        self.point2 = pnt2


def prep_image(input_img):
    '''Preps an image by denoising and fixing warp/perspective'''
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # morph = input_img
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #
    # # take morphological gradient
    # gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
    #
    # # split the gradient image into channels
    # image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
    #
    # channel_height, channel_width, _ = image_channels[0].shape
    #
    # # apply Otsu threshold to each channel
    # for i in range(0, 3):
    #     _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #     image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
    #
    # # merge the channels
    # image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    # # save the denoised image
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(threshold, canny_thresh_min, canny_thresh_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate = cv2.dilate(edged, kernel)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(dilate, kernel, iterations=1)
    return erosion


def find_contours(angle=None):
    '''Find the location and orientation of the pieces'''
    img = cv2.imread("renderrr.png", cv2.IMREAD_COLOR)
    # img = create_test_image(angle)
    img_Base = img.copy()

    prepped = prep_image(img)
    # Finding Contours
    contours, hierarchy = cv2.findContours(prepped,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        cv2.drawContours(img_Base, contour, -1, Colors.get_random_color(), 3)
        match = cv2.matchShapes(contour, contours_rect[0], 1, 0.0)
        if match < min_match_rectangle:
            print("contour is rectangle")
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)
            box = np.int0(box)
            cv2.putText(img_Base, str(int(rectangle[2])), tuple([int(rectangle[0][0]), int(rectangle[0][1])]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.drawContours(img_Base, [box], -1, (255, 0, 0), 2)

        match = cv2.matchShapes(contour, contours_diam[0], 1, 0.0)
        if match < min_match_diamond:
            print("contour is diamond")
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

        match = cv2.matchShapes(contour, contours_tri[0], 1, 0.0)
        if match < min_match_triangle:
            print("contour is triangle")
            _, triangle = cv2.minEnclosingTriangle(contour)
            triangle = np.int0(triangle)
            img_Base = get_base_triangle(triangle, img_Base)

    show_window(img_Base, True)

#for x in range(8):
find_contours()
cv2.destroyAllWindows()


#wit, groen, blauw, paars, geel, rood, oranje