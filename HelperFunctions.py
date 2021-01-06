import numpy as np
import cv2
from math import *
import Colors
import random
import MarkerHandler
from enum import Enum

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
    return img_Base, tuple(triangle_center), int(angle)


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


def prep_image(input_img, canny_thresh_min, canny_thresh_max):
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
    # result, unwarped = MarkerHandler.Unwarp(input_img)
    # if result is False:
    #     unwarped = input_img
    # show_window(unwarped)
    unwarped = input_img
    img = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(threshold, canny_thresh_min, canny_thresh_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilate = cv2.dilate(edged, kernel)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(dilate, kernel, iterations=1)
    return erosion


class Shape(Enum):
    large_tri = 1
    med_tri = 2
    small_tri = 3
    rect = 4
    diamond = 5
    empty = 6


class Piece:
    location = (0,0)
    rotation = 0
    shape = Shape.empty

    def __init__(self, loc, rot, shp):
        self.location = loc
        self.rotation = rot
        self.shape = shp

