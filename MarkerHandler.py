import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def get_perspective_points(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    plt.figure()
    plt.imshow(frame_markers)
    if len(ids) < 4:
        return False, None
    result = [0, 0, 0, 0]
    for i in range(4):
        c = corners[i][0]
        result[i] = [c[:, 0].mean()], [c[:, 1].mean()]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
    plt.legend()
    plt.show()
    return True, np.float32([(result[0]), (result[1]), (result[2]), (result[3])])


def unwarp(frame):
    h, w = frame.shape[:2]
    dst = np.float32([(w, 0),
                      (0, 0),
                      (w, h),
                      (0, h)])
    result, src = get_perspective_points(frame)
    if result is False:
        return False, None
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    return True, cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)


img = cv2.imread("Marker_test.jpg", cv2.IMREAD_COLOR)
result, unwarped = unwarp(img)
cv2.imshow('plaatje', unwarped)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('warped_img.jpg', unwarped)