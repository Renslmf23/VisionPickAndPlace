import cv2


img = cv2.imread("capture0.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
cv2.imshow("Test", img)
cv2.waitKey(0)