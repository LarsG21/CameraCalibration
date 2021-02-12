import cv2
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
img = aruco.drawMarker(aruco_dict, id=1,sidePixels=200)
cv2.imshow("Aruco", img)
cv2.imwrite("ArucoID1.png", img)
cv2.waitKey(0)