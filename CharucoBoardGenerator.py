import cv2
from cv2 import aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]
board = aruco.CharucoBoard_create(7, 5, 1, 0.5, aruco_dict)

imboard = board.draw((2000, 2000))
cv2.imwrite("chessboard3.png", imboard)