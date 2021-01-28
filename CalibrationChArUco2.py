import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
#matplotlib nbagg
import utils


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

board = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=1,
        markerLength=0.5,
        dictionary=aruco_dict)


directory1 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher"
directory2 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Nachher"

print(os.getcwd())
print('Path Exists ?')

print(os.path.exists(directory1))
print(os.path.exists(directory2))

counter = 0
images = []

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = im
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.waitKey(1000)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
            else:
                print('Something is Non !!!!')
                if res2[1] is None:
                    print('res2[1]')
                if res2[2] is None:
                    print('res2[2]')
                if len(res2[1]) < 3:
                    print('res2[1]<3')
                if not decimator%1==0:
                    print('not decimator%1==0')

        show = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=res2[1],
            charucoIds=res2[2])
        cv2.imshow('Markers_Detected', show)
        cv2.waitKey(1000)

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize



cap = cv2.VideoCapture(0)



while True:
    succsess, img = cap.read()
    cv2.imshow("Calib_ChArUco", img)

    if cv2.waitKey(1) & 0xff == ord('x'):
        cv2.putText(img, "Captured", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
        cv2.imshow("Calib_ChArUco", img)
        cv2.waitKey(500)
        images.append(img)                                  #In Array ablegen
        utils.saveImagesToDirectory(counter,img,directory1)
        counter += 1
        print("Captured")
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow("Calib_ChArUco")

allCorners,allIds,imsize = read_chessboards(images)

print(allCorners)
print(allIds)
print(imsize)


