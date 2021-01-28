import numpy as np
import cv2
import os

from cv2 import aruco
import pickle
import glob


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
ids_all = [] # Aruco ids corresponding to corners discovered
imgpoints = [] # 2d points in image plane.
image_size = None # Determined at runtime

directory1 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher"
directory2 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Nachher"


print(os.getcwd())
print('Path Exists ?')

print(os.path.exists(directory1))
print(os.path.exists(directory2))



counter = 0
images = []

def saveImagesToDirectory(counter,img,directory):
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("ERROR: Directory not Found")
    filename = 'savedImage' + str(counter) + '.jpg'
    cv2.imwrite(filename, img)  # in Ordner Speichern
    print(counter)

def undistortFunction(img,mtx,dist):
    succsess, image = cap.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)
    return dst

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=CHARUCO_BOARD,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

#reads in Calib Images
while True:
    succsess, img = cap.read()
    cv2.imshow("Calib_ChArUco", img)

    if cv2.waitKey(1) & 0xff == ord('x'):
        cv2.putText(img, "Captured", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
        cv2.imshow("Calib_ChArUco", img)
        cv2.waitKey(500)
        images.append(img)                                  #In Array ablegen
        saveImagesToDirectory(counter,img,directory1)
        counter += 1
        print("Captured")
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow("Calib_ChArUco")

#shows Images
for frame in images:            #Show Images
    cv2.imshow("All_Captured",frame)
    cv2.waitKey(200)
cv2.destroyWindow("All_Captured")

#findCorners
counter2 = 0
for img in images:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        image=gray,
        dictionary=ARUCO_DICT)
    print("Rejected Corners  ",len(rejectedImgPoints))
    # If found, add object points, image points (after refining them)
    if True:
        print("        Corners Found")
        img = aruco.drawDetectedMarkers(
            image=img,
            corners=corners)

        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

        if response > 20:
            print('response>20')
            objpoints.append(charuco_corners)
            ids_all.append(charuco_ids)
        if charuco_corners is None:
            print('Corners = NONE')
        if charuco_ids is None:
            print('IDs = NONE')
        # Draw and display the corners

        img = aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)
        saveImagesToDirectory(counter2,img,directory2)
        counter2 += 1
        cv2.imshow('img',img)
        cv2.waitKey(2000)
        if not image_size:
            image_size = gray.shape[::-1]
    else:
        print("         No Corners Found")


if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure
    exit()
# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size



calibration, mtx, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=objpoints,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)


#ret, mtx, distCoeffs, rotation_vectors, translation_vectors = calibrate_camera(allCorners=corners,allIds=ids_all,imsize=image_size)
print('Matrix:')
print(mtx)
print('Dist:')
print(distCoeffs)

#Wait
print("Take picture to undistort")
while True:
    succsess, img = cap.read()
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    if cv2.waitKey(1) & 0xff == ord('x'):
        succsess,image = cap.read()
        cv2.imshow("Distorted",image)
        saveImagesToDirectory("_distorted",image, "C:\\Users\\Lars\\Desktop\\TestBilder")
        undist = undistortFunction(image,mtx,dist)
        cv2.imshow("Undistorted",image)
        saveImagesToDirectory("_undistorted",undist, "C:\\Users\\Lars\\Desktop\\TestBilder")
        cv2.waitKey(2000)
    cv2.waitKey(1)

cv2.destroyAllWindows()
print('LiveView')

while True:
    succsess, img = cap.read()
    cv2.imshow("DistortedLive", img)
    undist = undistortFunction(img, mtx, dist)
    cv2.imshow("UndistortedLive", undist)
    cv2.waitKey(1)

cv2.destroyAllWindows()