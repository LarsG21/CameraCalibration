import numpy as np
import cv2
import os
from scipy.io import savemat

cap = cv2.VideoCapture(1)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

directory1 = "C:\\Users\\larsg\\OneDrive\\Desktop\\TestBilder\\Vorher2"
directory2 = "C:\\Users\\larsg\\OneDrive\\Desktop\\TestBilder\\Nachher2"


print(os.getcwd())
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
    #x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png',dst)
    return dst

#reads in Calib Images
while True:
    succsess, img = cap.read()
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('x'):
        cv2.putText(img, "Captured", (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
        cv2.imshow("Image", img)
        cv2.waitKey(500)
        images.append(img)                                  #In Array ablegen
        saveImagesToDirectory(counter,img,directory1)
        counter += 1
        print("Captured")
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyWindow("Image")

#shows Images
for frame in images:            #Show Images
    cv2.imshow("Test",frame)
    cv2.waitKey(200)
cv2.destroyWindow("Test")

#findCorners
counter2 = 0
for img in images:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    print("FindCorners")
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("        Corners Found")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        saveImagesToDirectory(counter2,img,directory2)
        counter2 += 1
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        print("         No Corners Found")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


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
        saveImagesToDirectory("_distorted",image,"C:\\Users\\larsg\\OneDrive\\Desktop\\TestBilder")
        undist = undistortFunction(image,mtx,dist)
        cv2.imshow("Undistorted",image)
        saveImagesToDirectory("_undistorted",undist,"C:\\Users\\larsg\\OneDrive\\Desktop\\TestBilder")
        cv2.waitKey(2000)
    cv2.waitKey(1)
cv2.destroyAllWindows()