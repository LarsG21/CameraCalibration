import cv2
import os
import ContourUtils
from scipy.spatial import distance as dist

def saveImagesToDirectory(counter,img,directory):
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("ERROR: Directory not Found")
    filename = 'savedImage' + str(counter) + '.jpg'
    cv2.imwrite(filename, img)  # in Ordner Speichern
    print(counter)



def saveFileToDirectory(filename,filetype,file,directory):
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("ERROR: Directory not Found")
    name = filename + '.' + filetype
    cv2.imwrite(name, file)  # in Ordner Speichern
    print("Saved",name)



def undistortFunction(img,mtx,dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png',dst)
    return dst

def calculatePixelsPerMetric(img,reorderd,ArucoSize,draw = True):
    (tltrX, tltrY) = ContourUtils.midpoint(reorderd[0], reorderd[1])  # top left,top right
    (blbrX, blbrY) = ContourUtils.midpoint(reorderd[2], reorderd[3])  # bottom left, botto right
    (tlblX, tlblY) = ContourUtils.midpoint(reorderd[0], reorderd[2])
    (trbrX, trbrY) = ContourUtils.midpoint(reorderd[1], reorderd[3])
    if draw:
        cv2.circle(img, (int(tltrX), int(tltrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(blbrX), int(blbrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(tlblX), int(tlblY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(trbrX), int(trbrY)), 1, (255, 255, 0), 2)
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),  # Draws Lines in Center
                 (255, 0, 255), 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

    dY = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dX = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    pixelsPerMetric = (dX + dY) / 2 * (1 / ArucoSize)  # Calculates Pixels/Lenght Parameter
    dimA = dY / pixelsPerMetric
    dimB = dX / pixelsPerMetric  # Dimention of Marker
    if draw:
        cv2.putText(img, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        cv2.putText(img, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
    return pixelsPerMetric

def undistortPicture(cap,saveImages,meanMTX,meanDIST):
    print("Take picture to undistort")
    while True:
        succsess, img = cap.read()
        cv2.imshow("Calib_Chess", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        if cv2.waitKey(1) & 0xff == ord('x'):
            succsess,image = cap.read()
            cv2.imshow("Distorted",image)
            if saveImages:
                saveImagesToDirectory("_distorted",image, "C:\\Users\\Lars\\Desktop\\TestBilder")
            undist = undistortFunction(image,meanMTX,meanDIST)
            cv2.imshow("Undistorted",undist)
            if saveImages:
                saveImagesToDirectory("_undistorted",undist, "C:\\Users\\Lars\\Desktop\\TestBilder")
            cv2.waitKey(2000)
        cv2.waitKey(1)

