import cv2
import os


def saveImagesToDirectory(counter,img,directory):
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("ERROR: Directory not Found")
    filename = 'savedImage' + str(counter) + '.jpg'
    cv2.imwrite(filename, img)  # in Ordner Speichern
    print(counter)


def undistortFunction(img,mtx,dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)
    return dst