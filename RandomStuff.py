import cv2
import utils

img = cv2.imread("Images/1.TIF")

croped = utils.cropImage(img)

cv2.imshow("Crop", croped)

cv2.waitKey(1000)