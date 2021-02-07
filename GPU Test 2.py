import cv2

img = cv2.imread('CalibrationImages/Run1/Calib00019.TIF', cv2.IMREAD_GRAYSCALE)
img = cv2.Umat(img)
while True:
    
    for i in range(3):
        img = cv2.GaussianBlur(img, (3, 3), 1)
    cv2.imshow("image",img)
