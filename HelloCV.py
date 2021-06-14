import cv2
import sys

print('Hello OpenCV', cv2.__version__)

img = cv2.imread('cat.bmp')

if img is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('catzz.png', img)
while True:
    if cv2.waitKey() == 27:
        break
cv2.destroyAllWindows()

