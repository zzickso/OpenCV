import sys
import numpy as np
import cv2


# 영상 불러오기
src1 = cv2.imread('graf1.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('graf3.png', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
feature1 = cv2.KAZE_create()
feature2 = cv2.AKAZE_create()
feature3 = cv2.ORB_create()

# 특징점 검출
kp1 = feature1.detect(src1)
kp2 = feature1.detect(src2)
kp3 = feature2.detect(src1)
kp4 = feature2.detect(src2)
kp5 = feature3.detect(src1)
kp6 = feature3.detect(src2)


print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of kp3:', len(kp3))
print('# of kp4:', len(kp4))
print('# of kp5:', len(kp5))
print('# of kp6:', len(kp6))

# 검출된 특징점 출력 영상 생성
dst1 = cv2.drawKeypoints(src1, kp1, None,) # flags 지운거
dst2 = cv2.drawKeypoints(src2, kp2, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 방향성, 원 크기 표시
dst3 = cv2.drawKeypoints(src1, kp3, None,) # flags 지운거
dst4 = cv2.drawKeypoints(src2, kp4, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 방향성, 원 크기 표시
dst5 = cv2.drawKeypoints(src1, kp5, None,) # flags 지운거
dst6 = cv2.drawKeypoints(src2, kp6, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 방향성, 원 크기 표시

cv2.imshow('KAZE1', dst1)
cv2.imshow('KAZE2', dst2)
cv2.imshow('AKAZE1', dst3)
cv2.imshow('AKAZE2', dst4)
cv2.imshow('ORB1', dst5)
cv2.imshow('ORB2', dst6)
cv2.waitKey()
cv2.destroyAllWindows()
