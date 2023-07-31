import numpy as np
import cv2

path = r'C:\Users\WANGH0M\Desktop\opencv\Ex_Files_OpenCV_Python_Developers\Ch02\02_01 Begin'
img = cv2.imread(path +'opencv-logo.png')
print(img)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)

cv2.imwrite('output.jpg', img)