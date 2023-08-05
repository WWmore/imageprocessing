
"""https://learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
Warp one triangle to another using OpenCV
"""

import numpy as np
import cv2
import sys

path = './else/robot.png'

# Read input image and convert to float
img1 = cv2.imread(path)

cv2.imshow("Input" , img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Output image is set to white
img2 = 255 * np.ones(img1.shape, dtype = img1.dtype)
 
# Define input and output triangles
"https://www.programminghead.com/Projects/find-coordinates-of-image-online.html"
tri1 = np.float32([[[790,710], [104,436], [633,349]]])
tri2 = np.float32([[[400,200], [160,270], [400,400]]])


# Find bounding box. 
r1 = cv2.boundingRect(tri1) 
r2 = cv2.boundingRect(tri2)

# Offset points by left top corner of the respective rectangles
tri1Cropped = []
tri2Cropped = []
     
for i in range(0, 3):
    tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
    tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))
 
# Crop input image
img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

cv2.imshow("Cropped" , img1Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Given a pair of triangles, find the affine transform.
warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )


# Apply the Affine Transform just found to the src image
img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )


# Get mask by filling triangle
mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)
 
img2Cropped = img2Cropped * mask
     
# Copy triangular region of the rectangular patch to the output image
img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped


cv2.imshow("Output" , img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
