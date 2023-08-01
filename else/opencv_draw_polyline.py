import cv2
import numpy as np

# path

#path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\gfg.png'
 
# Reading an image in default
image = cv2.imread('gfg.png')
 
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

# Window name in which image is
# displayed
window_name = 'Image'
 
# Polygon corner points coordinates
pts = np.array([[25, 70], [25, 160],
                [110, 200], [200, 160],
                [200, 70], [110, 20]],
               np.int32)
 
pts = pts.reshape((-1, 1, 2))
 
isClosed = True
 
# Blue color in BGR
color = (255, 0, 0)
 
# Line thickness of 2 px
thickness = 2
 
# Using cv2.polylines() method
# Draw a Blue polygon with
# thickness of 1 px
image = cv2.polylines(image, [pts],
                      isClosed, color, thickness)
 
# Displaying the image
  
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()




# Python program to explain
# cv2.polylines() method

# Reading an image in default
# mode
image = cv2.imread('gfg.png')

# Window name in which image is
# displayed
window_name = 'Image'

# Polygon corner points coordinates
pts = np.array([[25, 70], [25, 145],
				[75, 190], [150, 190],
				[200, 145], [200, 70],
				[150, 25], [75, 25]],
			np.int32)

pts = pts.reshape((-1, 1, 2))

isClosed = True

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 8 px
thickness = 8

# Using cv2.polylines() method
# Draw a Green polygon with
# thickness of 1 px
image = cv2.polylines(image, [pts],
					isClosed, color,
					thickness)

# Displaying the image

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
