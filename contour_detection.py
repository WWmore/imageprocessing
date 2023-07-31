import cv2
import numpy as np

"""
After the initial imports, we load the image, and then apply a binary threshold on
a grayscale version of the original image. By doing this, we operate all find-contour
calculations on a grayscale copy, but we draw on the original so that we can utilize
color information.
"""

path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\hammer.jpg'
# path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\gfg.png'
img = cv2.pyrDown(cv2.imread(path, cv2.IMREAD_UNCHANGED))

cv2.imshow("Original", img)
cv2.waitKey()
cv2.destroyAllWindows()

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
# print(ret, thresh)
contours, hier = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # find bounding box coordinates
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int0(box)
    # draw contours
    cv2.drawContours(img, [box], 0, (0,0, 255), 3)

    # calculate center and radius of minimum enclosing circle
    (x,y),radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x),int(y))
    radius = int(radius)
    # draw the circle
    circle = cv2.circle(img,center,radius,(0,255,0),2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow("Contours", img)
cv2.waitKey()
cv2.destroyAllWindows()

# """conversion of contour information to the (x, y) coordinates, 
# plus the height and width of the rectangle
# """
# x,y,w,h = cv2.boundingRect(img)
# cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# "calculate the minimum area enclosing the subject"
# rect = cv2.minAreaRect(c)
# box = cv2.boxPoints(rect)
# box = np.int0(box)

# cv2.drawContours(img, [box], 0, (0,0, 255), 3)

# "The last bounding contour we're going to examine is the minimum enclosing circle"
# (x,y),radius = cv2.minEnclosingCircle(c)
# center = (int(x),int(y))
# radius = int(radius) ##converting all these values to integers
# img = cv2.circle(img,center,radius,(0,255,0),2)

# cv2.imshow("Circles", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

def sort_contours(img, contours):
    # Sort the contours 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    # Draw the contour 
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx = -1, 
                            color = (255, 0, 0), thickness = 2)
    cv2.imshow("Copy Image", img_copy)
    cv2.waitKey()
    cv2.destroyAllWindows()