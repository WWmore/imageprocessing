"__author__ = Hui Wang"

""" Functions:
crop_image(img, pts)
"""

import numpy as np
import cv2
#---------------------------------------------------------------------------------

### Extract cropped patch:

def crop_image(img, pts):
    # pts = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])

    ## (1) Crop the bounding rectangular patch
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

    cv2.imshow("cropped" , cropped)
    cv2.imshow("white background" , dst2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
