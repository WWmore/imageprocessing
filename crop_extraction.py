import cv2
import numpy as np

path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\gfg.png'
#path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\hammer.jpg'
 

def crop(path):
    """
    https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    Steps
    1. find region using the poly points
    2. create mask using the poly points
    3. do mask op to crop
    4. add white bg if needed
    """
    img = cv2.imread(path)
    pts = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])

    ## (1) Crop the bounding rectangular patch
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst


    cv2.imshow("croped.png", croped)
    cv2.imshow("mask.png", mask)
    cv2.imshow("dst.png", dst)
    cv2.imshow("dst2.png", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def crop2(path):
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    cv2.fillPoly(mask, pts, (255))

    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(pts) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    cv2.imshow("cropped" , cropped )
    cv2.imshow("same size" , res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

crop2(path)

def crop3(path):

    """For the colored background version use the code like this:
    """
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(img,img,mask = mask)

    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    im2 = np.full((res.shape[0], res.shape[1], 3), (0, 255, 0), dtype=np.uint8 ) # you can also use other colors or simply load another image of the same size
    maskInv = cv2.bitwise_not(mask)
    colorCrop = cv2.bitwise_or(im2,im2,mask = maskInv)
    finalIm = res + colorCrop
    cropped = finalIm[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    cv2.imshow("cropped" , cropped )
    cv2.imshow("same size" , res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()