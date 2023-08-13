import cv2
import numpy as np
from rectangle_global_warp import global_warp
from rectangle_addseam import localwrap

def my_imfillholes(src):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) and len(hierarchy):
        for idx in range(len(contours)):
            if len(contours[idx]) < 100:
                cv2.drawContours(src, contours, idx, (0, 0, 0), cv2.FILLED, 8)


def mask_fg(rgbImg, thrs, mask):
    grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
    rows = rgbImg.shape[0]
    cols = rgbImg.shape[1]
    for i in range(rows):
      for j in range(cols):
        if grayImg[i, j] > thrs - 3:
          mask[i, j] = 1
        else:
          mask[i, j] = 0
    my_imfillholes(mask)
    for i in range(mask.shape[0]):
      mask[i, 0] = 1
      mask[i, mask.shape[1] - 1] = 1
    for i in range(mask.shape[1]):
      mask[0, i] = 1
      mask[mask.shape[0] - 1, i] = 1
    mask = cv2.filter2D(mask, -1, np.ones((7, 7), np.uint8))
    mask = cv2.filter2D(mask, -1, np.ones((2, 2), np.uint8))
    for i in range(rows):
      for j in range(cols):
        if mask[i, j] > 1:
          mask[i, j] = 1
        else:
          mask[i, j] = 0
    return mask



#------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ths = 254
    img = cv2.imread("./rectangle/1_input.jpg", 1)
    grayImg = None
    new_img = None
    outimg = None
    oriimg = None
    orimask = None
    col = img.shape[1]
    row = img.shape[0]
    s = col * row
    scale = np.sqrt(250000 / s)
    oriimg = cv2.resize(img, (int(col * scale), int(row * scale)), interpolation=cv2.INTER_NEAREST)
    
    # cv2.imshow("Original Image", oriimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    orimask = cv2.resize(mask, (int(col * scale), int(row * scale)), interpolation=cv2.INTER_NEAREST)
    disimg = np.zeros((oriimg.shape[0], oriimg.shape[1], 2), dtype=np.float32)
    orimask = mask_fg(oriimg, ths, orimask)
    oriimg = localwrap(oriimg, orimask, disimg)

    # cv2.imshow("LocalWrap Image", oriimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    outimg = global_warp(oriimg, disimg, orimask, outimg)
    outimg = cv2.resize(outimg, (col, row), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("final", outimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

