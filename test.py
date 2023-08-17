
"""
https://stackoverflow.com/questions/70506234/cv2-error-opencv4-5-4-1-error-5bad-argument-in-function-warpperspect
"""

import numpy as np
import cv2


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def auto_scan_image():
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    # document.jpg ~ docuemnt7.jpg
    path = './photos_apple'
    image = cv2.imread(path+'/1I6A4439.jpg')
    orig = image.copy()
    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # apply the four point transform to obtain a top-down
    # view of the original image
    rect = order_points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth = int(max([w1, w2]))
    maxHeight = int(max([h1, h2]))

    dst = np.float32([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]])

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    if 0:
        "NOTE: Hui added rescale, but the image is still out of the windows layout"
        r = 800.0 / warped.shape[0]
        dim = (int(warped.shape[1] * r), 800)
        warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)

        # show the original and scanned images
        print("STEP 3: Apply perspective transform")
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        "NOTE: totally gray image"
        import matplotlib.pyplot as plt 
        plt.imshow(warped)

if __name__ == '__main__':
    auto_scan_image()