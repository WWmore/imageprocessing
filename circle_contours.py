

import numpy as np
import cv2


def circle_Hough(subimg):
    """ in a computing way
    another way: Circle detection (HoughCircles, refer to Learning OpenCV book)
    """
    gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    bimg = cv2.medianBlur(gray_img, 5)
    cimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(bimg,cv2.HOUGH_GRADIENT,1,120, param1=100,param2=30,minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    def circle_detection(circles):
        "detect all the circles, plot circles + centers"
        print(circles, circles[0], circles[0,0])
        for i in circles[0,:]:
            # draw the outer circle
            print(i, i[0], i[1], i[2])
            cv2.circle(subimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(subimg,(i[0],i[1]),2,(0,0,255),3)

    centroid = np.array([int(subimg.shape[0]/2), int(subimg.shape[1]/2)])
    dist = max(subimg.shape[0], subimg.shape[1])
    index = 0
    for i, c in enumerate(circles[0,:]):
        d = np.linalg.norm(np.array([c[0], c[1]])-centroid)
        # print(i, d)
        if d < dist:
            dist = d
            index = i

    circle = circles[0][index]
    print(index, circle)
    center, radius = circle[:2], circle[2]

    # draw the outer circle
    cv2.circle(subimg,center,radius,(0,0,255),2)
    # draw the center of the circle
    cv2.circle(subimg,center,2,(0,0,255),10)
    # print(circle[0],circle[1])

    cv2.imshow("Contours",subimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def circle_moment(subimg): ## no use
    """https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208
    We can find the centroid of an image or calculate the area of a boundary field with the help of the notion called image moment. 
    What does a moment mean here? 
    The word ‘moment’ is a short period of time in common usage. 
    But in physics terminology, a moment is the product of the distance and another physical quantity meaning how a physical quantity is distributed or located. 
    So in computer vision, Image moment is how image pixel intensities are distributed according to their location. 
    It’s a weighted average of image pixel intensities and we can get the centroid or spatial information from the image moment.
    """
    gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    ## convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_img,127,255,0)
    ## calculate moments of binary image
    M = cv2.moments(thresh)

    ## calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    ## put text and highlight the center
    cv2.circle(subimg, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(subimg, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ## display the image
    cv2.imshow("Image", subimg)
    cv2.waitKey()
    cv2.destroyAllWindows()


def circle_contour(subimg): ## no use
    gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    ## another way: build Gaussian threshold (refer 03_06.py)
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow("Binary", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = subimg.copy()
    index = -1
    thickness = 4
    color = (255, 0, 255)
    cv2.drawContours(img2, contours, index, color, thickness)
    cv2.imshow("Contours",img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def circle_match(path, subimg): ## no use
    gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    ## another way: mathTemplate (refer 04_03.py), collapse to work
    template = cv2.imread(path+'ball_template.jpg',0)

    cv2.imshow("Frame",subimg)
    cv2.imshow("Template",template)

    result = cv2.matchTemplate(subimg, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(max_val,max_loc)
    cv2.circle(result,max_loc, 15,255,2)
    cv2.imshow("Matching",result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()