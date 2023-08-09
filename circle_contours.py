"__author__ = Hui Wang"

""" Functions:
circle_canny(img, is_crop=False, is_scale=False, is_RGB=False)
circle_Hough(subimg)
circle_blob(subimg)
circle_moment(subimg)
circle_contour(subimg)
circle_match(path, subimg)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 
#-------------------------------------------------------------------------------------

def circle_canny(img, is_crop=False, is_scale=False, is_RGB=False):
    """Use the Canny edge detection algorithm to detect edges in the image,
    then search for the left boundary-edge and right boundary-edge to rebuld the circle
    """
    #img = cv2.imread(name)

    if is_crop:
        num_row, num_col = img.shape[0], img.shape[1]
        img = img[:, int(num_col*0.2):int(num_col*0.8), :]

    if is_scale:
        # Define the scale percentage for resizing the image
        scale_percent = 15.625  # 100% means keeping the original size, you can change this value
        width = int(img.shape[1] * scale_percent / 100)  # Calculate the new width
        height = int(img.shape[0] * scale_percent / 100)  # Calculate the new height
        dim = (width, height)  # Create a tuple representing the new dimensions (width, height)

        # Resize the image using the calculated dimensions and interpolation method (INTER_AREA)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) ##is around (853, 1280)
    
    if is_RGB:
        # used for plt.imshow, otherwise cv2.imshow will turn to blue. Work inversely.
        # Convert the color channels from RGB to BGR format (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.imshow(img)

    # Canny Parameters:
    #   img: The input image on which edge detection will be performed.
    #   40: The lower threshold value. This value determines the intensity gradient below which edges are not considered.
    #   180: The upper threshold value. This value determines the intensity gradient above which edges are considered strong edges.
    #   apertureSize: The size of the Sobel kernel used for edge detection. It can be 3, 5, or 7. A larger value gives smoother edges.
    #   L2gradient: A Boolean flag to specify the gradient magnitude calculation method. If True, L2 norm (Euclidean distance) is used. If False, L1 norm (Manhattan distance) is used.
    # (40,180) is suitable for the image size around (853, 1280)
    # for drillbit, the number maybe (100, 280), but still not good
    edges = cv2.Canny(img, 40, 180,apertureSize=3, L2gradient=True)
    plt.imshow(edges)

    "Detecting the most left and right verticies of the sphere"
    # Initialize an empty list to store the coordinates that meet the condition
    coordinates = list()

    # Get the shape of the 'edges' array (assuming 'edges' is a 2D array or matrix)
    shape = edges.shape

    # Iterate over rows (y) first
    for x in range(shape[1]):
        # Iterate over columns (x)
        for y in range(shape[0]):
            # Check if the current row 'y' is between 400 and 500 (exclusive)
            if 400 < y < 500:
                # Check if the value of the element at position (y, x) in the 'edges' array is greater than 250
                if edges[y][x] > 250:
                    # If the conditions are met, add the coordinate (y, x) to the 'coordinates' list
                    coordinates.append((y, x))

    "Finding the circle parameter, and drawing the circle"
    # Calculate the diameter of the circle using the x-coordinates of the last and first elements in the 'coordinates' list
    diameter = coordinates[-1][1] - coordinates[0][1]

    # Calculate the radius of the circle by dividing the diameter by 2 and converting it to an integer
    radius = int(diameter / 2)

    # Calculate the x-coordinate of the center of the circle by taking the average of the x-coordinates of the last and first elements in the 'coordinates' list
    center_x = (coordinates[-1][1] + coordinates[0][1]) / 2

    # Calculate the y-coordinate of the center of the circle by taking the average of the y-coordinates of the last and first elements in the 'coordinates' list
    center_y = int((coordinates[-1][0] + coordinates[0][0]) / 2)

    # Create a tuple representing the center of the circle as (x, y)
    center = (int(center_x), int(center_y))

    # Set the color of the circle in BGR format (blue in this example, as (255, 0, 0))
    color = (255, 0, 0)

    # Create a copy of the original image to draw the circle on
    image = img.copy()

    # Draw the circle on the image using OpenCV's circle function
    # center: the center coordinates of the circle
    # radius: the radius of the circle
    # color: the color of the circle
    # thickness: the thickness of the circle outline (set to 2 in this example)
    cv2.circle(image, center, radius, color, thickness=2)

    # Display the image with the drawn circle using matplotlib
    #plt.imshow(image)

    return center, radius
#-------------------------------------------------------------------------------------


def circle_Hough(subimg):## no use
    """ in a computing way
    another way: Circle detection (HoughCircles, refer to Learning OpenCV book)
    """
    gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    bimg = cv2.medianBlur(gray_img, 5)
    # cimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2BGR)

    width, height = subimg.shape[0], subimg.shape[1]
    maxR = max(width//2, height//2)
    minR = min(width//5, height//5)
    circles = cv2.HoughCircles(bimg,cv2.HOUGH_GRADIENT,1.2,120, 
                               param1=100,param2=30,
                               minRadius=minR, maxRadius=maxR)
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

    # circle_detection(circles)
    # cv2.imshow("Contours",subimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    centroid = np.array([subimg.shape[0]//2, subimg.shape[1]//2])
    dist = np.sqrt((subimg.shape[0])**2 + (subimg.shape[1])**2)
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
    return center, radius


def circle_blob(subimg):## no use
    "https://learnopencv.com/blob-detection-using-opencv-python-c/"
    "note suitable for multi-small circles, since blob=ban dian in Chinese"

    # Read image
    ##im = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)

    def parameter():
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 1000
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.maxCircularity = 1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        params.maxConvexity = 1
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 1

        return params
    
    # Set up the detector with default parameters.
    params = parameter()
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    print(int(ver[0]))
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    
    if 1: 
        # Detect blobs.
        keypoints = detector.detect(subimg)
    else:
        # Apply Laplacian of Gaussian
        blobs_log = cv2.Laplacian(subimg, cv2.CV_64F)
        blobs_log = np.uint8(np.absolute(blobs_log))
        keypoints = detector.detect(blobs_log)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(subimg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    if True:
        plt.imshow("Keypoints", im_with_keypoints)
    else:
        cv2.imshow("Keypoints", im_with_keypoints)
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
    "from 03_06_compute_contour.py, 03_07_contour_centroid.py"
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

#--------------------------------------------------------------------------------


if __name__ == "__main__":
    def rescale(img):
        ## rescale the photo
        img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
        return img

    def crop(img):
        ## crop the photo
        num_row, num_col = img.shape[0], img.shape[1]
        subimg = img[:, int(num_col*0.2):int(num_col*0.8), :]
        return subimg


    if 1:
        name = './ball_photos/0B4A4656.jpg'
        is_crop=True
    else:
        name = './drill_photos/5.jpg'
        is_crop=False
    
    img = cv2.imread(name,1)

    #circle_blob(rescale(crop(img)))

    #circle_Hough(rescale(crop(img)))

    circle_canny(img, is_crop, True, True)