"""
The panorama stitching algorithm can be divided into four basic fundamental steps. These steps are as follows:

Detection of keypoints (points on the image) and extraction of local invariant descriptors (SIFT feature) from input images.
Finding matched descriptors between the input images.
Calculating the homography matrix using the RANSAC algorithm.
The homography matrix is then applied to the image to wrap and fit those images and merge them into one.

"""


import numpy as np
import cv2
import glob

path = './ball_photos/patch'

img1 = cv2.imread(path+'/1.png') 
img2 = cv2.imread(path+'/2.png') 

def match_two_images(img1, img2):
    # convert images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    # create feature matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # match descriptors of both images
    matches = bf.match(descriptors_1,descriptors_2)
    # sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    # draw first 50 matches
    matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

    # show the image
    cv2.imshow('image', matched_img)
    # save the image
    cv2.imwrite("matched_images.jpg", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#match_two_images(img1, img2)

def match_multi_patches(imgs): ##BUG: no use, no running
    "https://www.geeksforgeeks.org/opencv-panorama-stitching/"
    cv2.imshow('1',imgs[0])
    cv2.imshow('2',imgs[1])
    cv2.imshow('3',imgs[2])

    stitchy=cv2.Stitcher.create()
    (dummy,output)=stitchy.stitch(imgs)

    if dummy != cv2.STITCHER_OK:
    # checking if the stitching procedure is successful
    # .stitch() function returns a true value if stitching is
    # done successfully
        print("stitching ain't successful")
    else:
        print('Your Panorama is ready!!!')

    # final output
    cv2.imshow('final result',output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def stitch_images(images):
    "https://www.youtube.com/watch?v=Zs51cg4mb0k"
    imageStitcher = cv2.Stitcher.create()
    error, stitched_img = imageStitcher.stitch(images)

    if not error:
        cv2.imwrite('stitched output', stitched_img)
        cv2.imshow('stitched Img', stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('error happens')

if True:
    images = [cv2.imread(file) for file in glob.glob(path+"/*.png")]
else:
    image_path = glob.glob(path+"/*.png")
    images = []
    for image in image_path:
        img = cv2.imread(image)
        images.append(img)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

stitch_images(images)