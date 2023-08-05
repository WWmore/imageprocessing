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

def panorama1(folder_path):
    "https://www.makeuseof.com/create-panoramas-with-python-and-opencv/"
    import os
    def load_images(folder_path):
        # Load images from a folder and resize them.
        images = []
        for filename in os.listdir(folder_path):
            # Check if file is an image file
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load the image using OpenCV and resize it
                image = cv2.imread(os.path.join(folder_path, filename))
                images.append(image)
        return images
    
    def resize_images(images, width, height):
        resized_images = []
        for image in images:
            resized_image = cv2.resize(image, (width, height))
            resized_images.append(resized_image)
        return resized_images
    
    def stitch_images(images):
        stitcher = cv2.Stitcher.create()
        (status, stitched_image) = stitcher.stitch(images)
        if status == cv2.STITCHER_OK:
            return stitched_image
        else:
            return None
    
    def crop_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    
    def preview_and_save_image(image, folder_path, folder_name):
        # Display the stitched image
        cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Stitched Image', image)
        cv2.waitKey(0)

        # # Save the stitched image
        # output_filename = os.path.join(folder_path, folder_name + '_panorama.jpg')
        # cv2.imwrite(output_filename, image)
        # print('Stitched image saved for folder:', folder_name)
    
    def stitch_folder(folder_path, width=1800, height=1800):
        # Stitch all images in a folder and save the result.
        # Load the images from the folder
        images = load_images(folder_path)

        # Check if there are at least two images in the folder
        if len(images) < 2:
            print('Not enough images in folder:', folder_path)
            return

        # Resize the images
        resized_images = resize_images(images, width, height)

        # Stitch the images
        stitched_image = stitch_images(resized_images)
        if stitched_image is None:
            print('Stitching failed for folder:', folder_path)
            return

        # Crop the stitched image
        cropped_image = crop_image(stitched_image)

        # Preview and save the stitched image
        folder_name = os.path.basename(folder_path)
        preview_and_save_image(cropped_image, folder_path, folder_name)

    return stitch_folder(folder_path)


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

#stitch_images(images)


#panorama1('./else/tests')
panorama1('./ball_photos/patch')