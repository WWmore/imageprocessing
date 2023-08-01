"""
1. Read drillbit photos
2. Extract diagonal photo stips
3. Stitch the strips
4. Detect circles / ellipses
"""

from timeit import default_timer as timer
start = timer()

import os
import skimage
from skimage import data, graph, io
from matplotlib import pyplot as plt
import numpy as np

### Read only 1 image:
# filename = os.path.join(skimage.data_dir, 'drillbit.png')
# img = io.imread(filename)
# print(img.shape, img.size)
# imgplot = plt.imshow(img)

### Read multi-images
# path = r'C:\Users\WANGH0M\Desktop\Drillbit\18_07_2022_14_44_sub'
path = r'C:\Users\WANGH0M\Desktop\ball_drillbit\ball\0'
# path = r'C:\Users\WANGH0M\Desktop\ball_drillbit\ball\0_sub'
filename = os.path.join(path, '*.jpg')
img = io.imread_collection(filename)

# img = skimage.img_as_float(img)

###Print + Plot: 
# print(img, img.files)
# io.imshow_collection(img) ##very slow to run, no result
# plt.axis('off')


def get_rectangular_from_polar():
    "NOUSE: polar-transformed imge (from polar to rectangular)"
    from skimage.transform import warp_polar, rotate, rescale, resize,rotate
    from skimage.util import img_as_float
    num_col = img[0].shape[1]
    # image = img_as_float(img[0][:,int(1/5*num_col):int(4/5*num_col),:])
    radius = 2000
    # image_polar = warp_polar(image, radius=radius, channel_axis=-1)

    # plt.imshow(image)
    # plt.imshow(image_polar)
    # plt.axis('off')
    # plt.show()

    im0 = img_as_float(img[0][:,int(1/5*num_col):int(4/5*num_col),:])
    im1 = img_as_float(img[1][:,int(1/5*num_col):int(4/5*num_col),:])
    im2 = img_as_float(img[2][:,int(1/5*num_col):int(4/5*num_col),:])
    
    image0 = warp_polar(im0, radius=radius, channel_axis=-1)
    image1 = warp_polar(im1, radius=radius, channel_axis=-1)
    image2 = warp_polar(im2, radius=radius, channel_axis=-1)
    shape = (image0.shape[0], image0.shape[1]//2)

    image0 = resize(image0, shape, anti_aliasing=True)
    image1 = resize(image1, shape, anti_aliasing=True)
    image2 = resize(image2, shape, anti_aliasing=True)

    # image0 = rotate(image0,90)
    # image1 = rotate(image1,90)
    # image2 = rotate(image2,90)
    fig, ax = plt.subplots(nrows=3)

    ax[0].imshow(image0)
    ax[1].imshow(image1)
    ax[2].imshow(image2)
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

get_rectangular_from_polar()

def get_stitch_1image():
    ### Extract same and uniform subpixels of each photo:
    num_row, num_col = img[0].shape[0], img[0].shape[1]
    num_col_cut11, num_col_cut12 = int(num_col*0.4), int(num_col*0.6)
    num_col_cut21, num_col_cut22 = int(num_col*0.3), int(num_col*0.7)
    delta1 = num_col_cut12 - num_col_cut11
    delta2 = num_col_cut22 - num_col_cut21

    if 0:
        "choose central regular slicing-patches"
        stitch_img = img[0][:, num_col_cut11:num_col_cut12, :]
        print(stitch_img.size)
        for ig in img[1:]:
            sub = ig[:, num_col_cut11:num_col_cut12, :]
            # stitch_img = np.concatenate((stitch_img, sub),axis=1)
            stitch_img = np.hstack((stitch_img, sub)) ##faster
    else:
        """ extract / choose diagonal regular slicing-patches:
        suppose diagonal tangent angle is k, then
            left straight line: y = -k*x + num_row
            right straight line: y = -k*(x-delta) + num_row
            default k=1, then linearly chose the inters of matrix[row,col]
        """
        k = 1 ##tile angle
        def get_sub_diagonal_strip(image):
            sub_img = []
            for i in range(num_row):
                i_row = - k * i + num_row
                sub_img.append(image[i, i_row : i_row + delta1, :])
            return np.array(sub_img)

        sub_img0 = get_sub_diagonal_strip(img[0])
        stitch_img = sub_img0
        for ig in img[1:]:
            sub_img = get_sub_diagonal_strip(ig)
            stitch_img = np.hstack((stitch_img, sub_img))



    ### Print + Plot the image:
    # print(stitch_img.shape,stitch_img.size,num_row,num_img*(num_col_cut2-num_col_cut1))
    plt.imshow(stitch_img)
    plt.axis('off')
    plt.show()


### Print + Plot the image:
from skimage import transform

def get_1trapezoid_patch(image):
    ### Extract same and uniform subpixels of each photo:
    num_row, num_col = img[0].shape[0], img[0].shape[1]
    num_col_cut11, num_col_cut12 = int(num_col*0.4), int(num_col*0.6)
    num_col_cut21, num_col_cut22 = int(num_col*0.3), int(num_col*0.7)
    delta1 = num_col_cut12 - num_col_cut11
    delta2 = num_col_cut22 - num_col_cut21

    a,b = num_row, delta1
    src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
    dst = np.array([[num_col_cut11,0],[num_col_cut21, num_row],
                    [num_col_cut22, num_row,], [num_col_cut12, 0]])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(image, tform3, output_shape=(a, b))

    fig, ax = plt.subplots(nrows=2) #figsize=(8, 3))

    ax[0].imshow(image)
    ax[0].plot(dst[:, 0], dst[:, 1], '.r') ##plotting points on image
    ax[1].imshow(warped)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()


def get_trapezoid_patches(images, theta, distance, width):
    ### Extract same and uniform subpixels of each photo:
    num = len(images)
    # print(num)
    r = distance * np.sin(theta)
    delta1 = 2*(r-width/2*np.cos(theta))*np.tan(np.pi/num)
    delta2 = 2*(r+width/2*np.cos(theta))*np.tan(np.pi/num)
    print(delta1,delta2)

    ratio11 = (width-delta1)/width/2
    ratio12 = (width+delta1)/width/2
    ratio21 = (width-delta2)/width/2
    ratio22 = (width+delta2)/width/2   
    print(ratio11,ratio12,ratio21,ratio22)

    num_row, num_col = images[0].shape[0], images[0].shape[1]
    num_col_cut11, num_col_cut12 = int(num_col*ratio11), int(num_col*ratio12)
    num_col_cut21, num_col_cut22 = int(num_col*ratio21), int(num_col*ratio22)
    delta1 = num_col_cut12 - num_col_cut11
    delta2 = num_col_cut22 - num_col_cut21
    a,b = num_row, delta1

    def get_1patch(image, a,b):
        """The different transformations in skimage.transform have a estimate method 
        in order to estimate the parameters of the transformation from two sets of points 
        (the source and the destination)
        """
        src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
        dst = np.array([[num_col_cut11,0],[num_col_cut21, num_row],
                        [num_col_cut22, num_row,], [num_col_cut12, 0]])
        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = transform.warp(image, tform3, output_shape=(a, b))
        return warped

    def get_tform(a,b):
        src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
        dst = np.array([[num_col_cut11,0],[num_col_cut21, num_row],
                        [num_col_cut22, num_row,], [num_col_cut12, 0]])
        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        return tform3
    
    tform = get_tform(a,b)
    
    sub_img0 = transform.warp(images[0], tform, output_shape=(a, b))
    stitch_img = sub_img0
    for i, ig in enumerate(images[1:8]):
        sub_img = transform.warp(ig, tform, output_shape=(a, b))
        stitch_img = np.hstack((stitch_img, sub_img))
        print(i)
    ### Print + Plot the image:
    # print(stitch_img.shape,stitch_img.size,num_row,num_img*(num_col_cut2-num_col_cut1))
    plt.imshow(stitch_img)
    plt.axis('off')
    plt.show()

# theta1 = np.pi - 150.4/180*np.pi
# get_trapezoid_patches(img, theta1, distance=5, width=1)

end = timer()
print('time:',end-start)