"""
TODO:
1. read drillbit photos
"""


import skimage
import os
from skimage import data, segmentation, color
from skimage import graph
from matplotlib import pyplot as plt
import numpy as np

filename = os.path.join(skimage.data_dir, 'drillbit.png')
from skimage import io
img = io.imread(filename)

print(img.shape, img.size)
# imgplot = plt.imshow(img)

##Pseudocolor of it:
lum_img = img[:, :, 0]
## make a hot image:
plt.imshow(lum_img, cmap="hot")

## colorbar:
# plt.colorbar()

# ### circulize the image and highlight the cutters:
# camera = img ##data.camera()
# camera[:10] = 0
# mask = camera < 87
# camera[mask] = 255
# inds_x = np.arange(len(camera))
# inds_y = (4 * inds_x) % len(camera)
# camera[inds_x, inds_y] = 0

# l_x, l_y = camera.shape[0], camera.shape[1]
# X, Y = np.ogrid[:l_x, :l_y]
# outer_disk_mask = (X - l_x / 2)**2 + (Y - l_y / 2)**2 > (l_x / 2)**2
# camera[outer_disk_mask] = 0

# plt.figure(figsize=(4, 4))
# plt.imshow(camera, cmap='gray')
# plt.axis('off')
# plt.show()




# from skimage.util import view_as_blocks
# a = np.array([[1,5,9,13],
#               [2,6,10,14],
#               [3,7,11,15],
#               [4,8,12,16]])
# print(view_as_blocks(a, (2, 2)))

### extract subpixels of the photo:
num_row, num_col = img.shape[0], img.shape[1]
subimg = img[:, int(num_col/3):int(num_col/2), :]

### highlight the cutters with white color:
# subimg[:10] = 0
# mask = subimg < 87
# subimg[mask] = 255

### plotting the image:
# plt.figure(figsize=(4, 4))
plt.imshow(subimg, cmap='gray')
plt.axis('off')
plt.show()


# from skimage.transform import resize
# print(subimg.shape)
# subimg_resized = resize(subimg, (100, 10),anti_aliasing=True)
# plt.imshow(subimg_resized)
# print(subimg_resized.shape)