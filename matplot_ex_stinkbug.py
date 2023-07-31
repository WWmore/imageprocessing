
## https://matplotlib.org/stable/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


###read and plot 2D image: 
img = np.asarray(Image.open(r'C:\Users\WANGH0M\Desktop\imageProcess\stinkbug.png'))
imgplot = plt.imshow(img)


"""
Applying pseudocolor schemes to image plots
Pseudocolor can be a useful tool for enhancing contrast and visualizing your data more easily. 
This is especially useful when making presentations of your data using projectors - their contrast is typically quite poor.

Pseudocolor is only relevant to single-channel, grayscale, luminosity images. 
We currently have an RGB image. Since R, G, and B are all similar (see for yourself above or in your data), 
we can just pick one channel of our data using array slicing
"""

##Pseudocolor of it:
lum_img = img[:, :, 0]
# plt.imshow(lum_img)

## make a hot image:
plt.imshow(lum_img, cmap="hot")

## use another colormap + colorbar:
# imgplot = plt.imshow(lum_img)
# imgplot.set_cmap('nipy_spectral')
plt.colorbar()


##set limits of colormap: 
# plt.imshow(lum_img, clim=(0, 175))






# ### read and resize the image:
# img = Image.open(r'C:\Users\WANGH0M\Desktop\imageProcess\stinkbug.png')
# img.thumbnail((64, 64))  # resizes image in-place
# imgplot = plt.imshow(img)

# ### interpolate the image:
# imgplot = plt.imshow(img, interpolation="bilinear")
# imgplot = plt.imshow(img, interpolation="bicubic")