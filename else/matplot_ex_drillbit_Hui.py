import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


###read and plot 2D image: 
path = r'C:\Users\WANGH0M\Desktop\Drillbit\18_07_2022_14_44--0\0B4A9663.jpg'
img = np.asarray(Image.open(path))
imgplot = plt.imshow(img)

### read and resize the image:
# img = Image.open(path)
# img.thumbnail((64, 64))  # resizes image in-place
# imgplot = plt.imshow(img)

### interpolate the image:
# imgplot = plt.imshow(img)
# imgplot = plt.imshow(img, interpolation="bilinear")
# imgplot = plt.imshow(img, interpolation="bicubic")

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

## colorbar:
plt.colorbar()

##set limits of colormap: 
# plt.imshow(lum_img, clim=(0, 175))




# from matplotlib.colors import Normalize

# def normal_pdf(x, mean, var):
#     return np.exp(-(x - mean)**2 / (2*var))
# # Generate the space in which the blobs will live
# xmin, xmax, ymin, ymax = (0, 255, 0, 255)
# n_bins = 100
# xx = np.linspace(xmin, xmax, n_bins)
# yy = np.linspace(ymin, ymax, n_bins)

# # Generate the blobs. The range of the values is roughly -.0002 to .0002
# means_high = [20, 50]
# means_low = [50, 60]
# var = [150, 200]

# gauss_x_high = normal_pdf(xx, means_high[0], var[0])
# gauss_y_high = normal_pdf(yy, means_high[1], var[0])

# gauss_x_low = normal_pdf(xx, means_low[0], var[1])
# gauss_y_low = normal_pdf(yy, means_low[1], var[1])

# weights = (np.outer(gauss_y_high, gauss_x_high)
#            - np.outer(gauss_y_low, gauss_x_low))

# # # We'll also create a grey background into which the pixels will fade
# greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)

# # # First we'll plot these blobs using ``imshow`` without transparency.
# vmax = np.abs(weights).max()
# imshow_kwargs = {
#     'vmax': vmax,
#     'vmin': -vmax,
#     'cmap': 'RdYlBu',
#     'extent': (xmin, xmax, ymin, ymax),
# }
# # Create an alpha channel based on weight values
# # Any value whose absolute value is > .0001 will have zero transparency
# alphas = Normalize(0, .3, clip=True)(np.abs(weights))
# alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4

# # Create the figure and image
# # Note that the absolute values may be slightly different
# fig, ax = plt.subplots()
# ax.imshow(greys)
# ax.imshow(weights, alpha=alphas, **imshow_kwargs)

# # Add contour lines to further highlight different levels.
# ax.contour(weights[::-1], levels=[-.1, .1], colors='k', linestyles='-')
# ax.set_axis_off()
# plt.show()

# ax.contour(weights[::-1], levels=[-.0001, .0001], colors='k', linestyles='-')
# ax.set_axis_off()
# plt.show()



