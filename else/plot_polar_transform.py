import numpy as np
import matplotlib.pyplot as plt

import os
from skimage import data, io
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

radius = 705
angle = 35
# image = data.retina()

### Read only 1 image:
path = r'C:\Users\WANGH0M\Desktop\imageProcess'
filename = os.path.join(path, 'dots2.png')
image = io.imread(filename)
# imgplot = plt.imshow(image)


image = img_as_float(image)
rotated = rotate(image, angle)
image_polar = warp_polar(image, radius=radius, channel_axis=-1)
rotated_polar = warp_polar(rotated, radius=radius, channel_axis=-1)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Rotated")
ax[1].imshow(rotated)
ax[2].set_title("Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Polar-Transformed Rotated")
ax[3].imshow(rotated_polar)
plt.show()

shifts, error, phasediff = phase_cross_correlation(image_polar,
                                                   rotated_polar,
                                                   normalization=None)
print(f'Expected value for counterclockwise rotation in degrees: '
      f'{angle}')
print(f'Recovered value for counterclockwise rotation: '
      f'{shifts[0]}')