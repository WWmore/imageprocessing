"""
Affine transform
=================

Warping and affine transforms of images.
"""

from matplotlib import pyplot as plt

# import math
import numpy as np
from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform

from skimage import transform
text = data.text()
print(text.shape) ##(172, 448)

tform = transform.SimilarityTransform(scale=1, rotation=np.pi/4,
                                      translation=(text.shape[0]/2, -100))

rotated = transform.warp(text, tform)
print(rotated.shape) ##(172, 448)

back_rotated = transform.warp(rotated, tform.inverse)
print(back_rotated.shape, back_rotated) ##(172, 448)

"""
[[0.23685615 0.302831   0.32010785 ... 0.         0.         0.        ]
 [0.31140428 0.40076051 0.40856315 ... 0.         0.         0.        ]
 [0.34314991 0.42474517 0.42643042 ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]]
"""

fig, ax = plt.subplots(nrows=3)

ax[0].imshow(text, cmap=plt.cm.gray)
ax[1].imshow(rotated, cmap=plt.cm.gray)
ax[2].imshow(back_rotated, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()





text = data.text()

src = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
dst = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

tform3 = transform.ProjectiveTransform()
tform3.estimate(src, dst)
warped = transform.warp(text, tform3, output_shape=(50, 300))


fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

ax[0].imshow(text, cmap=plt.cm.gray)
ax[0].plot(dst[:, 0], dst[:, 1], '.r') ##plotting points on image
ax[1].imshow(warped, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()