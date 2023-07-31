
import skimage

from skimage import data
camera = data.camera()


##print(type(camera))
##<type 'numpy.ndarray'>
##print(camera.shape)

coins = data.coins()
from skimage import filters
threshold_value = filters.threshold_otsu(coins)


import os
filename = os.path.join(skimage.data_dir, 'moon.png')
from skimage import io
moon = io.imread(filename)


import os
from natsort import natsorted, ns
from skimage import io
list_files = os.listdir('.')
##list_files
##['01.png', '010.png', '0101.png', '0190.png', '02.png']
list_files = natsorted(list_files)
##list_files
##['01.png', '02.png', '010.png', '0101.png', '0190.png']
image_list = []
for filename in list_files:
   image_list.append(io.imread(filename))


>>> camera.shape
(512, 512)
>>> camera.size
262144

>>> camera.min(), camera.max()
(0, 255)
>>> camera.mean()
118.31400299072266


>>> # Get the value of the pixel at the 10th row and 20th column
>>> camera[10, 20]
153
>>> # Set to black the pixel at the 3rd row and 10th column
>>> camera[3, 10] = 0



>>> # Set the first ten lines to "black" (0)
>>> camera[:10] = 0

>>> mask = camera < 87
>>> # Set to "white" (255) the pixels where mask is True
>>> camera[mask] = 255


>>> inds_r = np.arange(len(camera))
>>> inds_c = 4 * inds_r % len(camera)
>>> camera[inds_r, inds_c] = 0



>>> nrows, ncols = camera.shape
>>> row, col = np.ogrid[:nrows, :ncols]
>>> cnt_row, cnt_col = nrows / 2, ncols / 2
>>> outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
...                    (nrows / 2)**2)
>>> camera[outer_disk_mask] = 0


>>> lower_half = row > cnt_row
>>> lower_half_disk = np.logical_and(lower_half, outer_disk_mask)
>>> camera = data.camera()
>>> camera[lower_half_disk] = 0