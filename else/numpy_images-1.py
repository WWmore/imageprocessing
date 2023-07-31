# Using a 2D mask on a 2D color image
import matplotlib.pyplot as plt
from skimage import data
cat = data.chelsea()
plt.imshow(cat)

reddish = cat[:, :, 0] > 160
cat[reddish] = [0, 255, 0]
plt.imshow(cat)
