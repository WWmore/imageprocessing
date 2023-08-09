
"__author__ = Hui Wang"

""" Based on the file extract_patches.py result, which return 8 rectangular patches
using stitching algorithm to get a panorama
"""

import numpy as np
import cv2
import glob
from stitching import AffineStitcher
#-----------------------------------------------------------------------------------


# print(AffineStitcher.AFFINE_DEFAULTS)
# Comparison:
# print(Stitcher.DEFAULT_SETTINGS)
# print(AffineStitcher.DEFAULT_SETTINGS).

if 1:
    path = './ball_photos'
else:
    path = './drill_photos'

names = [file for file in glob.glob(path+"/patch_8strip/*.png")]

settings = {# The whole plan should be considered
        "crop": False,
        # The matches confidences aren't that good
        "confidence_threshold": 0.5}    

stitcher = AffineStitcher(**settings)

panorama = stitcher.stitch(names)

"!NOTE: for both ways below, different time, the running result maybe different"
if 1:
    #cv2.imwrite(path + "/panorama_8strips.png", panorama)  
    cv2.imshow("Panorama" , panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.show()


