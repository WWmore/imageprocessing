
"__author__ = Hui Wang"

""" Based on the file extract_patches.py result, which return 8 rectangular patches
using stitching algorithm to get a panorama
"""

import numpy as np
import cv2
import glob
from stitching import AffineStitcher
#-----------------------------------------------------------------------------------

paths = ['./photos_ball/top','./photos_ball/front', './photos_drill/front', './photos_apple', './photos_cup']
path = paths[1] ### need to choose the path name

names = [file for file in glob.glob(path+"/rectangle/*.png")]

settings = {# The whole plan should be considered
        "crop": False, ##need to choose
        # The matches confidences aren't that good
        "confidence_threshold": 0.4} ##Hui: number need be carefully chosen

stitcher = AffineStitcher(**settings)

panorama = stitcher.stitch(names)

"!NOTE: for both ways below, different time, the running result maybe different"
if 1:
    cv2.imwrite(path + "/panorama.png", panorama)  
    cv2.imshow("Panorama" , panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.show()


