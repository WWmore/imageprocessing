
"__author__ = Hui Wang"

""" Based on the file extract_patches.py result, which return 8 rectangular patches
using stitching algorithm to get a panorama
"""

import numpy as np
import cv2
import glob
from stitching import AffineStitcher
#-----------------------------------------------------------------------------------

paths = ['./photos_ball/top','./photos_ball/front', 
             './photos_drill/top', './photos_drill/front','./photos_drill/every5',
             './photos_apple', './photos_cup']
path = paths[5] ### need to choose the path name

names = [file for file in glob.glob(path+"/rectangle/*.png")]

 ##Hui Note: below number need be carefully chosen, ussually from 0.4-0.6
 ##different run even at the same settings may get different panorams, need choose
settings = {"crop": False, ##need to choose
            # The matches confidences aren't that good
            "confidence_threshold": 0.5}

stitcher = AffineStitcher(**settings)

panorama = stitcher.stitch(names)

 ##Hui Note:  for both ways below, different time, the running result maybe different
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


