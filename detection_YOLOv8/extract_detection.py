"__author__ = Hui Wang"

""" Given images, detect the closest boundary polyline of it
"""
import cv2
from yolo_segmentation import YOLOSegmentation
import glob
import numpy as np

from levelsets import Levelset

#--------------------------------------------------------------------------

## Load image
paths = ['../photos_ball/top','../photos_ball/front', 
         '../photos_apple', '../photos_cup',
         'C:/Users/WANGH0M/Desktop/Drillbit/photos_drill/front']
ifolder = 4
path = paths[ifolder] ### need to choose the path name

images = [cv2.resize(cv2.imread(file), (0,0), fx=0.1, fy=0.1) for file in glob.glob(path+"/*.jpg")]

# Segmentation detector
ys = YOLOSegmentation("yolov8m-seg.pt")

is_plot = 1

for i, img in enumerate(images):
    print(i)

    #if i==0:
    bboxes, classes, segmentations, scores = ys.detect(img)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        #print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        #print(class_id)
        ##Huinote: drillbit case, class_id is not unique, not only 10, below is not accurate
        if ifolder==2 and class_id==47 or ifolder == 4 and class_id ==10:
            (x, y, x2, y2) = bbox

            if is_plot:
                ## get the bounding rectangle:
                cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

                ## get the contour polyline:
                cv2.polylines(img, [seg], True, (0, 0, 255), 4)
                np.savetxt(path+'/polyline/'+ str(i+1), seg.astype(int), fmt='%i', delimiter=',')
                
                center = ((x+x2)//2, (y+y2)//2)
                if True:
                    ## circle inside rectangle
                    radius = max((x2-x)//2, (y2-y)//2)
                else:
                    ## bounding circle from rectangle
                    radius = int(np.sqrt((x-x2)**2+(y-y2)**2)//2)
                cv2.circle(img, center, radius,(0,255,0),2)

                cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                #cv2.putText(img, str(i)+"-Contour", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                level = Levelset(img, seg)
                length,Pt1,Pt2 = level.diameter, level.Pt1, level.Pt2
                anchor = ((Pt1[0]+Pt2[0])//2-5, (Pt1[1]+Pt2[1])//2-5)

                ##Hui TODO: the ratio need to be computed
                ratio_px_mm = 14 / 14 
                mm = length / ratio_px_mm

                cv2.putText(img, "{} pixel".format(round(mm, 2)), anchor, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


    cropped = img[y:y2, x:x2].copy() ##note the x, y coordinates, should be y, x     

    if is_plot:
        cv2.imshow("cropped" , cropped)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        name = path + "/bounding/" + str(i+1)
        cv2.imwrite(name + ".png", cropped)  