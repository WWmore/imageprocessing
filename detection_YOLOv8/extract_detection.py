import cv2
from yolo_segmentation import YOLOSegmentation
import glob
import numpy as np
#--------------------------------------------------------------------------

## Load image
paths = ['../photos_ball/top','../photos_ball/front', 
         '../photos_apple', '../photos_cup']
path = paths[2] ### need to choose the path name

images = [cv2.resize(cv2.imread(file), (0,0), fx=0.1, fy=0.1) for file in glob.glob(path+"/*.jpg")]

# Segmentation detector
ys = YOLOSegmentation("yolov8m-seg.pt")

for i, img in enumerate(images):
    print(i)
    #if i==0:
    bboxes, classes, segmentations, scores = ys.detect(img)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = bbox
        if class_id == 47:

            ## get the bounding rectangle:
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

            ## get the contour polyline:
            cv2.polylines(img, [seg], True, (0, 0, 255), 4)

            ## compute the bounding circle from rectangle
            center = ((x+x2)//2, (y+y2)//2)
            radius = int(np.sqrt((x-x2)**2+(y-y2)**2)//2)
            cv2.circle(img, center, radius,(0,255,0),2)

            ##cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, str(i)+"-Contour", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()