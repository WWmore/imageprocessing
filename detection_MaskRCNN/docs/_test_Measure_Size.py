
"https://pysource.com/2022/11/29/measure-size-of-objects-in-real-time-with-computer-vision-opencv-with-python/"

import cv2
import numpy as np
import pyrealsense2 as rs

import mrcnn.model as modellib
from mrcnn.visualize import random_colors, get_mask_contours, InferenceConfig, draw_mask

#Load model
num_classes = 1
config = InferenceConfig(num_classes=num_classes, image_size=1024)
model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
model.load_weights("dnn_model/mask_rcnn_object_0002.h5", by_name=True)

# Generate random colors
colors = random_colors(num_classes)

#Load carema
cap = cv2.VideoCapture(r"ok2.mp4")

while True:

    img  = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.detect([image])

    if results:
        r = results[0]
        object_count = len(r["class_ids"])
        for i in range(object_count):
            # 1. Rectangle
            y1, x1, y2, x2 = r["rois"][i]
            cv2.rectangle(img,(x1, y1), (x2, y2), (25, 15, 220), 3)
            # Width
            width = x2 - x1
            #cv2.putText(img, str(width), (x1 , y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (25, 15, 220), 2)
            # 1.4 CM = 153
            # 14 MM = 153
            # Ratio Pixels to MM
            ratio_px_mm = 153 / 14
            mm = width / ratio_px_mm
            cm = mm / 10

        cv2.rectangle(img, (x1, y1 - 60), (x1 + 270, y1 - 5), (25, 15, 220), -1)
        cv2.putText(img, "{} CM".format(round(cm, 2)), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)