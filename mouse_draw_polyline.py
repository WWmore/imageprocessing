import numpy
import cv2

points = []

def draw_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 1, (255,0,0),-1)
        points.append((x,y))
        pts = numpy.array(points, numpy.int32)
        cv2.polylines(image,[pts],False,(255,0,0))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # HOW TO DELETE?
        del points[-1]
        pts = numpy.array(points, numpy.int32)
        cv2.polylines(image,[pts],True,(255,0,0))


name = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\hammer.jpg'

image = cv2.imread(name, cv2.IMREAD_UNCHANGED)

cv2.namedWindow('example', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('example', draw_point)


while(1):
   cv2.imshow('example',image)

   if cv2.waitKey(20) & 0xFF == 27: ### if press ESE, then close the window
        break

cv2.destroyAllWindows()

print (points)