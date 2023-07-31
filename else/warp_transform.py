import cv2
import numpy as np

path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\gfg.png'
path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\hammer.jpg'

# Load the image
img = cv2.imread(path) 

def warp(img, pt_A, pt_B, pt_C, pt_D):
    "https://theailearner.com/tag/cv2-getperspectivetransform/"
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

    cv2.imshow("Transformed patch" , out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


a, b = img.shape[0], img.shape[1]

# All points are in format [cols, rows]
pt_A = [0,0]
pt_B = [int(a/5), int(b/10)]
pt_C = [int(a/3), int(b/2)]
pt_D = [int(a/4), int(b/3)]

warp(img, pt_A, pt_B, pt_C, pt_D)