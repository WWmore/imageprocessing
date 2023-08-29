"__author__ = Hui Wang"

""" Functions:
Levelset: get the levelset of the waistline of the object
"""
import numpy as np
import cv2
import glob
#-----------------------------------------------------------------------------------

class Levelset:
    def __init__(self, img, poly, **kwargs):
        self.img  = img
        self.poly = poly
        self.diameter,self.Pt1,self.Pt2 = self.waistline(self.img, self.poly)

    def get_intersect0(self, a1, a2, b1, b2):##bug, no use
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1,a2,b1,b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def get_intersect(self, endPt1, endPt2, y0):
        if endPt1[0] == endPt2[0]:
            "poly_segment parallel to y-axis"
            x = endPt1[0]
        elif endPt1[1] == endPt2[1]:
            "mostly should not happen: poly-segment parallel to x-axis"
            x = (endPt1[0] + endPt2[0]) / 2
        else:
            x = (y0-endPt2[1]) * (endPt1[0]-endPt2[0]) / (endPt1[1]-endPt2[1]) + endPt2[0]
        return (int(x), y0)

    def waistline(self, img, poly):
        """get the largest waistline of the bounding closed polyline on the image
        constant y0 goes through the image[:,min:max], 
        which intersects the left and right polylines at two points A, B
        waist = |A-B|
        """
        x_min, x_max = np.min(poly[:,0]), np.max(poly[:,0]) ## integers
        y_min, y_max = np.min(poly[:,1]), np.max(poly[:,1]) ## integers
        y_arr0, y_arr1 = poly[:-1, 1], poly[1:,1]
        # print(y_min, y_max)
        dist,Ax,Ay,Bx,By = [],[],[],[],[]
        for y0 in range(y_min, y_max):
            if y0 != y_min:
                i1 = np.where(y_arr0>=y0)[0]
                i2 = np.where(y_arr1<=y0)[0]
                ind_l = np.intersect1d(i1,i2)[0] ## should be only one
                #print(ind_l)
                
                endPt1, endPt2 = poly[:-1,:][ind_l],  poly[1:,:][ind_l]
                A = self.get_intersect(endPt1, endPt2, y0)
                Ax.append(A[0])
                Ay.append(A[1])

                i1 = np.where(y_arr0<=y0)[0]
                i2 = np.where(y_arr1>=y0)[0]
                ind_r = np.intersect1d(i1,i2)[0] ## should be only one
                #print(ind_r)

                endPt1, endPt2 = poly[:-1,:][ind_r],  poly[1:,:][ind_r]
                B = self.get_intersect(endPt1, endPt2, y0)
                Bx.append(B[0])
                By.append(B[1])

                dist.append(np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2))

                # if y0%50 == 0:
                #     cv2.line(img, A, B, (0, 0, 255), 4)

        diameter = max(dist)
        ind = dist.index(diameter)
        endPt1, endPt2 = (Ax[ind], Ay[ind]), (Bx[ind], By[ind])
        cv2.line(img, endPt1, endPt2, (0, 0, 255), 4)
        return diameter, endPt1, endPt2

    def plot(self):
        diameter = self.waistline(self.img, self.poly)
        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return diameter

#------------------------------------------------------------------------------------------
if __name__ == "__main__":


    ## Load image
    paths = ['../photos_ball/top','../photos_ball/front', 
            '../photos_apple', '../photos_cup',
            'C:/Users/WANGH0M/Desktop/Drillbit/photos_drill/front']
    ifolder = 4
    path = paths[ifolder] ### need to choose the path name


    waist = []
    i = 0
    for file in glob.glob(path+"/*.jpg"):
        i += 1
        img = cv2.resize(cv2.imread(file), (0,0), fx=0.1, fy=0.1)
        try:
            poly = np.loadtxt(path+'/polyline/'+ str(i), dtype=int, delimiter=',')
            waist.append(Levelset(img, poly).diameter)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass

    diameter = max(waist)
    print('largest diameter=', diameter)
