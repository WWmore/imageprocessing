"__author__ = Hui Wang"

""" Functions:
ReadPatchContour(path)
class RectangularPatch
"""

import numpy as np
import cv2
import glob

from circle_contours import circle_canny
from read_files import read_csv
from crop_patch import crop_image
#-----------------------------------------------------------------------------------

def ReadPatchContour(path):
    phi = [file for file in glob.glob(path+"/phi/*.csv")]
    rho = [file for file in glob.glob(path+"/rho/*.csv")]
    data = {}
    for i in range(1, len(phi)+1):
        phi_i = read_csv(phi[i-1]) + np.pi
        rho_i = read_csv(rho[i-1])
        data[str(i)] = [phi_i, rho_i]

    phi_bdry = read_csv(path+'/phi_bdry.csv') + np.pi
    rho_bdry = read_csv(path+'/rho_bdry.csv')
    data['bdry'] = [phi_bdry, rho_bdry]
    return data


class RectangularPatch:
    def __init__(self, path, img, **kwargs):
        self.path = path

        self.phi_bdry, self.rho_bdry = kwargs.get('bdry')

        ### Read 1 image and get the cropped subimg
        #img = self.crop(img) ##Hui: need to check

        self.subimg = self.rescale2(img, scale_percent=15.625)
        #self.subimg = self.rescale3(img)
        subimg_copy = self.subimg.copy()
        
        ### Plot bounding circle
        self.center, self.radius = self.get_bounding_circle()

        ### Plot extracted boundary contour:
        pts_bdry = self.get_boundary_contour()

        ### Plot to show the extracted cropped_patch:
        crop_image(self.subimg, pts_bdry)

        ### Plot deformed cropped_patch:
        self.patch = self.get_rectangle_patch(subimg_copy, self.center, self.radius, **kwargs)
    #-------------------------------------------------------------------

    def change_jpg_to_png(self, path, img):
        # change jpg to png
        cv2.imwrite(path + "\\1.png", img)
        img = cv2.imread(path+'\\1.png',1)
        return img

    def crop(self, img):
        ## crop the photo
        num_row, num_col = img.shape[0], img.shape[1]
        subimg = img[:, int(num_col*0.2):int(num_col*0.8), :]
        return subimg
    
    def crop2(self, img):
        ## crop the photo
        num_row, num_col = img.shape[0], img.shape[1]
        subimg = img[:, int(num_col*0.15):int(num_col*0.85), :]
        return subimg
    
    def rescale(self, img):
        ## rescale the photo
        img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
        return img
    
    def rescale2(self, img, scale_percent=100):
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Define the scale percentage for resizing the image
        # 100% means keeping the original size
        width = int(img.shape[1] * scale_percent / 100)  # Calculate the new width
        height = int(img.shape[0] * scale_percent / 100)  # Calculate the new height
        dim = (width, height)  # Create a tuple representing the new dimensions (width, height)

        # Resize the image using the calculated dimensions and interpolation method (INTER_AREA)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img
    
    def rescale3(self, img):
        r = 800.0 / img.shape[0]
        dim = (int(img.shape[1] * r), 800)
        image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return image

    def convert_grayscale(self, subimg):
        ## convert it to a grayscale image
        gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        return gray_img

    def get_bounding_circle(self):
        if 1:
            "computed way"
            #center, radius = circle_Hough(self.subimg)
            center, radius = circle_canny(self.subimg,ynum1=450,ynum2=550)
        else:
            "use above function to get a constant circle of the first photo"
            a,b = self.subimg.shape[:2]
            center = np.array([a//2, b//2])
            radius = min(a//2, b//2)

        # draw the outer circle
        cv2.circle(self.subimg,center,radius,(0,0,255),2)
        # draw the center of the circle
        cv2.circle(self.subimg,center,2,(0,0,255),10)

        cv2.imshow("Contours",self.subimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return center, radius

    def get_boundary_contour(self):
        phi_bdry = self.phi_bdry
        rho_bdry = self.rho_bdry

        pts_bdry = np.c_[(self.radius * rho_bdry * np.cos(phi_bdry)).astype(int), (self.radius * rho_bdry * np.sin(phi_bdry)).astype(int)]

        pts_bdry = (self.center + pts_bdry).reshape((-1, 1, 2))

        isClosed = True
        
        # Using cv2.polylines() method, Draw a Blue polygon with thickness of 1 px
        bdry = cv2.polylines(self.subimg, [pts_bdry], isClosed, (255, 0, 0), thickness=2)
        
        # Displaying the image (Note these polys will appear in the later extracted-subimg)
        cv2.imshow('Boundary', bdry)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return pts_bdry
    
    def get_rectangle_patch(self, subimg_copy, center, radius, **kwargs):
        def get_list_of_pts(phi, rho):
            ### Read points lists of the extracted patch
            x = (radius * rho * np.cos(phi)).astype(int)
            y = (radius * rho * np.sin(phi)).astype(int)
            return np.c_[x, y] + center
        
        def strip(ptlist1, ptlist2, output_pts, width, height):
            subimg = subimg_copy
            # Compute the perspective transform M
            Alist, Blist = ptlist1[1:][::-1], ptlist1[:-1][::-1] ## from the top, left-two
            Dlist, Clist = ptlist2[1:][::-1], ptlist2[:-1][::-1] ## from the top, right-two

            for i in range(1):
                pt_A, pt_B = Alist[i], Blist[i]
                pt_C, pt_D = Clist[i], Dlist[i]
                input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])   
                M = cv2.getPerspectiveTransform(input_pts,output_pts)
                out = cv2.warpPerspective(subimg,M,(width, height),flags=cv2.INTER_LINEAR)

            num = len(ptlist1)
            for i in np.arange(num-2)+1:
                pt_A, pt_B = Alist[i], Blist[i]
                pt_C, pt_D = Clist[i], Dlist[i]
                input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])   
                M = cv2.getPerspectiveTransform(input_pts,output_pts)
                out2 = cv2.warpPerspective(subimg,M,(width, height),flags=cv2.INTER_LINEAR)
                out = cv2.vconcat([out, out2])
            
            return out

        pts_list = []
        num_list = len(kwargs.keys())
        for i in range(num_list):
            if str(i) in kwargs.keys():
                phi_i, rho_i = kwargs.get(str(i))
                pts_list.append(get_list_of_pts(phi_i, rho_i))

        numw, numh = len(pts_list),  len(pts_list[0])
        mid_ind = numh//2
        w = 0
        for i in range(numw-1):
            w += np.linalg.norm(pts_list[i+1][mid_ind]-pts_list[i][mid_ind])
        width = int(w/(numw-1)) ### should be int
        
        if 0:
            "computed mean height"
            h = np.zeros(numh-1)
            for j in range(numw):
                h += np.linalg.norm(pts_list[j][1:]- pts_list[j][:-1], axis=1)
            height = int(np.mean(h)/5) ### should be int, 5 is an chosen int
        else:
            "choose the ratio of [all_width: all_height] = [1:1.2]"
            height = int(width * numw / numh * 1.2)
        #print(width, height)

        ### Merge pieces of rectangular patches together to form 1 big rectangular patch
        output_pts = np.float32([[0, 0],
                            [0, height - 1],
                            [width - 1, height - 1],
                            [width - 1, 0]])

        patch = strip(pts_list[0], pts_list[1], output_pts, width, height)
        for i in range(numw-2):
            si = strip(pts_list[i+1], pts_list[i+2], output_pts, width, height)
            patch = cv2.hconcat([patch,si])

        if 0:
            "need to check if the image is flipped horizontally"
            patch = cv2.flip(patch, 1)

        cv2.imshow("Patch" , patch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return patch


#------------------------------------------------------------------------------------------
if __name__ == "__main__":

    paths = ['./photos_ball/top','./photos_ball/front', 
             './photos_drill/top', './photos_drill/front','./photos_drill/every5',
             './photos_apple', './photos_cup']
    path = paths[5] ### need to choose the path name

    path_csvs = ['./csv/csv_patch_8strip','./csv/csv_half_14strip']
    path_csv = path_csvs[1]

    data = ReadPatchContour(path_csv)
    
    images = [cv2.imread(file) for file in glob.glob(path+"/*.jpg")]
    images = images[::-1]

    for i, img in enumerate(images):
        print(i)
        if i==0:
            patch = RectangularPatch(path, img, **data).patch
            cv2.imwrite(path + "/rectangle/1.png", patch)  
            a,b = patch.shape[:2]

        else:
            pat = RectangularPatch(path, img, **data).patch
            pat = cv2.resize(pat, (b,a), interpolation=cv2.INTER_AREA)
            name =  path + "/rectangle/" + str(i+1)
            cv2.imwrite(name + ".png", pat)  
            patch = cv2.hconcat([patch, pat])


    # cv2.imwrite(path + "/stitching.png", patch)  
    # cv2.imshow("Stitching" , patch)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


