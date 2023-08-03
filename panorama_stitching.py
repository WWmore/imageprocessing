import numpy as np
import cv2
import glob

from circle_contours import circle_Hough
from read_files import read_csv
from crop_patch import crop_image

def ReadPatchContour(path, is_drill):
    path += '\\csv_bigpatch'
    is_sign = -1 if is_drill else 1

    data = {
        'phi_bottom': read_csv(path, '\\phi_bottom.csv') * is_sign + np.pi/2,
        'phi_left': read_csv(path, '\\phi_left.csv') * is_sign + np.pi/2,
        'rho_left':  read_csv(path, '\\rho_left.csv'),
        'phi_right':  read_csv(path, '\\phi_right.csv') * is_sign + np.pi/2,
        'rho_right':  read_csv(path, '\\rho_right.csv'),

        'phi_bdry':  read_csv(path, '\\phi_bdry.csv') * is_sign + np.pi/2,
        'rho_bdry':  read_csv(path, '\\rho_bdry.csv'),

        'phi1':  read_csv(path, '\\patch_list1_phi.csv') * is_sign + np.pi/2,
        'rho1':  read_csv(path, '\\patch_list1_r.csv'),

        'phi2':  read_csv(path, '\\patch_list2_phi.csv') * is_sign + np.pi/2,
        'rho2':  read_csv(path, '\\patch_list2_r.csv'),

        'phi3':  read_csv(path, '\\patch_list3_phi.csv') * is_sign + np.pi/2,
        'rho3':  read_csv(path, '\\patch_list3_r.csv'),

        'phi4':  read_csv(path, '\\patch_list4_phi.csv') * is_sign + np.pi/2,
        'rho4':  read_csv(path, '\\patch_list4_r.csv'),

        'phi5':  read_csv(path, '\\patch_list5_phi.csv') * is_sign + np.pi/2,
        'rho5':  read_csv(path, '\\patch_list5_r.csv'),

        'phi6':  read_csv(path, '\\patch_list6_phi.csv') * is_sign + np.pi/2,
        'rho6':  read_csv(path, '\\patch_list6_r.csv'),

        'phi7':  read_csv(path, '\\patch_list7_phi.csv') * is_sign + np.pi/2,
        'rho7':  read_csv(path, '\\patch_list7_r.csv'),
    }

    return data


class RectangularPatch:
    def __init__(self, path, img, is_drill=False, **kwargs):
        self.path = path
        self.is_sign = -1 if is_drill else 1

        self.phi_bottom = kwargs.get('phi_bottom')
        self.phi_left = kwargs.get('phi_left')
        self.rho_left = kwargs.get('rho_left')
        self.phi_right = kwargs.get('phi_right')
        self.rho_right = kwargs.get('rho_right')
        self.phi_bdry = kwargs.get('phi_bdry')
        self.rho_bdry = kwargs.get('rho_bdry')
        self.phi1 = kwargs.get('phi1')
        self.rho1 = kwargs.get('rho1')
        self.phi2 = kwargs.get('phi2')
        self.rho2 = kwargs.get('rho2')
        self.phi3 = kwargs.get('phi3')
        self.rho3 = kwargs.get('rho3')
        self.phi4 = kwargs.get('phi4')
        self.rho4 = kwargs.get('rho4')
        self.phi5 = kwargs.get('phi5')
        self.rho5 = kwargs.get('rho5')
        self.phi6 = kwargs.get('phi6')
        self.rho6 = kwargs.get('rho6')
        self.phi7 = kwargs.get('phi7')
        self.rho7 = kwargs.get('rho7')

        ### Read 1 image and get the cropped subimg
        img = self.change_jpg_to_png(path, img)
        img = self.rescale(img)
        self.subimg = self.crop(img)
        self.subimg_copy = self.subimg.copy()
        
        ### Plot bounding circle
        self.center, self.radius = self.get_bounding_circle()

        ### Plot extracted boundary contour:
        self.get_boundary_contour()

        ### Plot to show the extracted cropped_patch:
        self.get_cropped_patch()

        ### Plot deformed cropped_patch:
        self.patch = self.get_rectangular_patch()
    #-------------------------------------------------------------------

    def change_jpg_to_png(self, path, img):
        # change jpg to png
        cv2.imwrite(path + "\\1.png", img)
        img = cv2.imread(path+'\\1.png',1)
        return img

    def rescale(self, img):
        ## rescale the photo
        img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
        return img

    def crop(self, img):
        ## crop the photo
        num_row, num_col = img.shape[0], img.shape[1]
        subimg = img[:, int(num_col*0.2):int(num_col*0.8), :]

        # cv2.imshow("Sub-image", subimg)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return subimg

    def convert_grayscale(self, subimg):
        ## convert it to a grayscale image
        gray_img = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        return gray_img

    def get_bounding_circle(self):
        if False:
            "computed way"
            center, radius = circle_Hough(self.subimg)
        else:
            "use above function to get a constant circle of the first photo"
            center = np.array([244, 294])
            radius = 212 if self.is_sign==1 else 230
            # draw the outer circle
            cv2.circle(self.subimg,center,radius,(0,0,255),2)
            # draw the center of the circle
            cv2.circle(self.subimg,center,2,(0,0,255),10)
            # print(circle[0],circle[1])

            cv2.imshow("Contours",self.subimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return center, radius

    def get_boundary_contour(self):
        center, radius = self.center, self.radius

        phi_bottom = self.phi_bottom
        phi_left = self.phi_left
        rho_left  = self.rho_left 
        phi_right = self.phi_right
        rho_right = self.rho_right

        ### Plot the polyline
        pts_bottom = np.c_[(radius * np.cos(phi_bottom)).astype(int), (radius * np.sin(phi_bottom)).astype(int)]
        pts_left = np.c_[(radius * rho_left * np.cos(phi_left)).astype(int), (radius * rho_left * np.sin(phi_left)).astype(int)]
        pts_right = np.c_[(radius * rho_right * np.cos(phi_right)).astype(int), (radius * rho_right * np.sin(phi_right)).astype(int)]

        # bdry_pts = center + np.vstack((pts_bottom[::-1], pts_left, pts_right))

        ### Extract the bounded patch
        pts1 = (center + pts_left).reshape((-1, 1, 2))
        pts2 = (center + pts_bottom).reshape((-1, 1, 2))
        pts3 = (center + pts_right).reshape((-1, 1, 2))

        isClosed = False
        
        # Using cv2.polylines() method, Draw a Blue polygon with thickness of 1 px
        ply1 = cv2.polylines(self.subimg, [pts1], isClosed, (255, 0, 0), thickness=2)
        ply2 = cv2.polylines(self.subimg, [pts2], isClosed, (255, 0, 0), thickness=2)
        ply3 = cv2.polylines(self.subimg, [pts3], isClosed, (255, 0, 0), thickness=2)
        
        # Displaying the image (Note these polys will appear in the later extracted-subimg)
        cv2.imshow(self.path, ply1)
        cv2.imshow(self.path, ply2)
        cv2.imshow(self.path, ply3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def get_cropped_patch(self):
        ### Extract cropped patch:
        phi_bdry = self.phi_bdry
        rho_bdry = self.rho_bdry

        pts_bdry = np.c_[(self.radius * rho_bdry * np.cos(phi_bdry)).astype(int), (self.radius * rho_bdry * np.sin(phi_bdry)).astype(int)]

        pts_bdry = (self.center + pts_bdry).reshape((-1, 1, 2))

        crop_image(self.subimg, pts_bdry)


    ### Read the 5 lists of points of the extracted patch
    def get_list_of_pts(self, phi, rho):
        x = (self.radius * rho * np.cos(phi)).astype(int)
        y = (self.radius * rho * np.sin(phi)).astype(int)
        return np.c_[x, y] + self.center

    def strip(self, ptlist1, ptlist2, output_pts, width, height):
        subimg = self.subimg_copy
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
        
    def get_rectangular_patch(self):
        pts_list1 = self.get_list_of_pts(self.phi1, self.rho1)
        pts_list2 = self.get_list_of_pts(self.phi2, self.rho2)
        pts_list3 = self.get_list_of_pts(self.phi3, self.rho3)
        pts_list4 = self.get_list_of_pts(self.phi4, self.rho4)
        pts_list5 = self.get_list_of_pts(self.phi5, self.rho5)
        pts_list6 = self.get_list_of_pts(self.phi6, self.rho6)
        pts_list7 = self.get_list_of_pts(self.phi7, self.rho7)

        num = len(pts_list1)
        mid_ind = int(num/2)
        width1 = np.linalg.norm(pts_list2[mid_ind]-pts_list1[mid_ind])
        width2 = np.linalg.norm(pts_list3[mid_ind]-pts_list2[mid_ind])
        width3 = np.linalg.norm(pts_list4[mid_ind]-pts_list3[mid_ind])
        width4 = np.linalg.norm(pts_list5[mid_ind]-pts_list4[mid_ind])
        width5 = np.linalg.norm(pts_list6[mid_ind]-pts_list5[mid_ind])
        width6 = np.linalg.norm(pts_list7[mid_ind]-pts_list6[mid_ind])
        width = int((width1+width2+width3+width4+width5+width6)/6)

        hgt1 = np.linalg.norm(pts_list1[1:]- pts_list1[:-1], axis=1)
        hgt2 = np.linalg.norm(pts_list2[1:]- pts_list2[:-1], axis=1)
        hgt3 = np.linalg.norm(pts_list3[1:]- pts_list3[:-1], axis=1)
        hgt4 = np.linalg.norm(pts_list4[1:]- pts_list4[:-1], axis=1)
        hgt5 = np.linalg.norm(pts_list5[1:]- pts_list5[:-1], axis=1)
        hgt6 = np.linalg.norm(pts_list6[1:]- pts_list6[:-1], axis=1)
        hgt7 = np.linalg.norm(pts_list7[1:]- pts_list7[:-1], axis=1)        
        height = int(np.mean((hgt1+hgt2+hgt3+hgt4+hgt5+hgt6+hgt7)/5))
        # print(width, height)

        ### Merge pieces of rectangular patches together to form 1 big rectangular patch
        output_pts = np.float32([[0, 0],
                            [0, height - 1],
                            [width - 1, height - 1],
                            [width - 1, 0]])
        s1 = self.strip(pts_list1, pts_list2, output_pts, width, height)
        s2 = self.strip(pts_list2, pts_list3, output_pts, width, height)
        s3 = self.strip(pts_list3, pts_list4, output_pts, width, height)
        s4 = self.strip(pts_list4, pts_list5, output_pts, width, height)
        s5 = self.strip(pts_list5, pts_list6, output_pts, width, height)
        s6 = self.strip(pts_list6, pts_list7, output_pts, width, height)

        patch = cv2.hconcat([s1,s2,s3,s4,s5,s6])

        # if self.is_sign==1:
        #     patch = cv2.flip(patch, 1)

        cv2.imshow("Rectangular Patch" , patch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return patch



if __name__ == "__main__":

    if 1:
        path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\ball_photos'
        is_drill = False
    else:
        path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\drill_photos'
        is_drill = True

    data = ReadPatchContour(path, is_drill)
    
    images = [cv2.imread(file) for file in glob.glob(path+"\\*.jpg")]
    num_img = len(images)
    # print(num_img)

    i = 0
    for img in images:
        print(i)
        if i==0:
            patch = RectangularPatch(path, img, is_drill, **data).patch
        else:
            pat = RectangularPatch(path, img, is_drill, **data).patch
            patch = cv2.hconcat([pat, patch])
        i += 1

    cv2.imshow("Panorama" , patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



