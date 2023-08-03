import numpy as np
import cv2
import glob

from circle_contours import circle_Hough
from read_files import read_csv
from crop_patch import crop_image


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
    phi = read_csv(self.path, '\\patch_list1_phi.csv') * self.is_sign + np.pi/2
    rho = read_csv(self.path, '\\patch_list1_r.csv')
    pts_list1 = self.get_list_of_pts(phi, rho)

    phi = read_csv(self.path, '\\patch_list2_phi.csv') * self.is_sign + np.pi/2
    rho = read_csv(self.path, '\\patch_list2_r.csv')
    pts_list2 = self.get_list_of_pts(phi, rho)

    phi = read_csv(self.path, '\\patch_list3_phi.csv') * self.is_sign + np.pi/2
    rho = read_csv(self.path, '\\patch_list3_r.csv')
    pts_list3 = self.get_list_of_pts(phi, rho)

    phi = read_csv(self.path, '\\patch_list4_phi.csv') * self.is_sign + np.pi/2
    rho = read_csv(self.path, '\\patch_list4_r.csv')
    pts_list4 = self.get_list_of_pts(phi, rho)

    phi = read_csv(self.path, '\\patch_list5_phi.csv') * self.is_sign + np.pi/2
    rho = read_csv(self.path, '\\patch_list5_r.csv')
    pts_list5 = self.get_list_of_pts(phi, rho)

    num = len(pts_list1)
    mid_ind = int(num/2)
    width1 = np.linalg.norm(pts_list2[mid_ind]-pts_list1[mid_ind])
    width2 = np.linalg.norm(pts_list3[mid_ind]-pts_list2[mid_ind])
    width3 = np.linalg.norm(pts_list4[mid_ind]-pts_list3[mid_ind])
    width4 = np.linalg.norm(pts_list5[mid_ind]-pts_list4[mid_ind])
    width = int((width1+width2+width3+width4)/4)

    hgt1 = np.linalg.norm(pts_list1[1:]- pts_list1[:-1], axis=1)
    hgt2 = np.linalg.norm(pts_list2[1:]- pts_list2[:-1], axis=1)
    hgt3 = np.linalg.norm(pts_list3[1:]- pts_list3[:-1], axis=1)
    hgt4 = np.linalg.norm(pts_list4[1:]- pts_list4[:-1], axis=1)
    hgt5 = np.linalg.norm(pts_list5[1:]- pts_list5[:-1], axis=1)
    height = int(np.mean((hgt1+hgt2+hgt3+hgt4+hgt5)/5))
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

    patch = cv2.hconcat([s1,s2,s3,s4])

    if self.is_sign==1:
        patch = cv2.flip(patch, 1)

    cv2.imshow("Rectangular Patch" , patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return patch



path = r'C:\\Users\\WANGH0M\\Desktop\\opencv\\drill_photos'
is_drill = True

images = [cv2.imread(file) for file in glob.glob(path+"\\*.jpg")]
num_img = len(images)
# print(num_img)

cv2.imshow("Panorama" , patch)
cv2.waitKey(0)
cv2.destroyAllWindows()



