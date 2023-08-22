# imageprocessing by Hui

#### install by anaconda on Windows
1. conda create -n opencv
2. conda activate opencv
3. python3 --version
4. pip install --upgrade pip
5. pip install opencv-contrib-python
6. conda install numpy (or Jupyter, matplotlib else if needed)



## Algorithm pipeline
![pipeline](docs/pipeline.png?raw=true "Pipeline")



## Steps:
1. (may skip to the next) From Rhino_files / Grasshopper file to produce the .csv files, which list the phi and rho numbers of the bounding mesh patch. The mesh patch includes 14 strips (larger area, almost a circle), 8 strips (almost 1/2 circle), 6 sttips (too small to use) and 4 strips (too small to use) have already been pre-produced in csv folder. It is better to directly use these files and do not use Grasshopper to produce new one, except one knows how to produce.
2. In extract_patches.py: choose the chosen path of photo folder and csv. Then run the file to produce warped rectangular patches, which are saved in /rectangle folder.
3. In panorama.py: choose the path of the /rectangle folder, run the file to get stitched panorama image. The algorithm has two parameters: "crop" and "confidence_threshold" to chose. If "crop": True, the final panorama image will be rectangled with regular boundary, but it will crop some areas. "confidence_threshold" relates to a range from 0 to 1, usually 0.3-0.5 are good numbers, which needs to check and test.



## File notes:
1. extract_patches.py: the main file, which will call circle_contours.py, crop_patch.py, read_files.py. It will produced rectangular patches which are to be stitched in panorama.py file.
2. circle_controus.py: several ways to extract circles as contours. It seems circle_canny() is helpful in this case, but be carefuly for the parameters chosen, otherwise the bounding contour is not good by eyes.
3. crop_patch.py: crop the spherical patch.
4. read_files.py: read the .csv files in folder ball_photos.
5. folder photos_ball: 8 chosen photos with constant speed were chosen for both 'Top' view and 'Front' view; computed polyline-points (.csv) from Grasshopper (shoud be saved in csv/csv_patch_xstrip folder).
6. folder docs: some funtions related to OpenCV, Skimage, Pillow in folder else.
7. folder else: some prepared / learning files (in fact, no use).
8. Jupyter file "Ball photos stitching.ipynb": the test version of the extract_patches.py file.
9. Jupyter file "AffineStitch.ipynb": the test version of the panorama.py file.
