# imageprocessing by Hui

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


## install by anaconda on Windows
1. conda create -n opencv
2. conda activate opencv
3. python3 --version
4. pip install --upgrade pip
5. pip install opencv-contrib-python
6. conda install numpy (or Jupyter, matplotlib else if needed)

