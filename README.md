# imageprocessing

## File notes:
1. panorama_stitching.py: the main file, which will call circle_contours.py, crop_patch.py, read_files.py.
2. circle_controus.py: several ways to extract circles as contours, but only circle_Hough() is useful in this case.
3. crop_patch.py: crop the spherical patch.
4. read_files.py: read the .csv files in folder ball_photos.
5. folder ball_photos: 8 chosen photos with constant speed; computed polyline-points from Grasshopper (shoud be moved into python code).
6. folder docs: some funtions related to OpenCV, Skimage, Pillow in folder else.
7. folder else: some prepared / learning files (in fact, no use).
8. Jupyter file Ball photos stitching.ipynb: the initial test version (please refer the panorama_stitching.py file).


## install by anaconda on Windows
1. conda create -n opencv
2. conda activate opencv
3. python3 --version
4. pip install --upgrade pip
5. pip install opencv-contrib-python
6. conda install numpy (or Jupyter, matplotlib else if needed)

