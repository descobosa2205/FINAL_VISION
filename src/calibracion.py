import cv2
print("OpenCV should be 4.8.0.76 Current version:", cv2.__version__)
from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os
from pathlib import Path


# Cargamos la imagenes
BASE = Path(__file__).resolve().parent      # .../Lab_Project/src
DATA = BASE.parent / "Data"                # .../Lab_Project/Data
OUTPUT = BASE.parent / "output"



def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

imgs = [str(DATA / f"Imagen_{i}.jpg") for i in range(1, 19)]
imgs = load_images(imgs)

corners = [cv2.findChessboardCorners(imgs[i],(4,6)) for i in range(18)]

corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# List containing Gray images
imgs_gray = [cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY) for i in range(18)]

corners_refined = [cv2.cornerSubPix(i, cor[1], (4, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

imgs_copy = copy.deepcopy(imgs)

corners_draw = [cv2.drawChessboardCorners(img,(4,6),corner[1],True) for img, corner in zip(imgs_copy, corners_copy) if corner[0]] 

# Show images and save when needed

def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def write_image(output_folder: str, img_name: str, img: np.array):
    img_path = os.path.join(output_folder,img_name)
    cv2.imwrite(img_path, img)


for i in range(len(corners_draw)):
    write_image(OUTPUT,f"imagen_corners{i}.jpg",corners_draw[i])

def get_chessboard_points(chessboard_shape, dx, dy):
    eje_x, eje_y = chessboard_shape
    puntos = []
    for y in range(eje_y):
        for x in range(eje_x):
            coordenadas= [x*dx,y*dy,0]
            puntos.append(coordenadas)
    return np.array(puntos,dtype=np.float32)

chessboard_points = [get_chessboard_points((4, 6), 0.0315, 0.0315) for _ in corners]
valid_corners = [cor[1] for cor in corners if cor[0]]
valid_corners = np.asarray(valid_corners, dtype=np.float32)

cam_matrix = np.zeros((3,3),dtype=np.float32)
dist_coefs = np.zeros((1,4),dtype=np.float32)
img_shape = (imgs[0].shape[1],imgs[0].shape[0])
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, img_shape, cam_matrix, dist_coefs)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)