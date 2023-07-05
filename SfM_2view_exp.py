"""

Sunny Karmur: SFM_Point_Cloud


"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Sample test 
impath_left = 'C:/Users/sunny/Desktop/workshop/Github_projects/SFM_Point_Cloud/data/uf_left.png'
impath_right = 'C:/Users/sunny/Desktop/workshop/Github_projects/SFM_Point_Cloud/data/uf_right.png'
K = np.loadtxt('C:/Users/sunny/Desktop/workshop/Github_projects/SFM_Point_Cloud/data/K_iphone_reduced.txt')
dist = np.array([9.36724635e-02,-1.02969429e+00,1.05257590e-04,3.53158956e-04,2.44197170e+00]) # use the actual lens distortions params if available

## Use your data
#impath_left = 'data/your_left.png'
#impath_right = 'data/your_right.png'
#K = np.loadtxt('data/K_your.txt')
#dist = np.zeros(5) # use the actual lens distortions params if available

# source libs 
from geo3D_exp import SceneReconstruction3D
recon_3D = SceneReconstruction3D(K, dist)

# the SFM pipeline

recon_3D.load_image_pair(impath_left, impath_right)
recon_3D._extract_keypoints_sift()
recon_3D._estimate_fundamental_matrix()
recon_3D.draw_epipolar_lines()
recon_3D._estimate_essential_matrix()
recon_3D._find_camera_matrices_rt()
recon_3D._find_projection_matrices()
recon_3D._triangulate_3d_points()
recon_3D.plot_point_cloud()
