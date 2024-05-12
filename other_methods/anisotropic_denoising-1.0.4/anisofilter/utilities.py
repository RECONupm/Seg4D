"""
Description:
	utility functions for point cloud processing

- 2021 Noiseless Imaging Oy - Tampere, Finland -
- Zhongwei Xu, Alessandro Foi -

Copyright (c) 2019-2021 Noiseless Imaging Oy (Ltd).
All rights reserved.
This work (software, material, and documentation) shall only
be used for nonprofit noncommercial purposes.
Any unauthorized use of this work for commercial or for-profit purposes
is prohibited.
"""
import numpy as np
import os
import platform
import ctypes
from sklearn.neighbors import KDTree
from numpy import linalg as LA


def read_ply_single_class(ply_file):
    """
    Description: 
    	read the point cloud data from a PLY file in single precision 
    """
    file = open(ply_file, "r")
    all_lines = file.readlines()
    l = 0
    while all_lines[l] != 'end_header\n':
        l = l + 1

    pcd = []
    for x in range(l+1, len(all_lines)):
        line = all_lines[x]
        trimed_line = line.strip()
        one_row = [float(i) for i in trimed_line.split(" ")]
        pcd.append(one_row)
    pcd = np.array(pcd,dtype=np.float32)
    return pcd  
    

def pcd_std_est(pcd):    
    """ 
    Description: 
    	estimate the standard deviation and density of a point cloud  
    """
    # Check OS info to decide library file
    path = os.path.dirname(__file__)	
    if platform.system() == "Windows":
        ext = ".dll"
    if platform.system() == "Linux":
        ext = ".so"
    if platform.system() == "Darwin":
        ext = "_mac.so"    

    # define some pointers
    num_point = len(pcd)
    p_pcd = pcd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sigma_pcd = np.zeros(1, dtype=np.float32)
    dens_pcd = np.zeros(1, dtype=np.float32)
    p_sig_pcd = sigma_pcd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_dens_pcd = dens_pcd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


    # ------ C_FXN: func of kdtree knn search  ------ #
    libname = "libpcd_kdtree_knn"+ext 
    c_lib = ctypes.CDLL(os.path.join(path,libname))             
    c_fxn = c_lib.pcd_kdtree_knn
    c_fxn.restypes = None
    c_fxn.argtypes = (ctypes.POINTER(ctypes.c_float),   # pcd
                       ctypes.c_int,                    # n_p
                       ctypes.c_int,                    # num_k
                       ctypes.POINTER(ctypes.c_int))    # knn

    # ------ C_FXN_S: func for pcd standard-deviation estimation ------ #
    # A. Create library
    libname_s = "libpcd_std_est_var_numk"+ext
    c_library = ctypes.CDLL(os.path.join(path,libname_s))
    # B. Specify function signatures
    c_fxn_s = c_library.pcd_std_est_var_numk
    c_fxn_s.argtypes = (ctypes.POINTER(ctypes.c_float),     # pcd
                        ctypes.POINTER(ctypes.c_int),       # idx
                        ctypes.c_int,                       # n_p
                        ctypes.c_int,                       # num_k
                        ctypes.POINTER(ctypes.c_float),     # sigma_pcd
                        ctypes.POINTER(ctypes.c_float))     # density_pcd

    # C. Invoke function
    num_k = 50
    idx_knn = np.zeros((num_point, num_k), dtype=np.int32)
    p_idx_knn = idx_knn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_fxn(p_pcd, num_point, num_k, p_idx_knn)
    c_fxn_s(p_pcd, p_idx_knn, num_point, num_k, p_sig_pcd, p_dens_pcd)

    #============= recompute with bigger K, if sigma_pcd is large =============
    if 1.5 < sigma_pcd[0] * np.sqrt(dens_pcd[0]) < 3.5:
        print("Recompute std as Sigma > 1.5")
        num_k = 200
        idx_knn = np.zeros((num_point, num_k), dtype=np.int32)
        p_idx_knn = idx_knn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_fxn(p_pcd, num_point, num_k, p_idx_knn)
        c_fxn_s(p_pcd, p_idx_knn, num_point, num_k, p_sig_pcd, p_dens_pcd)

    # ============= recompute with bigger K, if sigma_pcd is large =============
    if 3.5 <= sigma_pcd[0] * np.sqrt(dens_pcd[0]) < 4.5:
        print("Recompute std as Sigma > 3.5")
        num_k = 300
        idx_knn = np.zeros((num_point, num_k), dtype=np.int32)
        p_idx_knn = idx_knn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_fxn(p_pcd, num_point, num_k, p_idx_knn)
        c_fxn_s(p_pcd, p_idx_knn, num_point, num_k, p_sig_pcd, p_dens_pcd)

    # ============= recompute with bigger K, if sigma_pcd is large =============
    if sigma_pcd[0] * np.sqrt(dens_pcd[0]) >= 4.5:
        print("Recompute std as Sigma > 4.5")
        num_k = 500
        idx_knn = np.zeros((num_point, num_k), dtype=np.int32)
        p_idx_knn = idx_knn.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_fxn(p_pcd, num_point, num_k, p_idx_knn)
        c_fxn_s(p_pcd, p_idx_knn, num_point, num_k, p_sig_pcd, p_dens_pcd)

    return [sigma_pcd[0], dens_pcd[0]]
    
def square_root_mean_point2surf_error(point_cloud_check, point_cloud_ori):
    """
    Description:
    	This function compares two point clouds(estimated one & ground truth),
    	which can have different number of points, by computing the sqaured mean
    	point - to - surface distance between them.
    	
    	INPUTS:
     		POINT_CLOUD_CHECK:      M * 3 matrix of the estimated point cloud
    		POINT_CLOUD_ORI:        N * 3 matrix of the ground truth point cloud
    	OUTPUTS:
    		SMSE_POINT2SURF:        squared mean point-to-surface distance
    		DISTANCE_POINT2SURF:    M * 1 matrix of point-to-surface distance
    	        	                for each point in POINT_CLOUD_CHECK
    """
    num_point_est = len(point_cloud_check)
    distance_point2surf = np.zeros(num_point_est, dtype=np.float32)

    kdt = KDTree(point_cloud_ori, leaf_size=10, metric='euclidean')
    dist, idx = kdt.query(point_cloud_check, k=1)  # query points from point_cloud_check
    dist2, idx2 = kdt.query(point_cloud_ori, k=5)  # query points from point_cloud_ori

    kk = 0
    for i in range(0, num_point_est):
        cap_i = idx[i]   # index of nearest point in point_cloud_ori

        # compute Local Coordinate
        knn_neigh_cap_i = point_cloud_ori[idx2[cap_i][0]]
        cap_m = knn_neigh_cap_i - np.mean(knn_neigh_cap_i, axis=0)
        p1 = cap_m.transpose().dot(cap_m)
        [cap_d, cap_v] = LA.eig(p1)
        cap_v_rotate = np.real(cap_v)
        cap_i_rotate = np.argsort(-cap_d)

        v_normal = cap_v_rotate[:, cap_i_rotate[2]]
        distance_point2surf[kk] = np.sqrt(((point_cloud_check[i] - point_cloud_ori[cap_i]).dot(v_normal)**2))
        kk = kk + 1
    smse_point2surf = np.sqrt(np.mean(distance_point2surf**2))
    return smse_point2surf          
    
    
def write_ply_only_pos(pcd, filename):
    """
    Description:
    	export a point cloud matrix into a PLY file
    	 
    	INPUTS:  
    		PCD      : N * 3 matrix in single precision, contains the point cloud 
    	           	   with only spatial coordinate information.
    		FILENAME : a string of the name of the PLY file to be saved.
    	OUTPUTS:
    		NONE	
    """
    fid = open(filename, "w")
    fid.close()
    fid = open(filename, "a")
    fid.write("ply\n")
    fid.write("format ascii 1.0\n")
    fid.write("comment Noiseless Imaging Oy (Ltd) generated\n")
    fid.write("element vertex " + str(len(pcd)) + "\n")
    fid.write("property float x\n")
    fid.write("property float y\n")
    fid.write("property float z\n")
    fid.write("end_header\n")
    for i in range(len(pcd)):
        fid.write(str(pcd[i, 0]) + " " + str(pcd[i, 1]) + " " + str(pcd[i, 2]) + "\n")
    fid.close()   
