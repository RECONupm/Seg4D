"""
ANISOFILTER gets a noisy point cloud, its noise standard deviation and density value as inputs,
outputs the denoised point cloud.

The algorithm is published in:

Z. Xu and A. Foi, "Anisotropic Denoising of 3D Point Clouds by Aggregation of Multiple 
Surface-Adaptive Estimates," in IEEE Transactions on Visualization and Computer Graphics, 
vol. 27, no. 6, pp. 2851-2868, 1 June 2021, doi: 10.1109/TVCG.2019.2959761.

Copyright (c) 2019-2021 Noiseless Imaging Oy (Ltd).
All rights reserved.
This work (software, material, and documentation) shall only
be used for nonprofit noncommercial purposes.
Any unauthorized use of this work for commercial or for-profit purposes
is prohibited.
"""

import os
import platform
import numpy as np
import time
import ctypes


def anisofilter(pcd_noi, sigma_pcd, dens_pcd):

    path = os.path.dirname(__file__) 
    if platform.system() == "Windows":
        libext = ".dll"
    if platform.system() == "Linux":
        libext = ".so"
    if platform.system() == "Darwin":
        libext = "_mac.so"

    ####### C_FXN: main func for filtering #######
    # A. Create library
    libname = "libsquare_neigh_ici_denoi_recur_pure_c_parallel"+libext
    c_library = ctypes.CDLL(os.path.join(path,libname))

    # B. Specify function signatures
    c_fxn = c_library.square_neigh_ici_denoi_recur_pure_c_parallel
    c_fxn.restypes = None 
    c_fxn.argtypes = (ctypes.POINTER(ctypes.c_float),                   # pcd
                      ctypes.c_int,                                     # n_p
                      ctypes.POINTER(ctypes.c_float),                   # sigma_map
                      ctypes.c_float,                                   # dens_pcd
                      ctypes.c_float,                                   # max_scale
                      ctypes.c_int,                                     # start_scale
                      ctypes.c_float,                                   # steps
                      ctypes.c_int,                                     # dim_inc_times
                      ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),     # out_idx_ball_matrix
                      ctypes.POINTER(ctypes.c_int),                     # idx_lens
                      ctypes.c_int,                                     # max_idx_lens
                      ctypes.c_int,                                     # itr
                      ctypes.POINTER(ctypes.c_int),                     # idx_knn
                      ctypes.c_int,                                     # num_k
                      ctypes.POINTER(ctypes.c_float))                   # closest_point_est


    #######  C_FXN2: func of kdtree radius search  #######
    libname2 = "libpcd_kdtree_radius"+libext
    c_lib2 = ctypes.CDLL(os.path.join(path,libname2))
    c_fxn2 = c_lib2.pcd_kdtree_radius
    c_fxn2.restypes = None
    c_fxn2.argtypes = (ctypes.POINTER(ctypes.c_float),                  # pcd
                       ctypes.c_int,                                    # n_p
                       ctypes.c_float,                                  # radius
                       ctypes.POINTER(ctypes.c_int),                    # out_lenghs
                       ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),    # out_idx_mat
                       ctypes.POINTER(ctypes.c_int))                    # max out length


    #######  C_FXN3: free the malloc memory from c_fxn2  #######
    c_fxn3 = c_lib2.free_the_stuff
    c_fxn3.restypes = None
    c_fxn3.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),    # out_idx_mat
                       ctypes.c_int)                                    # n_p


    #######  C_FXN4: func of kdtree knn search  #######
    libname4 = "libpcd_kdtree_knn"+libext
    c_lib4 = ctypes.CDLL(os.path.join(path,libname4))
    c_fxn4 = c_lib4.pcd_kdtree_knn
    c_fxn4.restypes = None
    c_fxn4.argtypes = (ctypes.POINTER(ctypes.c_float),              # pcd
                       ctypes.c_int,                                # n_p
                       ctypes.c_int,                                # num_k
                       ctypes.POINTER(ctypes.c_int))                # knn

    n_point = len(pcd_noi)

    # save ori dens_pcd
    dens_pcd_ori = dens_pcd

    # convert pcd to dens = 1
    pcd_noi = pcd_noi * np.sqrt(dens_pcd, dtype=np.float32)
    sigma_pcd = np.float32(sigma_pcd * np.sqrt(dens_pcd))
    dens_pcd = np.float32(1)

    # start neighborhood length(used in LPA - ICI during denosing part)
    start_scale = 3

    # the step value to increase the neighborhood size(used in LPA - ICI during denosing part)
    steps = np.sqrt(2, dtype=np.float32)

    # the maximun increase times(used in LPA - ICI during denosing part)
    dim_inc_times = 4
    max_scale = np.float32(start_scale * (steps ** dim_inc_times) / np.sqrt(dens_pcd))
    max_scale = max([max_scale, 3 * sigma_pcd])
    radius_square = np.float32(3 * max_scale ** 2)

    # OutLen: number of neighbor points in each point's surround ball
    OutLen = np.zeros(n_point, dtype=np.int32)
    p_out_len = OutLen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    OutMaxLen = np.zeros(1, dtype=np.int32)
    p_out_max_len = OutMaxLen.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    p_noipcd = pcd_noi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # knn radius search
    # p_idx_ball_out: array of pointers, idx of neighbor points in each point's surrounding ball
    int_P = ctypes.POINTER(ctypes.c_int)
    p_idx_ball_out = (int_P * n_point)()
    c_fxn2(p_noipcd, n_point, radius_square, p_out_len, p_idx_ball_out, p_out_max_len)

    # initialize sigma map
    sigma_map = sigma_pcd * np.ones(len(pcd_noi), dtype=np.float32)

    for itr in range(0, 2):
        n_p = len(pcd_noi) # get number of points after each iteration

        ######
        # compute and store knn of each point for once
        num_k = 50  # the size of knn neighbourhood


        # define pointers
        p_pcd_noi = pcd_noi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_sig_m = sigma_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        closest_point_est = np.zeros((n_p, 3), dtype=np.float32)
        p_closest_point_est = closest_point_est.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # compute knn matrix
        idx_knn2 = np.zeros((n_p, num_k), dtype=np.int32)
        p_idx_knn2 = idx_knn2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_fxn4(p_pcd_noi, n_p, num_k, p_idx_knn2)

        start_time = time.time()
        c_fxn(p_pcd_noi, n_p, p_sig_m, dens_pcd, max_scale, start_scale, steps, dim_inc_times,
              p_idx_ball_out, p_out_len, OutMaxLen[0], itr, p_idx_knn2, num_k, p_closest_point_est)

        pcd_noi = closest_point_est
        #print("--- main loop %s seconds ---" % (time.time() - start_time))
        print("///////////// itr = %d is finished, took %.5f seconds ///////////////" % (itr+1, time.time() - start_time))

    c_fxn3(p_idx_ball_out, n_point)

    pcd_de_m2c = closest_point_est / np.sqrt(dens_pcd_ori)

    return pcd_de_m2c
