# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:02:38 2023

@author: Luisja
"""
"""
Batch proccess of the algorithm proposed in:  

Z. Xu and A. Foi, "Anisotropic Denoising of 3D Point Clouds by Aggregation of Multiple 
Surface-Adaptive Estimates," in IEEE Transactions on Visualization and Computer Graphics, 
vol. 27, no. 6, pp. 2851-2868, 1 June 2021, doi: 10.1109/TVCG.2019.2959761.    

"""
import argparse


from anisofilter import utilities as UTI
from anisofilter import anisofilter 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',type=str,help='Path for the input file')
    parser.add_argument('--o',type=str,help='Path for the output file')
    args=parser.parse_args()  
    
    
    
    # READ PLY FILE
    pcd = UTI.read_ply_single_class(args.i)
    #ANISOTROPIC FILTER
    sigma_pcd, dens_pcd = UTI.pcd_std_est(pcd)
    pcd_de_m2c = anisofilter.anisofilter(pcd, sigma_pcd, dens_pcd)
    # WRITE PLY
    UTI.write_ply_only_pos(pcd_de_m2c, args.o)




if __name__=='__main__':
	main()