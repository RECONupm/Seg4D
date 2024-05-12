# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:16:07 2023

@author: Utilizador
"""

#%% LIBRARIES
import pandas as pd
import pycc

import numpy as np
from numpy.linalg import inv

from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.optimize import fsolve
from scipy.optimize import least_squares

from random import sample

from itertools import combinations

import matplotlib.pyplot as plt

import os
import datetime
import sys
import math
import yaml

#%% FUNCTIONS
def P2p_getdata (pc,nan_value=False,sc=True,color=True):
    """
    This fuction allows to transform the CC point cloud to a pandas dataframe 
        
    Parameters
    ----------
    pc(py.ccHObject) : The point cloud in CC format
    
    nan_value(bool) : True if we want to delete NaN values. False if not
    
    sc(bool) : True if we want to condier the Scalar Field layers
    
    color(bool) : True if we want to consider the R,G,B layers
    
    Returns
    -------
    pcd : Pandas frame of the point cloud
    """
    pcd = pd.DataFrame(pc.points(), columns=['X', 'Y', 'Z'])
    if color==True:
        ## GET THE RGB COLORS
        # pcd['R']=pc.colors()[:,0]
        # pcd['G']=pc.colors()[:,1] 
        # pcd['B']=pc.colors()[:,2] 
        pass
    if (sc==True):       
    ## ADD SCALAR FIELD TO THE DATAFRAME
        for i in range(pc.getNumberOfScalarFields()):
            scalarFieldName = pc.getScalarFieldName(i)  
            scalarField = pc.getScalarField(i).asArray()[:]              
            pcd.insert(len(pcd.columns), scalarFieldName, scalarField) 
            pcd=pcd.copy()
    ## DELETE NAN VALUES
    if (nan_value==True):
        pcd.dropna(inplace=True)
    return pcd    

def get_istance ():
    """
    This fuction allows to check if the selected entity is a folder or a point cloud
        
    Parameters:
    ----------
    Does not have input, operates over the selected entity in the interface
    
    Returns
    -------
    type_data (str): "point_cloud" or "folder"
    """

    CC = pycc.GetInstance() 
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]        
    if hasattr(entities, 'points'):
        type_data='point_cloud'
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   

    return type_data,number

def get_point_clouds ():
    """
    This fuction allows to check if the selected entity is a folder or a point cloud and return the number of entities (point_clouds) inside the selected folder or 1 if the selected entity is a point cloud
        
    Parameters:
    ----------
    Does not have input, operates over the selected entity in the interface
    
    Returns
    -------
    type_data (str): "point_cloud" or "folder"
    
    number (int): number of elements inside the folder or "1" if the selected entity is a point cloud
    """
    CC = pycc.GetInstance() 
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]        
    if hasattr(entities, 'points'):
        type_data='point_cloud'
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   

    return type_data,number  

def get_point_clouds_name ():
    """
    This fuction allows to extract a list with the name of the point clouds that are inside the selected folder
        
    Parameters:
    ----------
    Does not have input, operates over the selected entity in the interface
    
    Returns
    -------
    name_list (list): a list with the name of the point clouds inside the folder. If the selected entity is a point cloud the function return the name of this point cloud
    """
    CC = pycc.GetInstance() 
    name_list = []
    if not CC.haveSelection():
        raise RuntimeError("You need to select a folder or a point cloud")
    else:            
        entities = CC.getSelectedEntities()[0]
            
    if hasattr(entities,'points'):
        type_data='point_cloud'
        name_list= entities.getName()
        number=1
    else:
        type_data='folder'
        number = entities.getChildrenNumber()   
        for i in range (number):
            new_name=entities.getChild(i).getName()
            name_list.append(new_name)           
    return name_list    

def extract_longitudinal_axis (midpoints):
    """
    This fuction allows to extract the longitudinal axis of a rectangle defined by the four midpoints (from minBoundingRect) in 2D coordinates.
        
    Parameters:
    ----------
    midpoints (list): list of midpoints from the minBoundingRect. Only 2D coordinates 
    
    Returns
    -------
    p1 (list): coordinates of the first point that defines the longitudinal axis [x1,y1]
    
    p2 (list): coordinates of the second point that defines the longitudinal axis [x2,y2]
    
    longitudinal_axis (list): difference between the p2 and p1 coordinates [x3,y3]
    
    longitudinal_axis_norm (float): lenght of the longitudinal axis
    
    angle_deg (float): angle betwwen the axis and the X-axis
    
    """
    # Get all combinations of two midpoints
    point_combinations = list(combinations(midpoints, 2))
    
    # Calculate the distances between the point combinations
    distances = [np.linalg.norm(p2 - p1) for p1, p2 in point_combinations]
    
    # Find the index of the largest distance
    max_distance_index = np.argmax(distances)
    
    # Extract the two midpoints corresponding to the largest distance
    p1, p2 = point_combinations[max_distance_index]
    
    # Get the longitudinal axis (line connecting the two parallel sides)
    longitudinal_axis = p2 - p1
    longitudinal_axis_norm = distance.euclidean(p1, p2)
    
    #Calculate the angle between the longitudinal axis and the x-axis
    angle_rad = np.arctan2(longitudinal_axis[1], longitudinal_axis[0])
    angle_deg = np.degrees(angle_rad)
    return  p1,p2,longitudinal_axis, longitudinal_axis_norm, angle_deg

def minBoundingRect(hull_points_2d):
    """
    This fuction allows to extract the minimum bounding rectangle from a set of 2D points
        
    Parameters:
    ----------
    hull_points_2d (list): list of midpoints from the minBoundingRect. Only 2D coordinates [x1 y1
                                                                                            x2 y2...]
    
    Returns
    -------
    angle (float): cangle of the rectangle with respect to the X-axis
    
    min_bbox_1 (float): area of the rectangle
    
    min_bbox_2 (float): lenght of one of the rectangle axis
    
    min_bbox_3 (float): lenght of the other rectangle axis
    
    center_point (list): centroid of the rectangle [x1,y1]
    
    corner_points (list): points of the corners [x1,y1
                                                 x2,y2
                                                 x3,y3
                                                 x4,y4]
    
    mid_points (list): list of the coordinates of midpoints [x1,y1
                                                 x2,y2
                                                 x3,y3
                                                 x4,y4]
    
    """    

    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, 100000, 0, 0, 0, 0, 0, 0) 
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        R = np.array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ], [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = np.dot( [ center_x, center_y ], R )

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros( (4,2) ) # empty 2 column array
    corner_points[0] = np.dot( [ max_x, min_y ], R )
    corner_points[1] = np.dot( [ min_x, min_y ], R )
    corner_points[2] = np.dot( [ min_x, max_y ], R )
    corner_points[3] = np.dot( [ max_x, max_y ], R )
    
    
    # Calculate the midpoints on each side of the rectangle
    midpoints = np.zeros( (4,2) ) # empty 2 column array
    midpoints[0] = (corner_points[0] + corner_points[3]) / 2.0
    midpoints[1] = (corner_points[0] + corner_points[1]) / 2.0
    midpoints[2] = (corner_points[1] + corner_points[2]) / 2.0
    midpoints[3] = (corner_points[2] + corner_points[3]) / 2.0

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points,midpoints)

def rotate_point_cloud_3d(point_cloud, angle_deg):
    """
    This fuction allows rotate the the 3d point cloud around Z-axis given an angle
        
    Parameters:
    ----------
    point_cloud (numpy array): list with the 3D point cloud coordinates [x1,y1,z1
                                                                  x2,y2,z2]
    angle_deg (float): the angle of the point cloud with respect to a reference. In degrees
    
    Returns
    -------
    rotated_point_cloud ():  list with the 3D point cloud coordinates [x1,y1,z1
                                                                  x2,y2,z2]
    
    """    

    # Convert the rotation angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Compute the rotation matrix around the z-axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1]])
    
    # Apply the rotation matrix to the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    return rotated_point_cloud


def extract_points_within_tolerance(point_cloud, tolerance,rot):
    """
    This fuction allows to filter all those point that are out of a tolerance value from the main axis of its Minimum Bounding Rectangle
        
    Parameters:
    ----------
    point_cloud (list): list with the 3D point cloud coordinates [x1,y1,z1
                                                                  x2,y2,z2]
    
    tolerance (float): threshold value. Out of this value the points are deleted
    
    rot (bool): if we want to rotate the result to align the main axis of the set with respect to the X-axis
    
    Returns
    -------
    point_withint_tolerance (list): list with the resulted points [x1,y1,z1
                                                                  x2,y2,z2]
    
    skeleton (list): list with the resulted points [x1,y1,z1
                                                                  x2,y2,z2] but aligned with the X-axis    
    
    """  
    # Calculate the minimum bounding rectangle
    point_cloud_red=point_cloud[:,:2]

    _, _, _, _, center_point, corner_points,midpoints= minBoundingRect(point_cloud_red)
    
    p1,p2,longitudinal_axis, longitudinal_axis_norm, angle_deg=extract_longitudinal_axis(midpoints)
    perpendicular = np.array([-longitudinal_axis[1], longitudinal_axis[0]])
    perpendicular /= distance.euclidean([0, 0], perpendicular)
    # Calculate the dot product of each point with the direction vector
    dot_products = np.dot(point_cloud_red-p1, perpendicular)
        
    # Determine the points within the tolerance
    points_within_tolerance = point_cloud[np.abs(dot_products) <= tolerance]
    skeleton=points_within_tolerance
    if rot==True:
        points_within_tolerance=rotate_point_cloud_3d(points_within_tolerance, angle_deg)
    return points_within_tolerance, skeleton

def check_input (name_list,pc_name):
    
    """
    This fuction allows to check the type of input. Point cloud or folder.
        
    Parameters
    ----------
    name_list(list): list of available point clouds
    
    pc_name (str): name of the point cloud to get the data    
   
    Returns
    -------
    pc(py.ccHObject) : The point cloud in CC format
    """
    
    CC = pycc.GetInstance() 
    type_data, number = get_istance()

    # Some control loops to avoid wrongs assignations
    if type_data=='point_cloud' or type_data=='folder':
        pass
    else:
        raise RuntimeError("Please select a folder that contains points clouds or a point cloud")        
    if pc_name=="Not selected":
        raise RuntimeError("You need to select a valid point cloud from the list")   
    if number==0:
        raise RuntimeError("There are not entities in the folder")
    else:
        entities = CC.getSelectedEntities()[0]
        number = entities.getChildrenNumber()
        
    if type_data=='point_cloud':
        pc=entities
    else:
        for i, item in enumerate(name_list):
            if item == pc_name:
                pc = entities.getChild(i)
                break        
    return pc

def write_yaml_file(output_directory,data):
    
    """
    This fuction allows to save a yaml file in a output directory and with a data. The name is algorithm_configuration.yaml
        
    Parameters
    ----------
    output_directory (str): the directory of the output
    
    data (str): a dictionary with the data to be saved
   
    Returns
    -------
    """
    file_path=os.path.join(output_directory,'algorithm_configuration.yaml')
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)