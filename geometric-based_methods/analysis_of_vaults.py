# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:00:29 2023

@author: LuisJa
******** IMPORTANT: THE FUNCTION arch_estimation IS USED BY THE MODULE OF VAULTS. ANY CHANGE NEED TO BE CHECKED ALSO IN THIS MODULE. ALSO THE SET-UP WINDOW
"""
import sys
import os
import open3d as o3d
import numpy as np
import cccorelib
import pycc
import math
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree
import pandas as pd

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

import traceback

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
additional_modules_directory_2=os.path.sep.join(path_parts[:-1])
sys.path.insert(0, additional_modules_directory)
sys.path.insert(0, additional_modules_directory_2)
from main import P2p_getdata,get_istance,get_point_clouds_name
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1
from analysis_of_arches import arch_estimation

#%% INPUTS AT THE BEGINING
name_list=get_point_clouds_name()

#%% GUI
class GUI_vaults(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Initialize the variable for processing the vaults with a dictionary
        self.vault_parameters = {
            "type_of_vault": "Not selected",
            "step": 1,
            "thickness": 0.12,
            "zmax": 2.25,
            "zmin": 2.20,
            "e": 2,
            "min_points": 10,
            "s": 25
        }
        self.arch_parameters = {
            "type_of_arch": "Circular arch",
            "thickness": 0.12,
            "ransac_threshold": 0.05,
            "ransac_iterations": 5000,
            "ransac_min_samples": 100,
            "fixed_springing_line": False,
            "percent_springing":10,
            "load_pc_section": False,  
            "path_data": None
            }
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
    def main_frame (self, window):
        
        def save_file_dialog():
            # Abrir el diálogo para seleccionar la ruta de guardado
            directory = filedialog.askdirectory()
            self.output_directory = directory 
            # Mostrar la ruta seleccionada en el textbox correspondiente
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, self.output_directory)
            
        def destroy(self):
            window.destroy()  # Close the window
        
        def show_set_up_window (self):      

            def toggle_entry_state():
                if checkbox2_var.get():
                    label_percent_fix.config(state="normal")  # Make the entry editable
                    entry_percent_fix.config(state="normal")
                    entry_percent_fix.insert(0,self.arch_parameters["percent_springing"])
                else:
                    label_percent_fix.config(state="disabled")  # Make the entry read-only    
                    entry_percent_fix.config(state="disabled")
                    
            def destroy_2 ():
              set_up_window.destroy ()
              
            def run_algorithm_2 ():
                if entry_percent_fix.get() == "":
                    entry_percent_fix.config(state="normal")
                    entry_percent_fix.insert(0,0)
                self.arch_parameters["type_of_arch"]=combo_type.get()
                self.arch_parameters["ransac_threshold"]=float(entry_threshold_ransac.get())
                self.arch_parameters["ransac_iterations"]=int(entry_iterations_ransac.get())
                self.arch_parameters["ransac_min_samples"]=int(entry_minimum_samples.get())
                self.arch_parameters["fixed_springing_line"]=bool(checkbox2_var.get())
                self.arch_parameters["percent_springing"]=float(entry_percent_fix.get())
                self.arch_parameters["load_pc_section"]=bool(checkbox1_var.get())
                self.arch_parameters["thickness"]=float(entry_thickness.get())
                set_up_window.destroy ()
                
            set_up_window = tk.Toplevel(window)
            set_up_window.title("Analysis of arches")
            set_up_window.resizable (False, False)
            
            # Remove minimize and maximize button 
            set_up_window.attributes ('-toolwindow',-1)
            
            # Labels
            label_texts = [
                "Thickness threshold:",
                "Type of arch:",
                "Threshold value for RANSAC fitting:",
                "Number of iteration for RANSAC fitting:",
                "Minimum number of samples for fitting the model:",
                "Fixed springing line:",
                "Load the points of the main section"
            ]
            row_positions = [0,1,2,3,4,5,7]
            definition_of_labels_type_1 ("window",label_texts,row_positions,set_up_window,0)
            
            label_percent_fix=tk.Label(set_up_window, text="Percentage of points to fit the curve from the springs:", state="disabled")
            label_percent_fix.grid(row=6, column=0, sticky="w",pady=2)

            # Checkboxes

            # Control variables for options
            checkbox1_var = tk.BooleanVar()
            checkbox_1 = tk.Checkbutton(set_up_window, variable=checkbox1_var)
            checkbox_1.grid (row=7, column=1, sticky="e",pady=2)

            checkbox2_var = tk.BooleanVar()
            checkbox_2 = tk.Checkbutton(set_up_window, variable=checkbox2_var, command=toggle_entry_state)
            checkbox_2.grid (row=5, column=1, sticky="e",pady=2)

            # Combox
            algorithms = ["Circular arch","Pointed arch","Quarter arch"]
            combo_type = ttk.Combobox(set_up_window, values=algorithms, state="readonly")
            combo_type.current(0)
            combo_type.grid(row=1, column=1, sticky="e",pady=2)

            # Entries
            entry_thickness = tk.Entry(set_up_window,width=5)
            entry_thickness.insert(0,self.arch_parameters["thickness"])
            entry_thickness.grid(row=0, column=1, sticky="e",pady=2)
            
            entry_threshold_ransac = tk.Entry(set_up_window,width=5)
            entry_threshold_ransac.insert(0,self.arch_parameters["ransac_threshold"])
            entry_threshold_ransac.grid(row=2, column=1, sticky="e",pady=2)

            entry_iterations_ransac = tk.Entry(set_up_window,width=5)
            entry_iterations_ransac.insert(0,self.arch_parameters["ransac_iterations"])
            entry_iterations_ransac.grid(row=3, column=1, sticky="e",pady=2)

            entry_minimum_samples = tk.Entry(set_up_window,width=5)
            entry_minimum_samples.insert(0,self.arch_parameters["ransac_min_samples"])
            entry_minimum_samples.grid(row=4, column=1, sticky="e",pady=2)

            entry_percent_fix = tk.Entry(set_up_window,width=5, state="disabled")
            entry_percent_fix.insert(0,0)
            entry_percent_fix.grid(row=6, column=1, sticky="e",pady=2)

            # Buttons
            run_button = ttk.Button(set_up_window, text="OK", command=run_algorithm_2,width=10)
            cancel_button = ttk.Button(set_up_window, text="Cancel", command=destroy_2,width=10)
            run_button.grid(row=8, column=1, sticky="e",padx=100)
            cancel_button.grid(row=8, column=1, sticky="e")

        window.title ("Analysis of vaults")
        window.resizable (False, False)
        # Remove minimize and maximize button 
        window.attributes ('-toolwindow',-1)
      
        tooltip = tk.Label(window, text="", relief="solid", borderwidth=1)
        tooltip.place_forget()
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Choose point cloud:",
            "Type of vault:",
            "Space between sections:",
            "Thickness of each section:",
            "Maximum z value for extracting the axis:",
            "Minimum z value for extracting the axis:",
            "Smoothing parameter for the axis extraction:",
            "Expected clearance of the vault:",
            "Configuration of the arch estimator:",
            "Path for saving the data:"
        ]
        row_positions = [0,1,2,3,4,5,6,7,8,9]        
        definition_of_labels_type_1 ("window",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo1=ttk.Combobox (form_frame,values=name_list)
        combo1.grid(row=0,column=1, sticky="e", pady=2)
        combo1.set("Not selected")
        
        types = ["Barrel vault"]
        combo2=ttk.Combobox (form_frame,values=types, state="readonly")
        combo2.current(0)
        combo2.grid(column=1, row=1, sticky="e", pady=2)
        combo2.set("Not selected")
        
        # Entry
        entry_step = tk.Entry(form_frame,width=5)
        entry_step.grid(row=2,column=1, sticky="e",pady=2)
        entry_step.insert(1,self.vault_parameters["step"])
        
        entry_thickness = tk.Entry(form_frame,width=5)
        entry_thickness.grid(row=3,column=1, sticky="e",pady=2)
        entry_thickness.insert(0,self.vault_parameters["thickness"])
        
        entry_zmax = tk.Entry(form_frame,width=5)
        entry_zmax.grid(row=4,column=1, sticky="e",pady=2)
        entry_zmax.insert(0,self.vault_parameters["zmax"])
        
        entry_zmin = tk.Entry(form_frame,width=5)
        entry_zmin.grid(row=5,column=1, sticky="e",pady=2)
        entry_zmin.insert(0,self.vault_parameters["zmin"])  
        
        entry_s = tk.Entry(form_frame,width=5)
        entry_s.grid(row=6,column=1, sticky="e",pady=2)
        entry_s.insert(0,self.vault_parameters["s"])  
        
        entry_e = tk.Entry(form_frame,width=5)
        entry_e.grid(row=7,column=1, sticky="e",pady=2)
        entry_e.insert(0,self.vault_parameters["e"])      
        
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=9, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        # Button
        row_buttons=[8,9]  
        button_names=["...","..."]  
        _=definition_of_buttons_type_1("window",
                                       button_names,
                                       row_buttons,
                                       [lambda: show_set_up_window(self),lambda:save_file_dialog()],
                                       form_frame,
                                       2
                                       )
        
        _=definition_run_cancel_buttons_type_1("window",
                                     [lambda:run_algorithm_1(self, combo1.get(),combo2.get(),float(entry_step.get()),float(entry_thickness.get()),float(entry_zmax.get()),float(entry_zmin.get()),float(entry_e.get()),float(entry_s.get())),lambda:destroy(self)],
                                     10,
                                     form_frame,
                                     1
                                     )

        def run_algorithm_1 (self,training_pc_name,type_of_vault,step,thickness,zmax,zmin,e,s): 
            
            # INITIAL PARAMENTERS
            self.vault_parameters["type_of_vault"]=type_of_vault
            self.vault_parameters["step"]=step
            self.vault_parameters["thickness"]=thickness
            self.vault_parameters["zmax"]=zmax
            self.vault_parameters["zmin"]=zmin
            self.vault_parameters["e"]=e
            self.vault_parameters["s"]=s
            
            # DEFINITION OF THE CROP VOLUME
            thickness_x = self.vault_parameters["thickness"] 
            thickness_y = 99  # Infinite thickness along this axis
            thickness_z = 99  # Infinite thickness along this axis
            
            # PROCESSING THE INITIAL POINT CLOUD
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Preprocessing the input vult...")
            
            name_list=get_point_clouds_name() # Get the name of the selected entities
            
            # Error to prevent the abscene of point cloud
            CC = pycc.GetInstance() 
            type_data, number = get_istance()
            if type_data=='point_cloud' or type_data=='folder':
                pass
            else:
                raise RuntimeError("Please select a folder that contains points clouds or a point cloud")        
            if number==0:
                raise RuntimeError("There are not entities in the folder")
            else:
                entities = CC.getSelectedEntities()[0]
                number = entities.getChildrenNumber()
                
            if type_data=='point_cloud':
                pc_training=entities
            else:
                for ii, item in enumerate(name_list):
                    if item == training_pc_name:
                        pc_training = entities.getChild(ii)
                        break
            pcd=P2p_getdata(pc_training,False,False,True)
            
            # Error control to prevent not algorithm for the training
            if self.vault_parameters=="Not selected":
                raise RuntimeError ("Please select a vault type")
                
            # Creating a point cloud in open3d format
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array([pcd['X'], pcd['Y'], pcd['Z']]).T)
            
            # Calculate the centroid of the point cloud
            centroid = np.asarray(point_cloud.get_center())
            point_cloud_numpy=np.asarray(point_cloud.points)
            
            # EXTRACTION OF THE MAIN AXIS
            initial_point_cloud_z_restriction = point_cloud_numpy[:, 2] # Reduced point cloud according with a criteria (e.g. zmax-zmin)
            point_indices_z_restriction_numpy = np.where(np.logical_and(initial_point_cloud_z_restriction >= self.vault_parameters["zmin"], initial_point_cloud_z_restriction <= self.vault_parameters["zmax"]))[0] #Filter indices of points within the specified z-range   
            reduced_points_z_numpy = np.asarray(point_cloud_numpy)[point_indices_z_restriction_numpy] #Extract the points within the z-range
            cropped_initial_point_cloud_z_restriction = o3d.geometry.PointCloud()
            cropped_initial_point_cloud_z_restriction.points = o3d.utility.Vector3dVector(reduced_points_z_numpy)
            
            progress.setMethodTitle("Extracting the axis of the vault...")  
            
            # Perform DBSCAN clustering
            # Rotation of the point cloud according with the angle
            pc_results_prediction = pycc.ccPointCloud(reduced_points_z_numpy [:,0], reduced_points_z_numpy [:,1], reduced_points_z_numpy [:,2])
            pc_results_prediction.setName("Estimated_axis_of_the_vault")
            
            # Store in the database of CloudCompare
            CC.addToDB(pc_results_prediction)
            CC.updateUI()
            labels = cropped_initial_point_cloud_z_restriction.cluster_dbscan(eps=self.vault_parameters["e"], min_points=self.vault_parameters["min_points"])
            labels= np.asarray (labels)        
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters in labels, ignoring noise if present (-1 is noise)         
                          
            if num_clusters==2:
                
                # Initialize arrays to hold cluster points
                cluster_1 = np.empty((0, 3))
                cluster_2 = np.empty((0, 3))
                
                # Split clusters into separate NumPy arrays
                for cluster_label in range(num_clusters):
                    cluster_indices = np.where(labels == cluster_label)[0]
                    cluster_points = reduced_points_z_numpy[cluster_indices]
                    if cluster_label == 0:
                        cluster_1 = np.vstack((cluster_1, cluster_points))
                    elif cluster_label == 1:
                        cluster_2 = np.vstack((cluster_2, cluster_points)) 
            else:
                raise RuntimeError("It was not possible to determine the axis of the vault. Please review the maximum and mininum height as well as the expected clearance parameters") 
    
            # Calculate total length of the segment
            total_length = np.sum(np.linalg.norm(np.diff(cluster_1, axis=0), axis=1))
            
            # Calculate number of points for 1 cm spacing
            desired_spacing = 10  # in cm
            num_points = int(total_length / desired_spacing)
           
            # Perform linear interpolation between clusters
            interpolated_points_1 = []
            for i in range(len(cluster_1) - 1):
                start = cluster_1[i]
                end = cluster_1[i + 1]
                
                # Generate equally spaced points using linear interpolation
                for j in range(num_points):
                    t = (j + 1) / (num_points + 1)
                    point = start + t * (end - start)
                    interpolated_points_1.append(point)
            
            # Calculate total length of the segment
            total_length = np.sum(np.linalg.norm(np.diff(cluster_2, axis=0), axis=1))
            
            # Calculate number of points for 1 cm spacing
            desired_spacing = 5  # in cm
            num_points = int(total_length / desired_spacing)
            
            # Perform linear interpolation
            interpolated_points_2 = []
            for i in range(len(cluster_2) - 1):
                start = cluster_2[i]
                end = cluster_2[i + 1]
                
                # Generate equally spaced points using linear interpolation
                for j in range(num_points):
                    t = (j + 1) / (num_points + 1)
                    point = start + t * (end - start)
                    interpolated_points_2.append(point)
            
            # Convert interpolated_points_1 into a 2D array
            interpolated_array_1 = np.array(interpolated_points_1)
            
            # Convert interpolated_points_1 into a 2D array
            interpolated_array_2 = np.array(interpolated_points_2)      
            
            
            # Build a KDTree from interpolated_array_2
            tree = cKDTree(interpolated_array_2)
            
            # Empty list to store midpoints
            midpoints = []
            
            # Find nearest neighbors for each point in interpolated_array_1
            for point_1 in interpolated_array_1:
                # Query the KDTree for the nearest neighbor to point_1
                distance, index = tree.query(point_1)
                
                # Nearest neighbor point in interpolated_array_2
                nearest_neighbor = interpolated_array_2[index]
                
                # Calculate the midpoint between point_1 and its nearest neighbor
                midpoint = (point_1 + nearest_neighbor) / 2.0
                midpoints.append(midpoint)
            
            # Convert the list of midpoints into a NumPy array
            midpoints_array = np.array(midpoints)
            
            # Add a third column 'Z' with a constant value which is the average of Z
            z_column = np.full((midpoints_array.shape[0], 1), (self.vault_parameters["zmax"]+self.vault_parameters["zmin"]/2))
            midpoints_with_z = np.hstack((midpoints_array, z_column))
            
            # Extract x, y coordinates from midpoints_with_z
            x_values = midpoints_with_z[:, 0]
            y_values = midpoints_with_z[:, 1]
    
            # Fit the spline function to the points
            
            # Sort x and y together
            inds = x_values.argsort()
            x_sorted = x_values[inds]
            y_sorted = y_values[inds]
            spline = UnivariateSpline(x_sorted, y_sorted,s=self.vault_parameters["s"])
    
            # Generate x values at 1 cm intervals along the curve
            x_curve = np.arange(np.min(x_values), np.max(x_values), 0.01)  # 1 cm interval
    
            # Calculate corresponding y values using the fitted quadratic function
            y_curve=spline(x_curve)
            
            # Create a third column 'Z' with a constant value comprised between the max_Z and min_Z
            z_column_curve = np.full((len(x_curve), 1), (self.vault_parameters["zmax"]+self.vault_parameters["zmin"])/2)
    
            # Combine x_curve, y_curve, and z_column_curve into a 2D array
            points_on_curve_with_z_numpy = np.column_stack((x_curve, y_curve, z_column_curve)) 
            
            # It is neccesary to create the open3d point cloud of the axis in order to apply further transformations
            points_on_curve_with_z=o3d.geometry.PointCloud()
            points_on_curve_with_z.points=o3d.utility.Vector3dVector(points_on_curve_with_z_numpy) 
            
            # Rotation of the point cloud according with the angle
            pc_results_prediction = pycc.ccPointCloud(points_on_curve_with_z_numpy [:,0], points_on_curve_with_z_numpy [:,1], points_on_curve_with_z_numpy [:,2])
            pc_results_prediction.setName("Points used for estimating the axis")
            
            # Store in the database of CloudCompare
            CC.addToDB(pc_results_prediction)
            CC.updateUI()
            
            # LOOP FOR EXTRACTING THE SECTIONS
            
            progress.setMethodTitle("Extracting the sections (arches) of the vault...")              
            

            
            # Initialize parameters
            i=0
            is_inside_upper_bound=True
            is_inside_lower_bound=True
                       
            while is_inside_lower_bound or is_inside_upper_bound:            
                spline = UnivariateSpline(x_curve, y_curve) # Approximation of the points to a spline curve (which is a piecewise polynomial)
                
               
                # Behaviour in the first iteration
                if i==0:
                    initial_x_point=centroid[0]
                    initial_y_point = spline(initial_x_point)
                    x_point=centroid[0]
                    y_point = spline(x_point)
                    
                # Current rotation center and axis 
                rotation_center=(x_point,y_point,centroid[2])
                current_axis=points_on_curve_with_z_numpy [:,:2]
                
                # Derivative of the spline
                spline_derivative = spline.derivative()
                
                # Estimate the slope
                slope_at_point = spline_derivative(x_point)
                
                # Convert the slope to degrees
                angle_in_degrees = math.degrees(math.atan(slope_at_point))
    
                # Rotation of the point cloud in accordance with the local slope of the axus
                final_point_cloud = o3d.geometry.PointCloud()
                final_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
                final_R = point_cloud.get_rotation_matrix_from_axis_angle((0, 0, -math.radians(angle_in_degrees)))
                final_point_cloud.rotate(final_R,center=(rotation_center[0],rotation_center[1],rotation_center[2])) 
                final_point_cloud_numpy = np.asarray(final_point_cloud.points)
                final_point_cloud.points=o3d.utility.Vector3dVector(final_point_cloud_numpy)
    
                
                # Calculate the min and max bounds of the bounding box
                current_min_bound_numpy = rotation_center - np.array([self.vault_parameters["thickness"], thickness_y, thickness_z])
                current_max_bound_numpy = rotation_center + np.array([self.vault_parameters["thickness"], thickness_y, thickness_z])
                
                # Define the bounding box
                bbox = o3d.geometry.AxisAlignedBoundingBox(current_min_bound_numpy, current_max_bound_numpy)
                
                # Crop the point cloud using the bounding box
                cropped_final_point_cloud = final_point_cloud.crop(bbox)          
                
                # Undo the rotation of the section to place the results in the same coordinate system than the initial point cloud
                cropped_point_cloud = o3d.geometry.PointCloud()
                cropped_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(cropped_final_point_cloud.points))
                cropped_final_R = cropped_final_point_cloud.get_rotation_matrix_from_axis_angle((0, 0, math.radians(angle_in_degrees)))
                cropped_point_cloud.rotate(cropped_final_R,center=(rotation_center[0],rotation_center[1],rotation_center[2])) 
                cropped_point_cloud_numpy = np.asarray(cropped_point_cloud.points)
                pc_results_prediction = pycc.ccPointCloud(cropped_point_cloud_numpy[:,0], cropped_point_cloud_numpy[:,1], cropped_point_cloud_numpy[:,2])
                pc_results_prediction.setName("Section_"+str(i))
                
                # Store the results in the database of CloudCompare
                CC.addToDB(pc_results_prediction)
                CC.updateUI() 
                
                # Creation of a pandasframe with the data for the external module
                pcd = pd.DataFrame(cropped_point_cloud_numpy, columns=['X', 'Y', 'Z'])
                
                # Estimation of the arch curve (from external module)
                skeleton=arch_estimation(i,pcd,self.arch_parameters["type_of_arch"],self.arch_parameters["thickness"],self.arch_parameters["ransac_iterations"],self.arch_parameters["ransac_threshold"],self.arch_parameters["ransac_min_samples"],self.arch_parameters["fixed_springing_line"],self.arch_parameters["percent_springing"],self.arch_parameters["load_pc_section"],self.output_directory) 
                
                # Generate the longitudinal axis of the arches if is needed
                if self.arch_parameters["load_pc_section"]:    
                    npc_ske=pycc.ccPointCloud(skeleton[:,0],skeleton[:,1],skeleton[:,2])
                    npc_ske.setName('axis_of_arch_'+str(i))
                    CC.addToDB(npc_ske)  
                
                # For estimating the next x_coordinate i have to split the whole data in data wiht large x and lower x-coordiantes
                lower_group = points_on_curve_with_z_numpy[points_on_curve_with_z_numpy[:, 0] < x_point]
                upper_group = points_on_curve_with_z_numpy[points_on_curve_with_z_numpy[:, 0] >= x_point]            
                if is_inside_upper_bound:
                    reference_group=upper_group
                elif is_inside_upper_bound==False and is_inside_lower_bound:
                    reference_group=lower_group
    
                #Find the nearest point to the x_point that are placed at the upper_group (we are creating section along the positive direciton of the x-axis)
                tree = KDTree(reference_group[:, :1])  ## Create a KDTree with the upper_group points  we're interested in x coordinates
                
                indices = tree.query_ball_point(x_point, self.vault_parameters["step"]) # Find points within the step value.
                
                # Retrieve the actual points
                nearby_points = reference_group[indices]
                
                # Function to calculate distance
                def distance(point1, point2):
                    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                
                # Reference point
                ref_point = (x_point, y_point)
                
                # Finding the farthest point
                farthest_point = None
                min_distance = 9999        
                for point in nearby_points:
                    dist = distance(point, ref_point)
                    error= abs(self.vault_parameters["step"]-dist)
                    if error < min_distance:
                        min_distance = error
                        farthest_point = point
                x_point=farthest_point[0]
                y_point=farthest_point[1]
                if is_inside_upper_bound:     
                    if abs(x_point) - abs (max(points_on_curve_with_z_numpy [:,:1]))<0.001: # To stop one direction
                        is_inside_upper_bound=False
                        x_point=initial_x_point
                        y_point=initial_y_point
                elif is_inside_upper_bound==False and is_inside_lower_bound:
                    if abs (min(points_on_curve_with_z_numpy [:,:1][0]))-abs(x_point)<0.001: # To stop one direction   
                        is_inside_lower_bound=False
                
                # Update the data
                i=i+1
           
            # Stop the progress bar
            progress.stop()           
            print("The process has been finished")
            window.destroy() 
    
#%% START THE GUI        
if __name__ == "__main__":
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_vaults()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()


