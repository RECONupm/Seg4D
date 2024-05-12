# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:27:04 2023

@author: Luisja
"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cccorelib
import pycc
import os
import pandas as pd
import numpy as np


from scipy.spatial import ConvexHull
from scipy.stats import linregress
import math
import cv2
import matplotlib.pyplot as plt
import os
import sys
from sklearn import datasets, linear_model

import traceback

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% GUI
# Create the main window
class GUI_inclination(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Parameters
        self.parameters= {
            "tolerance": 0.02,
            "step": 0.5,
            "max_inclination": 2,
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
    def main_frame (self,window):
        
        def save_file_dialog():
            directory = filedialog.askdirectory()
            self.output_directory = directory 
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, self.output_directory)
        
        def destroy(self):
            window.destroy()  # Close the window

        window.title("Analysis of inclinations")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)

        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Thickness threshold:",
            "Step between sections:",
            "Maximum inclination allowed:",
            "Type strategy used to compute the center of gravity",
            "Path for saving the data:"
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)
        
        # Entries
        entry_tolerance = tk.Entry(form_frame,width=5)
        entry_tolerance.grid(row=0, column=1, sticky="e",pady=2)
        entry_tolerance.insert(0,self.parameters["tolerance"])
        
        entry_step = tk.Entry(form_frame,width=5)
        entry_step.grid(row=1,column=1, sticky="e",pady=2)
        entry_step.insert(0,self.parameters["step"])
        
        entry_maximum_inclination = tk.Entry(form_frame,width=5)
        entry_maximum_inclination.grid(row=2,column=1, sticky="e",pady=2)
        entry_maximum_inclination.insert(0,self.parameters["max_inclination"])
        
        algorithms = ["Points","Convex Hull","Min. bound rectangle", "Circle fitting"]
        combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly",width=15)
        combo_type.current(0)
        combo_type.grid(row=3, column=1, sticky="e",pady=2)
        
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=4, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        # Buttons
        row_buttons=[4]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("form_frame",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog()],
                                       form_frame,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self,float(entry_tolerance.get()),float(entry_step.get()),float (entry_maximum_inclination.get()),str(combo_type.get())),lambda:destroy(self)],
                                     6,
                                     form_frame,
                                     1
                                     )

    
        def run_algorithm_1(self,tolerance,step,limit,cal_type):        
            
            def calculate_polygon_area(points):
                """
                This function allow to calculate the area of a polygon based on a set of points
                    
                Parameters:
                ----------
                
                points (numpy array nx2): set of points for estimating the polygon area
                            
                Returns:
                ----------
                
                area (float): area of the polygon
                
                """            
                n = len(points)
                area = 0
                for i in range(n):
                    x1, y1 = points.iloc[i]
                    x2, y2 = points.iloc[(i + 1) % n]
                    area += (x1 * y2 - x2 * y1)
                area = abs(area) / 2
                return area
            def total_angle_from_x_y_angles(angle_x, angle_y):
                """
                This function allow to calculate the angle by considering angle along X-axis and Y-axis
                    
                Parameters:
                ----------
                
                angle x (float): angle in degrees along X-axis
                
                angle y (float): angle in degrees along Y-axis
                            
                Returns:
                ----------
                
                total_angle_deg (float): composed angle
                
                """   
                # Convert angles to radians
                angle_x_rad = math.radians(angle_x)
                angle_y_rad = math.radians(angle_y)
            
                # Assuming the length of the vector along each axis (before rotation) is 1
                # Calculate the components of the vector
                x = math.cos(angle_y_rad)
                y = math.cos(angle_x_rad)
                # The z component can be calculated considering the vector initially along Z-axis
                # and then rotated along X and Y axes
                z = math.sqrt(1 - x**2) * math.sqrt(1 - y**2)
            
                # Calculate the magnitude of the vector
                magnitude = math.sqrt(x**2 + y**2 + z**2)
            
                # Calculate the total angle with respect to the origin
                total_angle_rad = math.acos(z / magnitude)
            
                # Convert the total angle back to degrees
                total_angle_deg = math.degrees(total_angle_rad)
            
                return total_angle_deg
            
            # EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
            type_data, number = get_istance()
            
            if type_data=='point_cloud':
                raise RuntimeError("Please select the folder that contains the point clouds")          
            
            CC = pycc.GetInstance() 
            if number==0:
                raise RuntimeError("There are not entities in the folder")
            else:
                entities = CC.getSelectedEntities()[0]
                number = entities.getChildrenNumber()
            
            ## EXTRACTION OF SECTIONS
                data = []
                
                # Start the progress bar
                progress = pycc.ccProgressDialog()
                progress.start()
                progress.setMethodTitle("Estimating the inclination of each element...")
                
                # Defining the 1% for updating the progress bar
                one_percent = max(1, number // 100)  
            
            # Loope each element
                for i in range(number):
                                        
                    # For updating the progress bar
                    if i % one_percent == 0:
                        progress.update ((i / number) * 100)
                        
                    pc = entities.getChild(i)
                    pcd=P2p_getdata(pc,False, True, True)                 
                    j=step
                    # Create an empty list to store the data
                    data_element = [] # for storing the data of each section
                    while j<max (pcd['Z']):# For stoping at the maximum z
                        if j>=min (pcd['Z']):# For starting at the minimum z                        
                            upper_bound= j + tolerance
                            lower_bound= j - tolerance
                            section= pcd[(pcd['Z'] >= lower_bound) & (pcd['Z'] <= upper_bound)]
                            section_f=section[['X','Y']] # Deleting the Z coordinate for processing
                            if cal_type=="Points":
                                area= calculate_polygon_area(section_f[['X', 'Y']])
                                centroid_x = section_f['X'].mean()
                                centroid_y = section_f['Y'].mean()        
                            elif cal_type=="Convex Hull":
                                hull=ConvexHull(section_f[['X', 'Y']].values)                            
                                hull_indices = hull.vertices # Extract the indices of the convex hull vertices                            
                                hull_points = section_f[['X', 'Y']].values[hull_indices] # Calculate the area of the polygon defined by the convex hull  
                                area = ConvexHull(hull_points).area
                                centroid_x = np.mean(hull.points[:,0])
                                centroid_y = np.mean(hull.points[:,1])
                            elif cal_type=="Min. bound rectangle": # Fit the points to a rectangle                            
                                rect = cv2.minAreaRect(section_f[['X', 'Y']].values)                            
                                centroid_x, centroid_y = rect[0] # Extract the centroid and area of the fitted rectangle    
                                area = rect[1][0] * rect[1][1]
                                box=cv2.boxPoints(rect)
                            elif cal_type=="Circle fitting": # Fit the points to a circle                            
                                (centroid_x, centroid_y), radius = cv2.minEnclosingCircle(section_f[['X', 'Y']].values)                            
                                area = np.pi * radius**2
        
                            # Append the data as a dictionary to the list
                            data_element.append((j,centroid_x, centroid_y,area))
                            
                            # WRITE AND PLOT THE IMAGE OF EACH CROSS SECTION
                            
                            figure, axes = plt.subplots()
                            axes.set_aspect( 1 )
                            
                            # Create the plot along x-axis                        
                            plt.scatter(section_f['X'].values, section_f['Y'].values, label='Data Points')
                            
                            # Plot the centroid point
                            plt.scatter(centroid_x, centroid_y, color='red', marker='x', label='Centroid')
                           
                            #Plot curves in case of chosing it.
                            if cal_type =='Points':
                                pass
                            elif cal_type=='Circle fitting':
                                Circle=plt.Circle((centroid_x, centroid_y), radius, fill=False, color='green')
                                axes.add_artist(Circle)
                                plt.xlim(centroid_x-radius*2 , centroid_x+radius*2 )
                                plt.ylim(centroid_y-radius*2 ,centroid_y+radius*2 )
                            elif cal_type=='Rectangle fitting':
                                # We plot by means of the four corners since it is easier
                                # Extract x and y coordinates from the points
                                x_coords, y_coords = zip(*box)
                                x_coords = list(x_coords)  # Convert to a list
                                y_coords = list(y_coords)  # Convert to a list
                                # Plot the rectangle by connecting the points with lines
                                axes.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], marker='o', linestyle='-', color='g')
                            elif cal_type=="Convex Hull":
                                # Plot the convex hull polygon
                                for simplex in hull.simplices:
                                    axes.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'g-')                        
                            plt.xlabel('x-axis')
                            plt.ylabel('y-axis')
                            plt.title('Centroid estimation of Element_'+str(i) + ' at height ' + str(j))
                            plt.legend(loc="upper right")
                            plt.grid(True)
                            
                            # Save the plot as a PNG file
                            newdir= self.output_directory +'/Element_'+str(i)+'_sections'
                            # Check if the folder already exists
                            if not os.path.exists(newdir):
                            # If it doesn't exist, create it
                                os.mkdir(newdir)
                            j=round (j,2)
                            plt.savefig(newdir +'/Element_'+str(i)+ ' at height ' + str(j) + '.png')
                            # Clear the plot for the next iteration
                            plt.clf()                     
                        j= j + step #Updating the Step
                        
                    # WRITE FILE WITH THE GENERAL DATA
                    with open(self.output_directory +'/Element_'+str(i)+'.txt', 'w') as file:
                    # Write the header
                        file.write("Height\tCentroid along x-axis\tCentroid along y-axis\tArea\n")                    
                        # Write the data to the file
                        for item in data_element:
                            file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\n")
                    
                    
                    #Best fit line from j and centroid_x
                    column_0 = [row[0] for row in data_element]  # List comprehension to extract the first column            
                    column_1 = [row[1] for row in data_element]  # List comprehension to extract the first column
                    min_value = min(column_1)
                    
                    # Normalize the values in column_1
                    normalize_c_1 = [x - min_value for x in column_1]    
                    column_2 = [row[2] for row in data_element]  # List comprehension to extract the second column
                    min_value = min(column_2)
                    
                    # Normalize the values in column_1
                    normalize_c_2 = [x - min_value for x in column_2]
                    
                    # Along x-axis
                    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(normalize_c_1, column_0)
                    angle_rad_x = math.atan(slope_x)  # Calculate the angle in radians
                    angle_deg_x = 90-math.degrees(angle_rad_x)  # Convert the angle to degrees
                    
        
                    # Along y-axis
                    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(normalize_c_2, column_0)
                    angle_rad_y = math.atan(slope_y)  # Calculate the angle in radians
                    angle_deg_y = 90-math.degrees(angle_rad_y)  # Convert the angle to degrees
                    
                                   
                    # Extract the first column (column 0) and calculate the mean
                    area_element = [row[3] for row in data_element]  # List comprehension to extract the first column
                    mean_area_element = sum(area_element) / len(area_element)
                    #PRINT FIGURES FOR EACH SECTION
                    
                    # Create the plot along x-axis
                    plt.scatter(normalize_c_1, column_0, label='Data Points')
                    
                    # Generate x values for the best-fit line
                    best_fit_x = np.linspace(min(normalize_c_1), max(normalize_c_1), 100)
                    
                    # Calculate corresponding y values for the best-fit line
                    best_fit_y = slope_x * best_fit_x + intercept_x
                    plt.plot(best_fit_x, best_fit_y, 'r', label='Best fit line y='+str(round(slope_x,2))+'x + '+str(round(intercept_x,2)))    
                  
                    plt.xlabel('deviation of the center of gravity along x-axis')
                    plt.ylabel('height')
                    plt.title('Inclination analysis along x-axis of Element_'+str(i))
                    plt.legend()
                    plt.grid(True)
                    
                    # Save the plot as a PNG file
                    plt.savefig(self.output_directory +'/Element_'+str(i)+'_x_axis.png')
                    
                    # Clear the plot for the next iteration
                    plt.clf()
                    
                    # Create the plot along y-axis
                    plt.scatter(normalize_c_2, column_0, label='Data Points')
                    
                    # Generate x values for the best-fit line
                    best_fit_xx = np.linspace(min(normalize_c_2), max(normalize_c_2), 100)
                    
                    # Calculate corresponding y values for the best-fit line
                    best_fit_yy = slope_y * best_fit_xx + intercept_y
                    plt.plot(best_fit_xx, best_fit_yy, 'r', label='Best fit line y='+str(round(slope_y,2))+'x + '+str(round(intercept_y,2)))    
        
                  
                    plt.xlabel('deviation of the center of gravity along y-axis')
                    plt.ylabel('height')
                    plt.title('Inclination analysis along y-axis of Element_'+str(i))
                    plt.legend()
                    plt.grid(True)
                    
                    # Save the plot as a PNG file
                    plt.savefig(self.output_directory +'/Element_'+str(i)+'_y_axis.png')
                    
                    # Clear the plot for the next iteration
                    plt.clf()  
                    
                    # empty list to store new elements
                    data_element = []
                    # Cheking if the current inclination is lower that the maximum one
                    # Firsly we need to convert the angles properly
                    if angle_deg_x>90:
                        angle_deg_x=180-angle_deg_x
                    if angle_deg_y>90:
                        angle_deg_y=180-angle_deg_y
                        
                    # Definition of arg1,arg,2,arg3 that will be introduced as scalar fields in the point cloud
                    arg_1=np.full((len(pcd),), angle_deg_x)    
                    arg_2=np.full((len(pcd),), angle_deg_y)
                    arg_5=np.full((len(pcd),), total_angle_from_x_y_angles(angle_deg_x, angle_deg_y))
                    if angle_deg_x>=limit:
                         safe_x=False
                         arg_3=np.full((len(pcd),), 1)
                    else:
                         safe_x=True
                         arg_3=np.full((len(pcd),), 0)
                    if angle_deg_y>=limit:
                         safe_y=False
                         arg_4=np.full((len(pcd),), 1)
                    else:
                         safe_y=True
                         arg_4=np.full((len(pcd),), 0)
                    
                    data.append(('Element_'+str(i),max(pcd['Z'])-min(pcd['Z']),mean_area_element,angle_deg_x,angle_deg_y,total_angle_from_x_y_angles(angle_deg_x, angle_deg_y),safe_x,safe_y))            
                    
                    # NEW POINTS CLOUD WITH THE SCALAR FIELDS
                    numpy_cloud=pc.points()  
                    npc=pycc.ccPointCloud(numpy_cloud[:,0],numpy_cloud[:,1],numpy_cloud[:,2])
                    npc.setName('Element_'+str(i))
                    CC.addToDB(npc)
                    idx= npc.addScalarField("Excesive inclination along x-axis", arg_3)
                    npc.getScalarField(idx).computeMinAndMax()
                    idx= npc.addScalarField("Deflection along x-axis", arg_1)  
                    npc.getScalarField(idx).computeMinAndMax()
                    idx= npc.addScalarField("Excesive inclination along y-axis", arg_4) 
                    npc.getScalarField(idx).computeMinAndMax()
                    idx= npc.addScalarField("Deflection along y-axis", arg_2)  
                    npc.getScalarField(idx).computeMinAndMax()
                    idx= npc.addScalarField("Composed angle (x and y angles)", arg_5) 
                    npc.getScalarField(idx).computeMinAndMax()
                    
                    #WRITE FILE WITH THE GENERAL DATA
                     
                    # Open the file in write mode
                    with open(self.output_directory +'/inclination_analysis.txt', 'w') as file:
                     # Write the header
                         file.write("Identifier\tLenght\tMean area\tInclination angle along x-axis\tInclination angle along y-axis,\tComposed angle (x and y angles),\tIs within the inclination tolerante along x-axis?\tIs within the inclination tolerante along y-axis?\n")
                        
                    # Write the data to the file
                         for item in data:
                             file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t{item[6]}\t{item[7]}\n")   
            # Stop the progress bar
            progress.stop()
            print('The process has been finished')
            window.destroy()  # Close the window    

        
#%% START THE GUI        
if __name__ == "__main__":
    try:        
        window = tk.Tk()
        app = GUI_inclination()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()