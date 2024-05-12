# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:00:57 2023

@author: Luisja
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cccorelib
import pycc

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.optimize import fsolve

from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import os
import sys
import traceback
import warnings

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,extract_longitudinal_axis, minBoundingRect, extract_points_within_tolerance
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% GUI

class GUI_deflection(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Parameters
        self.parameters= {
            "tolerance": 0.02,
            "degree": 4,
            "relative_deflection": 300,
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
#%% FUNCTIONS
    # MAIN FRAME
    def main_frame (self, window):
        
        def save_file_dialog():
            

            directory = filedialog.askdirectory()
            self.output_directory = directory 
            
            # Mostrar la ruta seleccionada en el textbox correspondiente
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, self.output_directory)
            
        def destroy(self):
            window.destroy()  # Close the window
            
        window.title("Analysis of deflections")
        
        # Disable resizing the window
        window.resizable(False, False)
        
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)

        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()


        checkbox1_var = tk.BooleanVar()
        
        # Labels
        label_texts = [
            "Thickness threshold:",
            "Polinomic degree:",
            "Maximum relative deflection (L/300; L/500):",
            "Type of input for the scalar field",
            "Load the points of the main axis",
            "Path for saving the data:"
        ]
        row_positions = [0,1,2,3,4,5]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)

        # Checkbutton
        checkbox_1 = tk.Checkbutton(form_frame, variable=checkbox1_var)
        checkbox_1.grid (row=4, column=1, sticky="e",pady=2)

        # Entries       
        
        entry_tolerance = tk.Entry(form_frame,width=5)
        entry_tolerance.insert(0,self.parameters["tolerance"])
        entry_tolerance.grid(row=0, column=1, sticky="e",pady=2)

        entry_degree = tk.Entry(form_frame,width=5)
        entry_degree.insert(0,self.parameters["degree"])
        entry_degree.grid(row=1,column=1, sticky="e",pady=2)

        entry_relative_deflection = tk.Entry(form_frame,width=5)
        entry_relative_deflection.insert(0,self.parameters["relative_deflection"])
        entry_relative_deflection.grid(row=2, column=1, sticky="e",pady=2)
        
        entry_widget = tk.Entry(form_frame, width=30)
        entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)

        # Combox
        algorithms = ["Data", "Fit"]
        combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
        combo_type.current(0)
        combo_type.grid(row=3, column=1, sticky="e",pady=2)
        
        # Buttons
        row_buttons=[5]  
        button_names=["..."] 
        
        _=definition_of_buttons_type_1("form_frame",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog()],
                                       form_frame,
                                       2
                                       ) 
        
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self, float(entry_tolerance.get()),int(entry_degree.get()),int(entry_relative_deflection.get()),str(combo_type.get())),lambda:destroy(self)],
                                     6,
                                     form_frame,
                                     1
                                     )
        
        
        def run_algorithm_1(self,tolerance,degree,relative_threshold,cal_type):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Calculating the deflection plots of the beams...")
            
            type_data, number = get_istance()
            if type_data=='point_cloud':
                raise RuntimeError("Please select the folder that contains the point clouds")          
            
            ## EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
            CC = pycc.GetInstance() 
            if number==0:
                raise RuntimeError("There are not entities in the folder")
            else:
                entities = CC.getSelectedEntities()[0]
                number = entities.getChildrenNumber()
        
            ## CREATE A EMPTY VARIABLE FOR STORING RESULTS
                data = []
                
                # Defining the 1% for updating the progress bar
                one_percent = max(1, number // 100)
            
            ## LOOP OVER EACH ELEMENT
                for i in range(number):
                    
                    # For updating the progress bar
                    if i % one_percent == 0:
                        progress.update ((i / number) * 100)
                        
                    # Filter out the ComplexWarning
                    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
                    
                    # Get the point cloud selected as a pandas frame
                    pc = entities.getChild(i)
                    pcd=P2p_getdata(pc,False,True,True)
                    pcd_f,skeleton =extract_points_within_tolerance(pcd[['X','Y','Z']].values, tolerance,True)
                    
                    # FIT TO A POLINOMIAL CURVE
                    coefficients = np.polyfit(pcd_f[:,0], pcd_f[:,2], degree)
                    curve = np.poly1d(coefficients)
                    
                    # Find the inflection point (second derivative equal to 0)
                    second_derivative= np.polyder (curve,2)
                    second_derivative_roots=np.roots(second_derivative)
                    filter_arr_2 = []
                    
                    for element in second_derivative_roots:
                      if element>min(pcd_f[:,0]) and element<max(pcd_f[:,0]):
                        filter_arr_2.append(True)
                      else:
                        filter_arr_2.append(False)
                    second_derivative_roots_filtered = second_derivative_roots[filter_arr_2]
                    z_second_derivative_roots_filtered = [curve(x) for x in second_derivative_roots_filtered]
                    
                    
                    ## PLOTTING
                    
                    # Generate points on the curve for plotting
                    x_curve = np.linspace(pcd_f[:,0].min(), pcd_f[:,0].max(), 100)
                    z_curve = curve(x_curve)
                   
                    # Find the corresponding x values for the maximum and minimum z points
                    x_max_z_data = pcd_f[:,0][np.argmax(pcd_f[:,2])]
                    x_min_z_data = pcd_f[:,0][np.argmin(pcd_f[:,2])]
                    x_max_z_fit = x_curve[np.argmax(z_curve)]
                    x_min_z_fit = x_curve[np.argmin(z_curve)]
                    
                    # Find the maximum and minimum z values in the original data
                    max_z_data = np.max(pcd_f[:,2])
                    min_z_data = np.min(pcd_f[:,2])
                    
                    # Find the maximum and minimum z values from the polynomial fitting
                    max_z_fit = np.max(z_curve)
                    min_z_fit = np.min(z_curve)
                    
                    # Calculate the distances along the x-axis
                    x_distance = pcd_f[:,0].max() - pcd_f[:,0].min()
                    
                    # Calculate the distances between the maximum and minimum z points
                    z_distance_data = max_z_data - min_z_data
                    z_distance_fit = max_z_fit - min_z_fit
                    
                    #Calculate the relative deflection
                    print ("x_distance")
                    print (x_distance)
                    print ("z_distance")
                    print (z_distance_data)
                    Relative_data=x_distance/z_distance_data
                    print ("Relative_data")
                    print (Relative_data)
                    print ("x_distance")
                    print (x_distance)
                    print ("relative_threshold")
                    print (relative_threshold)
                    Maximum_deflection=x_distance/relative_threshold
                    print ("Max_deflect")
                    print (Maximum_deflection)
                    Relative_fit=x_distance/z_distance_fit
                    print ("x_distance")
                    print (x_distance)
                    print ("z_distance_fit")
                    print (z_distance_fit)
                    print ("Relative_fit")
                    print (Relative_fit)                 
                    # Create the plot
                    # Define a function for formatting the tick labels
                    def format_ticks(value, pos):
                        return "{:.2f}".format(value)
                    
                    plt.scatter(pcd_f[:,0], pcd_f[:,2], label='Data Points')
                    plt.plot(x_curve, z_curve, 'r', label='Polynomial Curve')
        
        
                    plt.scatter(x_min_z_data, min_z_data, color='green', marker='o', label='Min Z (Data)')
                    plt.scatter(x_max_z_data, max_z_data, color='blue', marker='o', label='Max Z (Data)')
                    plt.scatter(x_min_z_fit, min_z_fit, color='yellow', marker='o', label='Min Z (Fit)')
                    plt.scatter(x_max_z_fit, max_z_fit, color='purple', marker='o', label='Max Z (Fit)')
                    plt.scatter(second_derivative_roots_filtered, z_second_derivative_roots_filtered, color='red', marker='o', label='Inflection point')
                    
                    plt.xlabel('longitudinal direction')
                    plt.ylabel('vertical direction')
                    plt.title('Deflection analysis of Beam_'+str(i))
                    
                    # Apply the custom tick formatter to both x and y axes
                    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
                    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
                    
                    plt.legend()
                    plt.grid(True)
                    # Save the plot as a PNG file
                    plt.savefig(self.output_directory +'/Beam_'+str(i)+'.png')
                    # Clear the plot for the next iteration
                    plt.clf()
        
                    # Check if relative defleciton of the beam in accordace to the data or the curve fitting
                    if z_distance_data<=Maximum_deflection:
                        arr = np.full((len(pcd),), 0)
                        verified_data= True
                        arr_1=np.full((len(pcd),), z_distance_data)
                        arr_2=np.full((len(pcd),), Relative_data)
                    else:
                        arr = np.full((len(pcd),), 1)
                        verified_data= False
                        arr_1=np.full((len(pcd),), z_distance_data)
                        arr_2=np.full((len(pcd),), Relative_data)

                    if z_distance_fit<=Maximum_deflection:
                        arr = np.full((len(pcd),), 0)
                        verified_fit= True
                        arr_1=np.full((len(pcd),), z_distance_fit)
                        arr_2=np.full((len(pcd),), Relative_fit)
                    else:
                        arr = np.full((len(pcd),), 1)
                        verified_fit= False
                        arr_1=np.full((len(pcd),), z_distance_fit)
                        arr_2=np.full((len(pcd),), Relative_fit)    
                    
                    # Store the data as a tuple
                    data.append(('Beam_'+str(i),x_distance, z_distance_data, z_distance_fit, x_min_z_data-min(pcd_f[:,0]), x_min_z_fit-min(pcd_f[:,0]),z_second_derivative_roots_filtered,second_derivative_roots_filtered,Relative_data,Relative_fit,verified_data,verified_fit))
                    npc=pc.clone()
                    npc.setName('Beam_'+str(i))
                    CC.addToDB(npc)
                    idx=npc.addScalarField("Is deflected", arr)   
                    npc.getScalarField(idx).computeMinAndMax()
                    idx=npc.addScalarField("Relative_deflection", arr_2)   
                    npc.getScalarField(idx).computeMinAndMax()
                    idx=npc.addScalarField("Maximum deflection", arr_1)
                    npc.getScalarField(idx).computeMinAndMax()
                    
                    if checkbox1_var.get():    
                        npc_ske=pycc.ccPointCloud(skeleton[:,0],skeleton[:,1],skeleton[:,2])
                        npc_ske.setName('Skeleton_of_Beam_'+str(i))
                        CC.addToDB(npc_ske)
                    
                    # Reset the warning filter to its default state
                    warnings.resetwarnings()
                    
                
                # Open the file in write mode
                with open(self.output_directory +'/deflection_analysis.txt', 'w') as file:
                
                    # Write the header
                    file.write("Identifier\tLength\tDeflection from point data\tDeflection from polynomial data\tDistance to maximum deflection point from point data\tDistance to maximum deflection point from polynomial data\tInflection points (vertical coordinates)\tInflection points (horizontal coordinates)\tRelative deflection from point data\tRelative deflection from polynomial data\tIs within the relative deflection tolerante accoring to the data from the points?\tIs within the relative deflection tolerante accoring to the data from the curve fitting?\n")
                    
                    # Write the data to the file
                    for item in data:
                        file.write(f"{item[0]}\t{item[1]:.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t{item[6]}\t{item[7]}\t{item[8]:.3f}\t{item[9]:.3f}\t{item[10]}\t{item[11]}\n")
                print('The process has been finished')  
            
            # Stop the progress_Bar
            progress = pycc.ccProgressDialog()
            window.destroy()  # Close the window

#%% START THE GUI
if __name__ == "__main__":
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_deflection()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()