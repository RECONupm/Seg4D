# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:57:56 2023

@author: Luisja



******** IMPORTANT: THE FUNCTION arch_estimation IS USED BY THE MODULE OF VAULTS. ANY CHANGE NEED TO BE CHECKED ALSO IN THIS MODULE
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import os
import sys

import cccorelib
import pycc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import traceback
#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,extract_longitudinal_axis, minBoundingRect, extract_points_within_tolerance
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1
from ransac import RANSAC

#%% ARCH ESTIMATION FUNCTION

def arch_estimation(i,pcd,type_of_arch,thickness,num_iter_ransac,threshold_ransac,d_min,fixed_springs,percent,load_pc_section,save_path): 
    
    # Possible issues with input variables
    if percent >100:
        percent=100
    if percent <0:
        percent=0
        
    # Extract the point cloud in a format to be process, algo the main axis (skeleton) of the arch
    pcd_f,skeleton =extract_points_within_tolerance(pcd[['X','Y','Z']].values, thickness,True)
    if fixed_springs: # In case of having fixed springs
    
        # Calculate the threshold because we chosen the fix springs option
        difference_height=(pcd_f[:,2].max()-pcd_f[:,2].min())*(percent/100)
        threshold_height = difference_height + (pcd_f[:,2].min())
        
        # Filter the DataFrame based on the condition
        filtered_pcd_f = pcd_f[pcd_f[:,2] < threshold_height]
        if len (filtered_pcd_f)<6 or len (filtered_pcd_f)<d_min:
            raise RuntimeError("The threshold is too restrictive. At least one of the arches has less than 6 points or less than the number of minimum points for fitting the model. Please increse the percentage value") 
        
        # Create a istance of RANSAC depending on the type of arch (combo_type.get())
        if type_of_arch=="Pointed arch":                
            midpoint = sum(pcd_f[:,0]) / len(pcd_f[:,0])
        ransac = RANSAC(filtered_pcd_f[:,0],filtered_pcd_f[:,2],num_iter_ransac,d_min,threshold_ransac,type_of_arch,midpoint)  
    else: # In case of not having fixed springs

        # Create a istance of RANSAC depending on the type of arch (combo_type.get())
        ransac = RANSAC(pcd_f[:,0],pcd_f[:,2],num_iter_ransac,d_min,threshold_ransac,type_of_arch)  
    
    # execute ransac algorithm
    _,outliers,inliers=ransac.execute_ransac()
    
    if fixed_springs and type_of_arch=="Circular arch" or type_of_arch=="Quarter arch": # In case of having fixed springs
        
    # initialize the inliers and outliers lists
        inliers=[]
        outliers=[]
        
        # get best model from ransac and store the data for plotting the best fit curve
        a, b, r = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2] 
        
        # compute the error between the whole data and the best model. This model was obtained from the reduced data
        for ii in range(len(pcd_f)):
            dis = np.sqrt((pcd_f[ii,0]-a)**2 + (pcd_f[ii,2]-b)**2)
            if dis >= r:
                distance=dis - r
            else:
                distance= r - dis   
            if distance > threshold_ransac:
                outliers.append(pcd_f[ii,:])
            else:
                inliers.append(pcd_f[ii,:]) 
        
        # Creating a NumPy array to be compatible with the rest of the code
        if len (inliers)>0:
            inliers_array = np.array(inliers)
            inliers = inliers_array [:, [0, 2]]
        if len (outliers)>0:
            outliers_array = np.array(outliers)
            outliers = outliers_array [:, [0, 2]] 
    
    # CREATE THE PLOT
    if len (inliers)>0:
        plt.scatter(inliers[:,0], inliers[:,1],color='g', label='Points consider as inliers')
    if len (outliers)>0:
        plt.scatter(outliers[:,0], outliers[:,1],color='r', label='Points consider as outliers')              
    
    # Create a scatter plot with blue dots for the best fit curve
    plt.scatter(ransac.best_x_coordinates, ransac.best_y_coordinates, c='b', s=10, label="Estimated arch by using RANSAC")
    plt.axis('scaled')
    plt.xlabel('Longitudinal direction')
    plt.ylabel('Vertical direction')        
    plt.title('Section of arch_'+str(i))
    plt.legend()
    plt.grid(True)    
        
    # Save the plot as a PNG file
    if type_of_arch=="Circular arch":
        plt.savefig(save_path+'/circular_arch_'+str(i)+'.png')
    elif type_of_arch=="Pointed arch":
        plt.savefig(save_path+'/pointed_arch_'+str(i)+'.png')
    elif type_of_arch=="Quarter arch":
        plt.savefig(save_path+'/quarter_arch_'+str(i)+'.png')            
   
    # Clear the plot for the next iteration
    plt.clf()     
    return skeleton

#%% GUI
# Create the main window
class GUI_arches(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Parameters
        self.parameters= {
            "tolerance": 0.02,
            "threshold_ransac": 0.05,
            "iterations_ransac": 5000,
            "min_samples": 100,
            "percent_fix": 10
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
    #%% FUNCTIONS
    # MAIN FRAME
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
            
        def toggle_entry_state():
            if checkbox2_var.get():
                label_percent_fix.config(state="normal")  # Make the entry editable
                entry_percent_fix.config(state="normal")
            else:
                label_percent_fix.config(state="disabled")  # Make the entry read-only    
                entry_percent_fix.config(state="disabled")  
                
        window.title("Arch analyzer")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', -1)
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Thickness threshold:",
            "Type of arch:",
            "Threshold value for RANSAC fitting:",
            "Number of iteration for RANSAC fitting:",
            "Minimum number of samples for fitting the model:",
            "Fixed springing line:",
            "Load the points of the main section",
            "Path for saving the data:"
            
        ]
        row_positions = [0,1,2,3,4,5,7,8]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)
        
        label_percent_fix=tk.Label(form_frame, text="Percentage of points to fit the curve from the springs:", state="disabled")
        label_percent_fix.grid(row=6, column=0, sticky="w",pady=2)
        
        
        # Checkboxes
        
        # Variables de control para las opciones
        checkbox1_var = tk.BooleanVar()
        checkbox_1 = tk.Checkbutton(form_frame, variable=checkbox1_var)
        checkbox_1.grid (row=7, column=1, sticky="e",pady=2)
        
        checkbox2_var = tk.BooleanVar()
        checkbox_2 = tk.Checkbutton(form_frame, variable=checkbox2_var, command=toggle_entry_state)
        checkbox_2.grid (row=5, column=1, sticky="e",pady=2)

        # Combobox
        algorithms = ["Circular arch","Pointed arch","Quarter arch"]
        combo_type = ttk.Combobox(form_frame, values=algorithms, state="readonly")
        combo_type.current(0)
        combo_type.grid(row=1, column=1, sticky="e",pady=2)
        
        # Entries
        
        entry_tolerance = ttk.Entry(form_frame,width=5)
        entry_tolerance.insert(0,self.parameters["tolerance"])
        entry_tolerance.grid(row=0, column=1, sticky="e",pady=2)
        
        entry_threshold_ransac = ttk.Entry(form_frame,width=5)
        entry_threshold_ransac.insert(0,self.parameters["threshold_ransac"])
        entry_threshold_ransac.grid(row=2, column=1, sticky="e",pady=2)
        
        entry_iterations_ransac = ttk.Entry(form_frame,width=5)
        entry_iterations_ransac.insert(0,self.parameters["iterations_ransac"])
        entry_iterations_ransac.grid(row=3, column=1, sticky="e",pady=2)
        
        entry_minimum_samples = ttk.Entry(form_frame,width=5)
        entry_minimum_samples.insert(0,self.parameters["min_samples"])
        entry_minimum_samples.grid(row=4, column=1, sticky="e",pady=2)
        
        entry_percent_fix = ttk.Entry(form_frame,width=5, state="disabled")
        entry_percent_fix.insert(0,self.parameters["percent_fix"])
        entry_percent_fix.grid(row=6, column=1, sticky="e",pady=2)
        
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=8, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        # Buttons
        row_buttons=[8]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("form_frame",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog()],
                                       form_frame,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self, str(combo_type.get()),float(entry_tolerance.get()),int(entry_iterations_ransac.get()),float(entry_threshold_ransac.get()),int(entry_minimum_samples.get()),bool(checkbox2_var.get()),entry_percent_fix.get(),bool(checkbox1_var.get()),str(entry_widget.get())),lambda:destroy(self)],
                                     9,
                                     form_frame,
                                     1
                                     )
        
        
        

   
        def run_algorithm_1(self,type_of_arch,thickness,num_iter_ransac,threshold_ransac,d_min,fixed_springs,percent,load_pc_section,save_path):
                
            type_data, number = get_istance()
    
            if percent=='' and fixed_springs:
                raise RuntimeError("Please introduce a value for the percent of points to consider. Example: 10 for 10% of the total points")
            else:
                if percent=='':
                    percent=0
                else:
                    percent=float(percent)
            if type_data=='point_cloud':
                raise RuntimeError("Please select the folder that contains the point clouds")          
            ## EXTRACT THE NUMBER OF CLOUDS IN THE SELECTED FOLDER
            CC = pycc.GetInstance() 
            if number==0:
                raise RuntimeError("There are not entities in the folder")
            else:
                entities = CC.getSelectedEntities()[0]
                number = entities.getChildrenNumber()
           
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Estimating the best fit curve to each arch...")
            
            # Defining the 1% for updating the progress bar
            one_percent = max(1, number // 100)
            
            # LOOP OVER EACH ELEMENT AND PERFORM THE RANSAC
            for i in range(number):
                # For updating the progress bar
                if i % one_percent == 0:
                    progress.update ((i / number) * 100)
                        
             
                # Get the point cloud selected as a pandas frame
                pc = entities.getChild(i)
                pcd=P2p_getdata(pc,False,False,True)        
                skeleton=arch_estimation(i,pcd,type_of_arch,thickness,num_iter_ransac,threshold_ransac,d_min,fixed_springs,percent,load_pc_section,save_path) 
                
                # Generate the arches
                npc=pc.clone()
                npc.setName('arch_'+str(i))
                CC.addToDB(npc)
                
                # Generate the longitudinal axis of the arches if is needed
                if load_pc_section:    
                    npc_ske=pycc.ccPointCloud(skeleton[:,0],skeleton[:,1],skeleton[:,2])
                    npc_ske.setName('axis_of_arch_'+str(i))
                    CC.addToDB(npc_ske)    
           
            # Stop the progress bar
            progress.stop()
            print('The process has been completed!')  
            window.destroy()  # Close the window
   

#%% START THE GUI
if __name__ == "__main__":
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_arches()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()

