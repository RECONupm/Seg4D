# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 19:06:20 2023

@author: Luisja
"""

import cccorelib
import pycc
import os
import sys
import subprocess
import traceback

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import pandas as pd
import numpy as np
import open3d as o3d


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

CC = pycc.GetInstance()
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
params.parentWidget = CC.getMainWindow()
processing_file=os.path.join(current_directory,'anisotropic_denoising-1.0.4\\anisotropic_denoising-1.0.4.exe')
output_file=os.path.join(os.path.dirname(current_directory),'temp\\','output.ply')

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_ad(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
            
    def main_frame (self, window):    # Main frame of the GUI  

        # Function to save and get the output_directory
        def save_file_dialog(tab):
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory 
                
                # Update the entry widget of the tab               
                t1_entry_widget.delete(0, tk.END) 
                t1_entry_widget.insert(0, self.output_directory)   
                
        # Destroy the window
        def destroy (self): 
            window.destroy ()
        
        window.title("Anisotropic denoising")
        # Disable resizing the window
        window.resizable(False, False)
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', 1)
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Select a point cloud:",
            "Choose output directory:"
        ]
        row_positions = [0,1]        
        definition_of_labels_type_1 ("window",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        combo_point_cloud.set("Not selected")
        
        # Entry
        t1_entry_widget = ttk.Entry(form_frame, width=30)
        t1_entry_widget.grid(row=1, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.output_directory)
        
        # Buttons
        row_buttons=[1]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("output",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog(1)],
                                       form_frame,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("window",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get()),lambda:destroy(self)],
                                     2,
                                     form_frame,
                                     1
                                     )
        
        def run_algorithm_1(self,name_list,pc_name):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Denoising the selected point cloud...")
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_name)
            
            # Convert to a pandasdataframe
            pcd_training=P2p_getdata(pc_training,False,True,True)
            
            # Convert X, Y, Z columns to a NumPy array
            points = pcd_training[['X', 'Y', 'Z']].values

            # Pass to open3d geometry
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Write the input file
            o3d.io.write_point_cloud(os.path.join(self.output_directory,pc_name+'.ply'),pcd, write_ascii=True)
            
            # Run the cmd of anisotropic filter 
            command= processing_file + ' --i "' + os.path.join(self.output_directory,pc_name+'.ply') + '" --o "' + os.path.join(self.output_directory,'denoised_'+pc_name+'.ply"')
            os.system(command)
            
            # read the output_file
            pcd = o3d.io.read_point_cloud(os.path.join(self.output_directory,'denoised_'+pc_name+'.ply'))
            
            # Convert Open3D.o3d.geometry.PointCloud to numpy array
            xyz_load = np.asarray(pcd.points)
            point_cloud = pycc.ccPointCloud(xyz_load[:,0], xyz_load[:,1], xyz_load[:,2])
            point_cloud.setName(pc_name+'_denoised')
            CC.addToDB(point_cloud)        
         
            # Update the DB
            CC.updateUI()
            
            # Stop the progress bar
            progress.stop()  
            print('The denosing stage has been finished')
            self.destroy()  # Close the window
                
#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_ad()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()
        