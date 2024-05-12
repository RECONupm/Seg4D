# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:00:29 2023

@author: Luisja
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import cccorelib
import pycc
import colorsys
import os
import sys
import traceback

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)
additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

CC = pycc.GetInstance()

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI

class GUI_cc(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)      
        self.output_directory=os.getcwd() # Directory to save the files (output)
    
    # Convert RGB to HSV function
    def rgb_to_hsv(self,row):
        try:
            r, g, b = row['R'], row['G'], row['B']
        except KeyError:
            raise KeyError("One or more of the layers ('R', 'G', 'B') are not in the scaler field of the point cloud")
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return pd.Series([h, s, v])
    
    # Convert RGB to YCBCR function
    def rgb_to_ycbcr(self,row):
        try:
            r, g, b = row['R'], row['G'], row['B']
        except KeyError:
            raise KeyError("One or more of the layers ('R', 'G', 'B') are not in the scaler field of the point cloud")
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        return pd.Series([y, cb, cr])
    
    # Convert RGB to YIQ function
    def rgb_to_yiq(self,row):
        try:
            r, g, b = row['R'], row['G'], row['B']
        except KeyError:
            raise KeyError("One or more of the layers ('R', 'G', 'B') are not in the scaler field of the point cloud")
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b
        q = 0.211 * r - 0.523 * g + 0.312 * b
        return pd.Series([y, i, q])
    
    # Convert RGB to YUV function
    def rgb_to_yuv(self,row):
        try:
            r, g, b = row['R'], row['G'], row['B']
        except KeyError:
            raise KeyError("One or more of the layers ('R', 'G', 'B') are not in the scaler field of the point cloud")
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        return pd.Series([y, u, v])
    
    # Create Scalar fields for the different color functions
    def color_conversion (self,selected_algorithms,pc,pcd):
         for algorithm in selected_algorithms:
             print(f"Coverting to: {algorithm}")
             if algorithm == 'HSV':
                 
                 pcd[['H(HSV)', 'S(HSV)', 'V(HSV)']] = pcd.apply(self.rgb_to_hsv, axis=1) # Add data to the dataframe
                 pc.addScalarField("H(HSV)", pcd['H(HSV)'])                 
                 pc.addScalarField("S(HSV)", pcd['S(HSV)'])
                 pc.addScalarField("V(HSV)", pcd['V(HSV)'])
                 
                 # To visualize properly the Scalar field
                 pc.getScalarField(pc.getScalarFieldIndexByName("H(HSV)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("S(HSV)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("V(HSV)")).computeMinAndMax()
             elif algorithm == 'YCbCr':            
                 ## ADD YCBCR TO THE DATAFRAME
                 pcd[['Y(YCbCr)', 'Cb(YCbCr)', 'Cr(YCbCr)']] = pcd.apply(self.rgb_to_ycbcr, axis=1) # Add data to the dataframe
                 pc.addScalarField("Y(YCbCr)", pcd['Y(YCbCr)'])
                 pc.addScalarField("Cb(YCbCr)", pcd['Cb(YCbCr)'])
                 pc.addScalarField("Cr(YCbCr)", pcd['Cr(YCbCr)']) 
                 
                 # To visualize properly the Scalar field
                 pc.getScalarField(pc.getScalarFieldIndexByName("Y(YCbCr)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("Cb(YCbCr)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("Cr(YCbCr)")).computeMinAndMax()
             elif algorithm == 'YIQ': 
                 pcd[['Y(YIQ)', 'I(YIQ)', 'Q(YIQ)']] = pcd.apply(self.rgb_to_yiq, axis=1) # Add data to the dataframe
                 pc.addScalarField("Y(YIQ)", pcd['Y(YIQ)'])
                 pc.addScalarField("I(YIQ)", pcd['I(YIQ)'])
                 pc.addScalarField("Q(YIQ)", pcd['Q(YIQ)'])
                 
                 # To visualize properly the Scalar field
                 pc.getScalarField(pc.getScalarFieldIndexByName("Y(YIQ)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("I(YIQ)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("Q(YIQ)")).computeMinAndMax()
             elif algorithm == 'YUV': 
                 pcd[['Y(YUV)', 'U(YUV)', 'V(YUV)']] = pcd.apply(self.rgb_to_yuv, axis=1) # Add data to the dataframe
                 pc.addScalarField("Y(YUV)", pcd['Y(YUV)'])
                 pc.addScalarField("U(YUV)", pcd['U(YUV)'])
                 pc.addScalarField("V(YUV)", pcd['V(YUV)']) 
                 
                 # To visualize properly the Scalar field
                 pc.getScalarField(pc.getScalarFieldIndexByName("Y(YUV)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("U(YUV)")).computeMinAndMax()
                 pc.getScalarField(pc.getScalarFieldIndexByName("V(YUV)")).computeMinAndMax()
    
    def main_frame (self,window):
        
        def destroy(self):
            window.destroy()  # Close the window
  
        window.title("Color maps")        
        window.resizable(False, False) # Disable resizing the window        
        window.attributes('-toolwindow', 1) # Remove minimize and maximize buttons (title bar only shows close button)       
        form_frame = tk.Frame(window, padx=10, pady=10)  # Create a frame for the form
        form_frame.pack()
    
        # Control variables
        algorithm1_var = tk.BooleanVar()
        algorithm2_var = tk.BooleanVar()
        algorithm3_var = tk.BooleanVar()
        algorithm4_var = tk.BooleanVar()
    
        # Labels
        label_texts = [
            "Select a point cloud:",
            "HSV (Hue-Saturation-Value)",
            "YCbCr (Luma-Chrominance)",
            "YIQ (Brightness-Chrominance-Quadr-phase chrominance)",
            "YUV (Brightness-Chrominance"
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        combo_point_cloud.set("Not selected")
    
        # Checkboxes
        algorithm1_checkbox = tk.Checkbutton(form_frame, variable=algorithm1_var)
        algorithm1_checkbox.grid(row=1, column=1, sticky="e")
        
        algorithm2_checkbox = tk.Checkbutton(form_frame, variable=algorithm2_var)
        algorithm2_checkbox.grid(row=2, column=1, sticky="e")       
        
        algorithm3_checkbox = tk.Checkbutton(form_frame, variable=algorithm3_var)
        algorithm3_checkbox.grid(row=3, column=1, sticky="e")       
        
        algorithm4_checkbox = tk.Checkbutton(form_frame, variable=algorithm4_var)
        algorithm4_checkbox.grid(row=4, column=1, sticky="e")        
        
        # Buttons
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get(),bool(algorithm1_var.get()),bool(algorithm2_var.get()),bool(algorithm3_var.get()),bool(algorithm4_var.get())),lambda:destroy(self)],
                                     5,
                                     form_frame,
                                     1
                                     )
            
        def run_algorithm_1(self,name_list,pc_name,hsv,ycbcr,yiq,yuv):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Generating the new color scales...")
            
                            
            selected_algorithms = []          
            if hsv:
                selected_algorithms.append("HSV")            
            if ycbcr:
                selected_algorithms.append("YCbCr")            
            if yiq:
                selected_algorithms.append("YIQ")                
            if yuv:
                selected_algorithms.append("YUV") 
                
            # Check if the selection is a point cloud
            pc=check_input(name_list,pc_name)                
            # Convert to a pandasdataframe
            pcd=P2p_getdata(pc,False,True,True)    
            self.color_conversion(selected_algorithms,pc,pcd)
            CC.addToDB(pc)
            CC.updateUI() 
               
            # Stop the progress bar
            progress.stop()   
            print('The color scales has been added to the scalar fields of the point cloud')  
            self.destroy()  # Close the window

    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
        

#%% START THE GUI        
if __name__ == "__main__":
    try:        
        window = tk.Tk()
        app = GUI_cc()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()