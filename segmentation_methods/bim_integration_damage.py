# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:57:38 2024

@author: Digi_2
"""

#%% LIBRARIES
import os
import subprocess
import sys
import yaml

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import traceback
import pandas as pd

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1
#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_bid(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Optimal flow
        self.parameters_bi= {
            "strategy": "Point to family",
            "type_of_damage": "Point to family",
            "point_cloud":"input_point_cloud.txt"
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
        
    def main_frame (self, window):    # Main frame of the GUI  
        
        # FUNCTIONS 
        
        # Function to create tooltip
        def create_tooltip(widget, text):
            widget.bind("<Enter>", lambda event: show_tooltip(text))
            widget.bind("<Leave>", hide_tooltip)

        def show_tooltip(text):
            tooltip.config(text=text)
            tooltip.place(relx=0.5, rely=0.5, anchor="center", bordermode="outside")
            
        def hide_tooltip(event):
            tooltip.place_forget()

        tooltip = tk.Label(window, text="", relief="solid", borderwidth=1)
        tooltip.place_forget()
        
        # Function to save and get the output_directory
        def save_file_dialog(tab):
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory                
                if tab ==1:  # Update the entry widget of the tab                   
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, self.output_directory)
        
        # Destroy the window
        def destroy (self): 
            window.destroy ()
        
        
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        window.title ("BIM Integration")
        window.resizable (False, False)     
        window.attributes ('-toolwindow',-1) # Remove minimize and maximize button
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        label_texts = [
            "Choose point cloud:",
            "Representation strategy",
            "Type of damage",
            "Choose revit project:",
        ]
        row_positions = [0,1,2,3]
        definition_of_labels_type_1 ("window",label_texts,row_positions,form_frame,0) 
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        combo_point_cloud.set("Select the point cloud used for feature selection:")
        
        combobox_insert = [
            ["Point to family","Point to polygon", "Point to patches"],
            ['sqrt','log2'],
            ]
        row_positions = [1,2]
        selected_element = ["Point to family","sqrt"]
        bi_comboboxes =definition_of_combobox_type_1 ("bi",combobox_insert,row_positions, selected_element,form_frame,1)
        
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        row_buttons=[3]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("window",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog(1)],
                                       form_frame,
                                       2
                                       )
        
        _=definition_run_cancel_buttons_type_1("window",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get()),lambda:destroy(self)],
                                     4,
                                     form_frame,
                                     1
                                     )
        # To run the optimal flow   
        def run_algorithm_1 (self,name_list,pc_training_name): 
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            
            # Transform the point cloud into a dataframe and select only the interseting columns
            feature_selection_pcd=P2p_getdata(pc_training,False,True,True)
            
            # Save the point cloud with the features selected
            input_path_point_cloud=os.path.join(self.output_directory,"input_point_cloud.txt")
            feature_selection_pcd[self.features2include].to_csv(input_path_point_cloud,sep=' ',header=True,index=False)
                
            # YAML file
            yaml = {
                'ALGORITHM': "BIM_integration",
                'INPUT_POINT_CLOUD': input_path_point_cloud,                
                'OUTPUT_DIRECTORY': self.output_directory,
                'CONFIGURATION': {
                    'strategy': self.parameters_bi["strategy"],
                    'type_of_damage': self.parameters_bi["type_of_damage"]
                }
            }            
            
            
            write_yaml_file (self.output_directory,yaml)
            
            # # RUN THE COMMAND LINE      
            # command = path_bim_integration + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            # print (command)
            # os.system(command)
            print("The process has been finished")

    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
         
#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_bid()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()