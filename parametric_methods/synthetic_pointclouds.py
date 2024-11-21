# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:19:06 2024

@author: psh
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
import time
from PIL import Image, ImageTk

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

config_file=os.path.join(current_directory,r'..\configs\executables.yml')

# Read the configuration from the YAML file for the set-up
with open(config_file, 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)
path_vault = os.path.join(current_directory,config_data['VAULT'])
path_slab = os.path.join(current_directory,config_data['SLAB'])

#%% ADDING assets
current_directory= os.path.dirname(os.path.abspath(__file__))
path_vault_png= os.path.join(current_directory,'..','assets','spc',r"vault_scheme.png")

#%% GUI
class GUI_spc(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
          
        # Vault
        self.set_up_parameters_vault= {
            "initial_radius": 3.0,
            "final_radius": 5.0,
            "lenght": 8.0,
            "height": 3.0,
            "sid": 0.2,
            "n_points": 100,
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
    
    def load_and_display_image(self, path, width, height, parent_widget, column, row, columnspan=1, rowspan=1, sticky="nsew", pady= 1, padx= 1,):
        # Load image using PIL
        image_pil = Image.open(path)

        # Resize image to desired dimensions
        image_pil_resized = image_pil.resize((width, height))

        # Convert image to PhotoImage
        image_tk = ImageTk.PhotoImage(image_pil_resized)

        # Create a label to display the image
        image_label = tk.Label(parent_widget, image=image_tk)
        image_label.image = image_tk
        image_label.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, pady= pady, padx= padx, sticky=sticky)
        
    def main_frame (self, root):    # Main frame of the GUI  
        
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

        tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
        tooltip.place_forget() 
        
        # Function to save and get the output_directory
        def save_file_dialog(tab):
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory                
                if tab ==1:  # Update the entry widget of the tab                   
                    t1_entry_widget.delete(0, tk.END)
                    t1_entry_widget.insert(0, self.output_directory)    
                # elif tab==2:
                #     t2_entry_widget.delete(0, tk.END)
                #     t2_entry_widget.insert(0, self.output_directory)  
                
        # Destroy the window
        def destroy (self): 
            root.destroy ()        
    
        
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Generation of Synthetic Point Clouds")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button 
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")   
        
        tab1 = ttk.Frame(tab_control) # Create 2 tabs
        tab1.pack()
        tab_control.add(tab1, text='Vaults')
        tab2 = ttk.Frame(tab_control) # Create 2 tabs
        tab2.pack() 
        tab_control.add(tab2, text='Slabs')
        tab_control.pack(expand=1, fill="both")
       
        
        # TAB1 = VAULTS      
        
        # Labels
        label_texts = [
            "(1) Initial radius of the vault:",
            "(2) Final radius of the vault:",
            "(3) Lenght of the vault:",
            "(4) Height of the vault:",
            "(5) Step increment dimension:",
            "Number of points:",
            "Choose output directory:",
        ]
        row_positions = [0,1,2,3,4,5,6]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,0)
        
        label_texts = [
            "m",
            "m",
            "m",
            "m",
            "m",
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,2) 

        # Entry
        t1_entry_initial_radius = ttk.Entry(tab1, width=10)
        t1_entry_initial_radius.insert(0,self.set_up_parameters_vault["initial_radius"])
        t1_entry_initial_radius.grid(row=0, column=1, sticky="e", pady=2)
        
        t1_entry_final_radius = ttk.Entry(tab1, width=10)
        t1_entry_final_radius.insert(0,self.set_up_parameters_vault["final_radius"])
        t1_entry_final_radius.grid(row=1, column=1, sticky="e", pady=2)
        
        t1_entry_lenght = ttk.Entry(tab1, width=10)
        t1_entry_lenght.insert(0,self.set_up_parameters_vault["lenght"])
        t1_entry_lenght.grid(row=2, column=1, sticky="e", pady=2)
        
        t1_entry_height = ttk.Entry(tab1, width=10)
        t1_entry_height.insert(0,self.set_up_parameters_vault["height"])
        t1_entry_height.grid(row=3, column=1, sticky="e", pady=2)
        
        t1_entry_sid = ttk.Entry(tab1, width=10)
        t1_entry_sid.insert(0,self.set_up_parameters_vault["sid"])
        t1_entry_sid.grid(row=4, column=1, sticky="e", pady=2)
        
        t1_entry_n_points = ttk.Entry(tab1, width=10)
        t1_entry_n_points.insert(0,self.set_up_parameters_vault["n_points"])
        t1_entry_n_points.grid(row=5, column=1, sticky="e", pady=2)
        
        t1_entry_widget = ttk.Entry(tab1, width=30)
        t1_entry_widget.grid(row=6, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.output_directory)

        # Buttons
        row_buttons=[5]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("tab1",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog(1)],
                                       tab1,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("tab1",
                                     [lambda:run_algorithm_1(self,float(t1_entry_initial_radius.get()),float(t1_entry_final_radius.get()),float(t1_entry_lenght.get()),float(t1_entry_height.get()),float(t1_entry_sid.get()),int(t1_entry_n_points.get())),lambda:destroy(self)],
                                     7,
                                     tab1,
                                     1
                                     )
        
        self.load_and_display_image(path_vault_png, 200, 150, tab1, column=0, columnspan=3, row=8, rowspan=2, sticky="nsew")
        
        # To run the vault tab   
        def run_algorithm_1 (self,initial_radius,final_radius,lenght,height,sid,n_points):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Adding parameters...")
            
            # Update de data
            self.set_up_parameters_vault["initial_radius"]=initial_radius
            self.set_up_parameters_vault["final_radius"]=final_radius
            self.set_up_parameters_vault["lenght"]=lenght
            self.set_up_parameters_vault["height"]=height
            self.set_up_parameters_vault["sid"]=sid
            self.set_up_parameters_vault["n_points"]=n_points
                
            # YAML file
            yaml_vault = {
                'ALGORITHM': "Vault",
                'OUTPUT_DIRECTORY': self.output_directory,
                'CONFIGURATION': {
                    'initial_radius': self.set_up_parameters_vault["initial_radius"],
                    'final_radius': self.set_up_parameters_vault["final_radius"],
                    'lenght': self.set_up_parameters_vault["lenght"],
                    'height': self.set_up_parameters_vault["height"],
                    'sid': self.set_up_parameters_vault["sid"],
                    'n_points': self.set_up_parameters_vault["n_points"]
                }
            }            
            
            write_yaml_file (self.output_directory,yaml_vault)
            
            # RUN THE COMMAND LINE      
            command = path_vault + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            print (command)
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            os.system(command)
            os.chdir(current_directory)
            
            # RUN THE COMMAND LINE
            progress.setMethodTitle("Generating the point cloud(s)...")
            os.system(command)    
            # CREATE THE RESULTING POINT CLOUD 
            progress.setMethodTitle("Openning the point cloud(s)...")
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'point_cloud.txt'), sep=',')  # Use sep='\t' for tab-separated files       
            # # Select only the 'Predictions' column
            pc_result = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_result.setName("Generated_Point_Cloud")
            # idx = pc_result.addScalarField("Classification",pcd_prediction['Predictions']) 
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_result)
            # pc_result.setCurrentDisplayedScalarField(idx)
            # pc_result.getScalarField(pc_result.getScalarFieldIndexByName("Classification")).computeMinAndMax()
            CC.updateUI()
            root.destroy()
            # Revome files
            os.remove(os.path.join(self.output_directory,'point_cloud.txt'))
            
            # Stop the progress_Bar
            progress = pycc.ccProgressDialog()
            print("The process has been finished")

#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        root = tk.Tk()
        app = GUI_spc()
        app.main_frame(root)
        root.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        root.destroy()