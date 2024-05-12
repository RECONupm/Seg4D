# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:12:02 2024

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

# #CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_checkbutton_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1
#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))

config_file=os.path.join(current_directory,r'..\configs\executables.yml')

# Read the configuration from the YAML file for the set-up
with open(config_file, 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)
path_jakteristics= os.path.join(current_directory,config_data['JAKTERISTICS'])

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_gf(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        self.initial_params = [
            "Eigenvalue sum",
            "Omnivariance",
            "Eigenentropy",
            "Anisotropy",
            "Planarity",
            "Linearity",
            "PCA1",
            "PCA2",
            "Surface Variation",
            "Sphericity",
            "Verticality",
            "Nx",
            "Ny",
            "Nz"
        ]
        self.selected_params = []
        self.values_list=[]
        self.features=[]
        
        self.parameters= {
            "radius": [0.1,0.2,0.4,0.8,1.6,3.2]
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
        
        def save_file_dialog():
            # Abrir el diálogo para seleccionar la ruta de guardado
            directory = filedialog.askdirectory()
            self.output_directory = directory 
            # Mostrar la ruta seleccionada en el textbox correspondiente
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, self.output_directory)
        
        def save_selection():
            selected_params = [self.params[i] for i, var in enumerate(vars_params) if var.get()]
            
            with open("selection.txt", "w") as file:
                file.write("\n".join(selected_params))
            
        # Destroy the window
        def destroy (self): 
            window.destroy ()
        
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        window.title ("Compute geometrical features")
        window.resizable (False, False)     
        window.attributes ('-toolwindow',-1) # Remove minimize and maximize button
        
        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
        
        # Labels
        params = [
            "Eigenvalue sum",
            "Omnivariance",
            "Eigenentropy",
            "Anisotropy",
            "Planarity",
            "Linearity",
            "PCA1",
            "PCA2",
            "Surface Variation",
            "Sphericity",
            "Verticality",
            "Nx",
            "Ny",
            "Nz"
        ]
        
        # Variables to store the state of checkbuttons
        vars_params = [tk.BooleanVar() for _ in range(len(self.initial_params))]

        # Create checkbuttons and associate variables
        for i, param in enumerate(self.initial_params):
            checkbutton = tk.Checkbutton(form_frame, text=param, variable=vars_params[i])
            checkbutton.grid(row=i, column=0, sticky=tk.W)
            
        label_texts=[
            "Select the point cloud:",
            "Search radius (m):",
            "Select a directory for the temporal files:",
        ]
        row_positions = [0,3,6]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,2)
        
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=7, column=2, sticky="w", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        entry_radius = ttk.Entry(form_frame, width=30)
        entry_radius.insert(0, self.parameters["radius"])
        entry_radius.grid(row=4, column=2, sticky="w", pady=2)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list)
        combo_point_cloud.grid(column=2, row=1, sticky="w", pady=2)
        combo_point_cloud.set("Select the point cloud:")
        
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(form_frame, orient="vertical")
        separator.grid(row=0, column=1, rowspan=13, sticky="ns", padx=5)
        
        
        # Buttons
        row_buttons=[7]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("form_frame",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog()],
                                       form_frame,
                                       3
                                       ) 
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get(), str(entry_radius.get())),lambda:destroy(self)],
                                     13,
                                     form_frame,
                                     2
                                     )
        
        def run_algorithm_1 (self,name_list,pc_training_name,radius):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Computing the geometrical features...")
            
            self.selected_params = [param for param, var in zip(self.initial_params, vars_params) if var.get()]
            
            if len(self.selected_params) == 1:
                print("The feature " + str(self.selected_params) + " has been included")
            elif len(self.selected_params)==0:
                print("There are not features selected")
            else:
                print("The features " + str(self.selected_params) + " have been included")
            
            # To adapt the names to the jackteristic library
            selected_features_lower = [feature.lower().replace(' ', '_') if feature not in ['PCA1', 'PCA2'] else feature for feature in self.selected_params]
     
            # To adapt the radius to the jackteristic library
            radius = [float(x) for x in radius.split()]
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            
            # Transform the point cloud into a dataframe and select only the interseting columns
            features_pcd=P2p_getdata(pc_training,False,True,True)
            
            # Save the point cloud with the features selected
            input_path_point_cloud=os.path.join(self.output_directory,"input_point_cloud.txt")
            
            features_pcd[['X', 'Y', 'Z']].to_csv(input_path_point_cloud,sep=' ',header=True,index=False)
            
            # YAML file
            yaml = {
                'ALGORITHM': "Jakterisitcs",
                'INPUT_POINT_CLOUD': input_path_point_cloud,
                'OUTPUT_DIRECTORY': self.output_directory,
                'CONFIGURATION': {
                    'input_features': selected_features_lower,
                    'radius': radius
                    }
                }
            
            write_yaml_file (self.output_directory,yaml)
            
            # RUN THE COMMAND LINE      
            command = path_jakteristics + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory + '"'
            os.system(command)

            # CREATE THE RESULTING POINT CLOUD 
            
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'pc_computed.txt'), sep=' ')  # Use sep='\t' for tab-separated files  
            
            # EXTRACT COLUMN NAMES FOR SCALAR FIELDS
            scalar_fields = list(pcd_prediction.columns[3:])  # Exclude first three columns (X, Y, Z)
            
            # CREATE THE RESULTING POINT CLOUD 
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
                        
            # ADD SCALAR FIELDS
            try:
                # Add scalar fields and compute min and max values
                for field in scalar_fields:
                    idx = pc_training.addScalarField(field, pcd_prediction[field])
                    pc_training.getScalarField(idx).computeMinAndMax()
            except RuntimeError: 
                pass
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.updateUI()
            window.destroy()
            
            # Revome files
            os.remove(os.path.join(self.output_directory, 'input_point_cloud.txt'))
            os.remove(os.path.join(self.output_directory, 'pc_computed.txt'))
            os.remove(os.path.join(self.output_directory, 'algorithm_configuration.yaml'))
            
            # Stop the progress bar
            progress.stop() 
            print("The process has been finished")


#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW 
        window = tk.Tk()
        app = GUI_gf()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()
