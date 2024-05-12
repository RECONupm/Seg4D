# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:58:33 2024

@author: Digi_2
"""

#%% LIBRARIES
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import yaml

import cccorelib
import pycc
import os
import subprocess

import pandas as pd
import numpy as np

import os
import sys
import traceback


#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,extract_longitudinal_axis, minBoundingRect, extract_points_within_tolerance
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% INITAL OPERATIONS
type_data, number = get_istance()

CC = pycc.GetInstance() 
current_directory=os.path.dirname(os.path.abspath(__file__))
params = pycc.FileIOFilter.LoadParameters()
path_potree_converter=os.path.join(current_directory,'potree-2.1.1\\','PotreeConverter.exe')

#%% GUI
# Create the main window
class GUI_potree_converter(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory

    # MAIN FRAME
    def main_frame (self, window):
        
        def save_file_dialog():
            # Abrir el diálogo para seleccionar la ruta de guardado
            directory = filedialog.askdirectory()
            self.output_directory = directory
            
            # Mostrar la ruta seleccionada en el textbox correspondiente
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, self.output_directory)
            
        def destroy (self):
            window.destroy ()
        
        window.title("Potree Converter")
        
        # Disable resizing the window
        window.resizable(False, False)
        
        # Remove minimize and maximize buttons (title bar only shows close button)
        window.attributes('-toolwindow', -1)

        # Create a frame for the form
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()

        # Labels
        label_out = tk.Label(form_frame, text="Choose output directory:")
        label_out.grid(row=0, column=0, sticky="w",pady=2)

        # Entry
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=0, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
        
        # Buttons
        row_buttons=[0]  
        button_names=["..."]  
        _=definition_of_buttons_type_1("form_frame",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog()],
                                       form_frame,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self),lambda:destroy(self)],
                                     1,
                                     form_frame,
                                     1
                                     )


        def run_algorithm_1 (self):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Converting the points cloud to potree format...")
            #SELECTION CHECK
            if not CC.haveSelection():
                raise RuntimeError("No folder or entity selected")
            else:
                
                entities = CC.getSelectedEntities()[0]
            # RUN THE CMD FOR POINT CLOUD 
                if hasattr(entities, 'points'):
 
                    pc_name = entities.getName()
                    output_file = os.path.join(self.output_directory,pc_name)
                    pc_name_full=pc_name+'.las'
                    input_file=os.path.join(self.output_directory, pc_name_full)  
                    params = pycc.FileIOFilter.SaveParameters()
                    result = pycc.FileIOFilter.SaveToFile(entities, input_file, params)
                
                    command = path_potree_converter + ' -i ' + input_file + ' -o ' + output_file + ' --generate-page'
                    os.system(command)
                    current_name = os.path.join(output_file,'.html')
                    new_name = os.path.join(output_file,'index.html')
                    
                    # Rename the file
                    os.rename(current_name, new_name)
                    os.remove (input_file)
            # RUN THE CMD FOLDER
                else:
                    entities = CC.getSelectedEntities()[0]
                    number = entities.getChildrenNumber()  
                    for i in range (number):
                        if hasattr(entities.getChild(i), 'points'):
                            pc = entities.getChild(i)
                            pc_name = pc.getName()
                            output_file= os.path.join(self.output_directory,pc_name)                
                            pc_name_full=pc_name+'.las'
                            input_file=os.path.join(self.output_directory,pc_name_full)   
                            params = pycc.FileIOFilter.SaveParameters()
                            result = pycc.FileIOFilter.SaveToFile(entities.getChild(i), input_file, params)
                                
                            command = path_potree_converter + ' -i ' + input_file + ' -o ' + output_file + ' --generate-page'
                            os.system(command)
                            current_name = os.path.join(output_file,'.html')
                            new_name = os.path.join(output_file,'index.html')
                            
                            # Rename the file
                            os.rename(current_name, new_name)
                            os.remove (input_file)
            # Stop the progress bar
            progress.stop()     
            print("Potree Converter has finished")
    
    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
    

#%% START THE GUI
if __name__ == "__main__":
    try:
        # START THE MAIN WINDOW        
        window = tk.Tk()
        app = GUI_potree_converter()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()