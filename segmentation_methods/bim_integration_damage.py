# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:32:55 2024

@author: psh
"""

#%% LIBRARIES
import os
import subprocess
import sys
import json

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import traceback
import pandas as pd

# CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory = os.path.sep.join(path_parts[:-2]) + '\\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata, get_point_clouds_name, check_input
from main_gui import (
    definition_of_labels_type_1,
    definition_of_buttons_type_1,
    definition_run_cancel_buttons_type_1,
)

#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory = os.path.dirname(os.path.abspath(__file__))

#%% INITIAL OPERATIONS
name_list = get_point_clouds_name()

#%% GUI
class GUI_bid(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # Default parameters
        self.parameters_bi = {
            "strategy": "Damage as point",
            "point_cloud": "input_point_cloud.txt"
        }

        # Output directory
        self.output_directory = os.getcwd()
    
    def damage_as_point(self, pc_file_path, rfa_file_path):
        """
        Create a JSON file for the strategy `damage_as_point`.
        """
        pc_file_path_json = pc_file_path.replace("\\", "/").replace("/", "\\")
        rfa_file_path_json = rfa_file_path.replace("\\", "/").replace("/", "\\")
    
        json_content = {
            "file": pc_file_path_json,
            "family_point": rfa_file_path_json
        }
        json_file_path = os.path.join(self.output_directory, "config.json")
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)
    
        print(f"JSON file created in: {json_file_path}")
    
    
    def damage_as_path(self, pc_file_path, rfa_file_path):
        """
        Create a JSON file for the strategy `damage_as_path`.
        """
        pc_file_path_json = pc_file_path.replace("\\", "/").replace("/", "\\")
        rfa_file_path_json = rfa_file_path.replace("\\", "/").replace("/", "\\")
    
        json_content = {
            "file": pc_file_path_json,
            "family_path": rfa_file_path_json
        }
        json_file_path = os.path.join(self.output_directory, "config.json")
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file, indent=4)
    
        print(f"JSON file created in: {json_file_path}")

    def main_frame(self, window):
        def save_file_dialog():
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, self.output_directory)
            
        # Setting the main window title and properties
        window.title("BIM Integration")
        window.resizable(False, False)
        window.attributes('-toolwindow', -1)  # Remove minimize and maximize buttons
    
        form_frame = tk.Frame(window, padx=10, pady=10)
        form_frame.pack()
    
        # Creating labels for form fields
        label_texts = [
            "Choose point clouds:",
            "Strategy for representing the damage:",
            "Output directory:"
        ]
        row_positions = [0, 1, 2]
        definition_of_labels_type_1("window", label_texts, row_positions, form_frame, 0)
    
        # Listbox for multiple point cloud selection
        global listbox_point_cloud
        listbox_point_cloud = tk.Listbox(form_frame, selectmode="multiple", width=30, height=5)
        listbox_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        for name in name_list:
            listbox_point_cloud.insert(tk.END, name)
    
        # Combobox for selecting the strategy
        combo_strategy = ttk.Combobox(form_frame, values=["Damage as point", "Damage as path"], width=20)
        combo_strategy.grid(column=1, row=1, sticky="e", pady=2)
        combo_strategy.set("Damage as point")
    
        # Entry field for the output directory
        entry_widget = ttk.Entry(form_frame, width=30)
        entry_widget.grid(row=2, column=1, sticky="e", pady=2)
        entry_widget.insert(0, self.output_directory)
    
        # Button to select directory
        definition_of_buttons_type_1(
            "window", ["..."], [2],
            [save_file_dialog], form_frame, 2
        )
    
        # Run and cancel buttons
        definition_run_cancel_buttons_type_1(
            "window",
            [lambda:run_algorithm_1(self), lambda: window.destroy(self)],
            3, form_frame, 1
        )
        
        def run_algorithm_1(self):
            """
            Processes the selected clouds according to the chosen strategy.
            Generates the corresponding TXT and JSON files.
            """
            selected_indices = listbox_point_cloud.curselection()
            selected_names = [listbox_point_cloud.get(i) for i in selected_indices]
            strategy = combo_strategy.get()
        
            if not selected_names:
                messagebox.showerror("Error", "You must select at least one point cloud.")
                return
        
            combined_data = []
            centroids = []
        
            try:
                for pc_name in selected_names:
                    pc = check_input(name_list, pc_name)
                    pcd = P2p_getdata(pc, False, True, True)
        
                    # Check for duplicates if there are already combined data
                    if combined_data:
                        current_points = pcd[["X", "Y", "Z"]]
                        combined_points = pd.DataFrame(combined_data, columns=pcd.columns)[["X", "Y", "Z"]]
                        duplicates = combined_points.merge(current_points, on=["X", "Y", "Z"], how="inner")
        
                        if not duplicates.empty:
                            raise ValueError(f"Duplicate points found in the cloud '{pc_name}'.")
        
                    combined_data.extend(pcd.values)
        
                    # Calculate centroids for `Damage as path`
                    if strategy == "Damage as path":
                        centroid = pcd[["X", "Y", "Z", "category_of_damage"]].mean().values
                        centroids.append(centroid)
        
                # Generate output according to strategy
                if strategy == "Damage as point":
                    combined_df = pd.DataFrame(combined_data, columns=pcd.columns)
                    output_file_path = os.path.join(self.output_directory, "damage.txt")
                    combined_df.to_csv(output_file_path, sep=' ', header=True, index=False)
                    print(f"TXT file in: {output_file_path}")
        
                    rfa_file_path = os.path.join(os.path.dirname(__file__), "damage_as_point.rfa")
                    self.damage_as_point(output_file_path, rfa_file_path)
        
                elif strategy == "Damage as path":
                    centroids_df = pd.DataFrame(centroids, columns=["X", "Y", "Z", "category_of_damage"])
                    output_file_path = os.path.join(self.output_directory, "damage.txt")
                    centroids_df.to_csv(output_file_path, sep=' ', header=True, index=False)
                    print(f"TXT file in: {output_file_path}")
        
                    rfa_file_path = os.path.join(os.path.dirname(__file__), "damage_as_path.rfa")
                    self.damage_as_path(output_file_path, rfa_file_path)
        
            except Exception as e:
                messagebox.showerror("Error", f"Error processing point clouds: {e}")
                traceback.print_exc()

            print("Process completed.")

    def show_frame(self, window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()

#%% RUN THE GUI
if __name__ == "__main__":
    try:
        window = tk.Tk()
        window.wm_iconbitmap('')
        app = GUI_bid()
        app.main_frame(window)
        window.mainloop()
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()
        window.destroy()