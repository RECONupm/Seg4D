# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:58:50 2024

@author: LuisJa
"""
#%% LIBRARIES
import os
import json
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
import numpy as np

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory = os.path.join(
    os.path.sep.join(path_parts[:-2]), "main_module"
)

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1
from execution_logging import (
    append_execution_exception,
    append_execution_result,
    execution_failure_message,
    run_logged_command,
    start_execution_log,
)
from point_cloud_processing import spatial_majority_filter
from supervoxel_processing import (
    broadcast_supervoxel_predictions,
    majority_labels_by_supervoxel,
    reduce_features_by_supervoxel,
)
#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))

config_file=os.path.join(current_directory,r'..\configs\executables.yml')

# Read the configuration from the YAML file for the set-up
with open(config_file, 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)
path_optimal_flow= os.path.join(current_directory,config_data['OPTIMAL_FLOW'])
path_random_forest= os.path.join(current_directory,config_data['RANDOM_FOREST'])
path_support_vector_machine= os.path.join(current_directory,config_data['SUPPORT_VECTOR_MACHINE'])
path_linear_regression= os.path.join(current_directory,config_data['LINEAR_REGRESSION'])
path_prediction= os.path.join(current_directory,config_data['PREDICTION'])
path_aml= os.path.join(current_directory,config_data['TPOT'])

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_mls(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
       
        # Features2include
        self.features2include=[] 
        self.values_list=[]
        self.features=[]
        self._classifier_process = None
        self._classifier_log_handle = None
        self._classifier_cancel_requested = False
        self._classifier_progress_window = None
        self._classifier_progress_bar = None
        self._classifier_progress_message = None
        self._classifier_progress_value = None
        self._classifier_progress_button = None
        
        
        # Optimal flow
        self.set_up_parameters_of= {
            "selectors": ['kbest_f','rfe_lr','rfe_tree','rfe_rf','rfecv_tree','rfecv_rf','rfe_svm','rfecv_svm'],
            "percentage": 25,
            "cv": 5,
            "point_cloud":"input_point_cloud.txt"
        }
        
        # Random forest
        self.set_up_parameters_rf= {
            "n_estimators": 200,
            "criterion": "gini",
            "max_depth": 0,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0,
            "max_features": "sqrt",
            "max_leaf_nodes": 0,
            "min_impurity_decrease": 0,
            "bootstrap": "True",
            "class_weight": "None",
            "ccp_alpha": 0,
            "max_samples": 0,
            "n_jobs": -1
            }
        
        # Support Vector Machine- Support Vector Classification
        self.set_up_parameters_svm_svc= {
            "c": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef()": 0.0,
            "shrinking": "True",
            "probability": "False",
            "tol": 0.001,
            "class_weight": 'balanced',
            "max_iter": 1000,
            "decision_function_shape": 'ovr',
            "break_ties":"False",
            }
        
        # Logistic regression multilabel
        self.set_up_parameters_lr= {
            "penalty": "l2",
            "dual": "False",
            "tol": 0.0001,
            "c": 1,
            "fit_intercept": "True",
            "intercept_scaling": 1.0,
            "class_weight": "No",
            "solver": 'lbfgs',
            "max_iter": 100,
            "multi_class": 'auto',
            "n_jobs":-1,
            "l1_ratio":0       
            }
            
        # Auto-ml
        self.set_up_parameters_aml= {
            "generations": 5,
            "population_size": 20,
            "mutation_rate": 0.9,
            "crossover_rate": 0.1,
            "scoring": "balanced_accuracy",
            "cv": 5,
            "max_time_mins": 60,
            "max_eval_time_mins": 5,
            "early_stop": 0         
            }     

        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        self.file_path_features=os.path.join(current_directory, self.output_directory)
        self.file_path_pkl=os.path.join(current_directory, self.output_directory)
        self.postprocessing_parameters = {
            "neighbors": 16,
            "minimum_agreement": 0.70,
            "iterations": 1,
        }

    def _show_classifier_progress(self, title):
        self._close_classifier_progress()
        owner = self.winfo_toplevel()
        window = tk.Toplevel(owner)
        window.title(title)
        window.transient(owner)
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._cancel_classifier)
        panel = ttk.Frame(window, padding=12)
        panel.pack(fill="both", expand=True)
        self._classifier_progress_message = tk.StringVar(
            value="Preparing point clouds..."
        )
        ttk.Label(
            panel,
            textvariable=self._classifier_progress_message,
            width=62,
        ).pack(anchor="w")
        self._classifier_progress_value = tk.DoubleVar(value=0)
        self._classifier_progress_bar = ttk.Progressbar(
            panel,
            variable=self._classifier_progress_value,
            maximum=100,
            length=470,
            mode="determinate",
        )
        self._classifier_progress_bar.pack(fill="x", pady=10)
        self._classifier_progress_button = ttk.Button(
            panel,
            text="Cancel",
            command=self._cancel_classifier,
        )
        self._classifier_progress_button.pack(side="right")
        self._classifier_progress_window = window
        window.update_idletasks()
        window.grab_set()
        window.lift()
        window.focus_force()

    def _set_classifier_progress(self, percent=None, message=None, pulse=False):
        if (
            self._classifier_progress_window is None
            or not self._classifier_progress_window.winfo_exists()
        ):
            return
        if message is not None and self._classifier_progress_message is not None:
            self._classifier_progress_message.set(message)
        if self._classifier_progress_bar is not None:
            if pulse:
                if str(self._classifier_progress_bar.cget("mode")) != "indeterminate":
                    self._classifier_progress_bar.configure(mode="indeterminate")
                    self._classifier_progress_bar.start(12)
            else:
                if str(self._classifier_progress_bar.cget("mode")) != "determinate":
                    self._classifier_progress_bar.stop()
                    self._classifier_progress_bar.configure(mode="determinate")
                if percent is not None and self._classifier_progress_value is not None:
                    self._classifier_progress_value.set(float(percent))
        self._classifier_progress_window.update_idletasks()

    def _close_classifier_progress(self):
        if self._classifier_progress_bar is not None:
            self._classifier_progress_bar.stop()
        window = self._classifier_progress_window
        if window is not None and window.winfo_exists():
            try:
                window.grab_release()
            except tk.TclError:
                pass
            window.destroy()
        self._classifier_progress_window = None
        self._classifier_progress_bar = None
        self._classifier_progress_message = None
        self._classifier_progress_value = None
        self._classifier_progress_button = None

    def _cancel_classifier(self):
        process = self._classifier_process
        if process is not None and process.poll() is None:
            self._classifier_cancel_requested = True
            self._set_classifier_progress(
                message="Cancelling the classifier...", pulse=True
            )
            if self._classifier_progress_button is not None:
                self._classifier_progress_button.configure(state="disabled")
            process.terminate()
            return
        self._close_classifier_progress()
        
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
        
        # Function to get the selectors of the set_up_paramenters
        def on_ok_button_click():
            selected_items = listbox.curselection()
            selected_values = [listbox.get(index) for index in selected_items]
            messagebox.showinfo("Selectors", f"Selected options: {', '.join(selected_values)}")
        
        # Function to save and get the output_directory
        def save_file_dialog(tab):
            directory = filedialog.askdirectory()
            if directory:
                self.output_directory = directory                
                if tab ==1:  # Update the entry widget of the tab                   
                    t1_entry_widget.delete(0, tk.END)
                    t1_entry_widget.insert(0, self.output_directory)    
                elif tab==2:
                    t2_entry_widget.delete(0, tk.END)
                    t2_entry_widget.insert(0, self.output_directory)  
                elif tab==3:
                    t3_entry_widget.delete(0, tk.END)
                    t3_entry_widget.insert(0, self.output_directory)      
                
        # Destroy the window
        def destroy (self):
            process = self._classifier_process
            if process is not None and process.poll() is None:
                if not messagebox.askyesno(
                    "Cancel classifier",
                    "A classifier is still running. Do you want to stop it?",
                    parent=root,
                ):
                    return
                process.terminate()
            if self._classifier_log_handle is not None:
                self._classifier_log_handle.close()
                self._classifier_log_handle = None
            self._classifier_process = None
            self._close_classifier_progress()
            root.destroy()
        
        # Safe the set_up_paramenters in accordance with the GUI
        def save_setup_parameters (self,algo,*params): 
            if algo=="Random Forest":
                self.set_up_parameters_rf["n_estimators"]=params[0]
                self.set_up_parameters_rf["criterion"]=params[1]
                self.set_up_parameters_rf["max_depth"]=params[2]
                self.set_up_parameters_rf["min_samples_split"]=params[3]
                self.set_up_parameters_rf["min_samples_leaf"]=params[4]
                self.set_up_parameters_rf["min_weight_fraction_leaf"]=params[5]
                self.set_up_parameters_rf["max_features"]=params[6]
                self.set_up_parameters_rf["max_leaf_nodes"]=params[7]
                self.set_up_parameters_rf["min_impurity_decrease"]=params[8]
                self.set_up_parameters_rf["bootstrap"]=params[9]
                self.set_up_parameters_rf["class_weight"]=params[10]
                self.set_up_parameters_rf["ccp_alpha"]=params[11]
                self.set_up_parameters_rf["max_samples"]=params[12]            
                self.set_up_parameters_rf["n_jobs"]=params[13]             
                
            elif algo=="Support Vector Machine":
                self.set_up_parameters_svm_svc ["c"]=params[0]
                self.set_up_parameters_svm_svc ["kernel"]=params[1]
                self.set_up_parameters_svm_svc ["degree"]=params[2]
                self.set_up_parameters_svm_svc ["gamma"]=params[3]
                self.set_up_parameters_svm_svc ["coef()"]=params[4]
                self.set_up_parameters_svm_svc ["shrinking"]=params[5]
                self.set_up_parameters_svm_svc ["probability"]=params[6]
                self.set_up_parameters_svm_svc ["tol"]=params[7]
                self.set_up_parameters_svm_svc ["class_weight"]=params[8]
                self.set_up_parameters_svm_svc ["max_iter"]=params[9]
                self.set_up_parameters_svm_svc ["decision_function_shape"]=params[10]
                self.set_up_parameters_svm_svc ["break_ties"]=params[11]
                
            elif algo== "Logistic Regression":
                self.set_up_parameters_lr ["penalty"]=params[0]
                self.set_up_parameters_lr ["dual"]=params[1]
                self.set_up_parameters_lr ["tol"]=params[2]
                self.set_up_parameters_lr ["c"]=params[3]
                self.set_up_parameters_lr ["fit_intercept"]=params[4]
                self.set_up_parameters_lr ["intercept_scaling"]=params[5]
                self.set_up_parameters_lr ["class_weight"]=params[6]
                self.set_up_parameters_lr ["solver"]=params[7]
                self.set_up_parameters_lr ["max_iter"]=params[8]
                self.set_up_parameters_lr ["multi_class"]=params[9]
                self.set_up_parameters_lr ["l1_ratio"]=params[10]
                self.set_up_parameters_lr ["n_jobs"]=params[11]
               
            elif algo=="Auto Machine Learning":             
              
                self.set_up_parameters_aml["generations"]=params[0]
                self.set_up_parameters_aml["population_size"]=params[1]
                self.set_up_parameters_aml["mutation_rate"]=params[2]
                self.set_up_parameters_aml["crossover_rate"]=params[3]
                self.set_up_parameters_aml["scoring"]=params[4]
                self.set_up_parameters_aml["cv"]=params[5]
                self.set_up_parameters_aml["max_time_mins"]=params[6]
                self.set_up_parameters_aml["max_eval_time_mins"]=params[7]
                self.set_up_parameters_aml["early_stop"]=params[8]
                
        # Load the configuration files for prediction
        def load_configuration_dialog():
           global load_configuration   
           file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
           if file_path:
               load_configuration = file_path 
               self.file_path_pkl=file_path 
               t3_entry_pkl_widget.delete(0, tk.END)
               t3_entry_pkl_widget.insert(0, self.file_path_pkl)
                    
        def load_features_dialog():
            global load_features 
            file_path = filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
            if file_path:
                load_features = file_path 
                self.file_path_features=file_path
                t3_entry_features_widget.delete(0, tk.END)
                t3_entry_features_widget.insert(0, self.file_path_features)
                
        # Window for the configuration of the machine learning algorithms
        def show_set_up_window (self,algo):      
            def close_set_up_window():
                if set_up_window.winfo_exists():
                    set_up_window.destroy()

            def on_ok_button_click(algo):
                if algo=="Random Forest":
                    save_setup_parameters(self,algo, int(rf_entries[0].get()), str(rf_comboboxes[1].get()),int(rf_entries[2].get()),float(rf_entries[3].get()),float(rf_entries[4].get()),float(rf_entries[5].get()),str(rf_comboboxes[6].get()),int(rf_entries[7].get()),float(rf_entries[8].get()),str(rf_comboboxes[9].get()),str(rf_comboboxes[10].get()),float(rf_entries[11].get()),int(rf_entries[12].get()),int(rf_entries[13].get()))
                elif algo=="Support Vector Machine":
                    save_setup_parameters(self, algo,float(svm_scv_entries[0].get()),str(svm_scv_comboboxes[1].get()),int(svm_scv_entries[2].get()),str(svm_scv_comboboxes[3].get()),float(svm_scv_entries[4].get()),str(svm_scv_comboboxes[5].get()),str(svm_scv_comboboxes[6].get()),float(svm_scv_entries[7].get()),str(svm_scv_comboboxes[8].get()),int(svm_scv_entries[9].get()),str(svm_scv_comboboxes[10].get()),str(svm_scv_comboboxes[11].get()))
                elif algo=="Logistic Regression":
                    save_setup_parameters(self, algo,str(lr_comboboxes[0].get()),str(lr_comboboxes[1].get()),float(lr_entries[2].get()),float(lr_entries[3].get()),str(lr_comboboxes[4].get()),float(lr_entries[5].get()),str(lr_comboboxes[6].get()),str(lr_comboboxes[7].get()),int(lr_entries[8].get()),str(lr_comboboxes[9].get()),float(lr_entries[10].get()),int(lr_entries[11].get()))
                elif algo=="Auto Machine Learning":                    
                    save_setup_parameters(self,algo,int(aml_entries[0].get()),int(aml_entries[1].get()),float(aml_entries[2].get()),float(aml_entries[3].get()),str(aml_comboboxes[4].get()),int(aml_entries[5].get()),int(aml_entries[6].get()),int(aml_entries[7].get()),int(aml_entries[8].get()))
                    # Entries

                close_set_up_window()  # Close the window after saving parameters
                
            # Setup window
            set_up_window = tk.Toplevel(root)
            set_up_window.withdraw()
            set_up_window.title("Set Up the algorithm")
            set_up_window.resizable (False, False)
            # Remove minimize and maximize button 
            set_up_window.attributes ('-toolwindow',-1)
            set_up_window.transient(root)
            set_up_window.protocol("WM_DELETE_WINDOW", close_set_up_window)
               
            if algo=="Random Forest":
               
                #Labels        
                label_texts = [
                    "Number of trees:",
                    "Function to measure the quality of a split:",
                    "Maximum depth of the tree:",
                    "Minimum number of samples for splitting:",
                    "Minimum number of samples to at a leaf node:",
                    "Minimum weighted fraction:",
                    "Number of features to consider for best split:",
                    "Maximum number of leaf nodes:",
                    "Impurity to split the node:",
                    "Use of bootstrap:",
                    "Weights associated to the classes:",
                    "Complexity parameter used for Minimal cost:",
                    "Number of samples to draw to train each base estimator:",
                    "Number of cores to use (-1 means all):"
                ]
                row_positions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]        
                definition_of_labels_type_1 ("rf",label_texts,row_positions,set_up_window,0)

                # Entries
                entry_insert = [
                    self.set_up_parameters_rf ["n_estimators"],
                    self.set_up_parameters_rf["max_depth"],
                    self.set_up_parameters_rf["min_samples_split"],
                    self.set_up_parameters_rf["min_samples_leaf"],
                    self.set_up_parameters_rf["min_weight_fraction_leaf"],
                    self.set_up_parameters_rf["max_leaf_nodes"],
                    self.set_up_parameters_rf["min_impurity_decrease"],
                    self.set_up_parameters_rf["ccp_alpha"],
                    self.set_up_parameters_rf["max_samples"],
                    self.set_up_parameters_rf["n_jobs"]
                    ]
                row_positions = [0,2,3,4,5,7,8,11,12,13]        
                rf_entries = definition_of_entries_type_1 ("rf",entry_insert,row_positions,set_up_window,1) 

                # Combobox
                combobox_insert = [
                    ["gini","entropy","log_loss"],
                    ['sqrt','log2'],
                    ['True','False'],
                    ['No',"balanced","balanced_subsample"]
                    ]
                row_positions = [1,6,9,10]
                selected_element = ["gini","sqrt","True"]
                rf_comboboxes =definition_of_combobox_type_1 ("rf",combobox_insert,row_positions, selected_element,set_up_window,1)                                 
        
                # Buttons  
                _=definition_ok_cancel_buttons_type_1('rf',[lambda: on_ok_button_click(algo),None],14,set_up_window,1)
                
            elif algo== "Support Vector Machine":
                
                # Labels
                label_texts = [
                    "Regularization parameter:",
                    "Kernel type:",
                    "Degree of the polynomial kernel function:",
                    "Kernel coefficient:",
                    "Idenpendent term in kernel function:",
                    "Use shrinking heuristics:",
                    "Enable probaility estimates:",                    
                    "Tolerance for stopping criterion:",                    
                    "Class weight:",
                    "Max number of iterations:",
                    "Returned function:",
                    "Break ties:",                     
                ]
                row_positions = [0,1,2,3,4,5,6,7,8,9,10,11]        
                definition_of_labels_type_1 ("svm_scv",label_texts,row_positions,set_up_window,0)
                
                # Entries
                entry_insert = [
                    self.set_up_parameters_svm_svc ["c"],
                    self.set_up_parameters_svm_svc ["degree"],
                    self.set_up_parameters_svm_svc ["coef()"],
                    self.set_up_parameters_svm_svc ["tol"],
                    self.set_up_parameters_svm_svc ["max_iter"],
                    ]
                row_positions = [0,2,4,7,9]        
                svm_scv_entries = definition_of_entries_type_1 ("svm_scv",entry_insert,row_positions,set_up_window,1) 
                
                # Combobox
                combobox_insert = [
                    ["linear","poly","rbf","sigmoid"],
                    ["scale","auto"],
                    ["True","False"],
                    ["True","False"],
                    ["No","balanced"],
                    ["ovo","ovr"],
                    ["True","False"]
                    ]
                row_positions = [1,3,5,6,8,10,11]
                selected_element = ["rbf","scale","True","False","No","ovr","False"]
                svm_scv_comboboxes =definition_of_combobox_type_1 ("svm_scv",combobox_insert,row_positions, selected_element,set_up_window,1)
                            
                # Buttons  
                _=definition_ok_cancel_buttons_type_1("svm_scv",[lambda: on_ok_button_click(algo),None],12,set_up_window,1) 
                
            elif algo== "Logistic Regression":
                
                # Labels
                label_texts = [
                    "Penalty function:",
                    "Constrained problem (dual):",
                    "Tolerance for stopping criterion:",
                    "Inverse of regularization strength:",
                    "Add a bias to the model:",
                    "Intercept scaling:",
                    "Class weight:",
                    "Type of solver:",
                    "Max number of iterations:",
                    "Type of multiclass fitting strategy:",
                    "Elastic-Net mixing parameter:",
                    "Number of cores to use (-1 means all):"
                ]
                row_positions = [0,1,2,3,4,5,6,7,8,9,10,11]        
                definition_of_labels_type_1 ("lr",label_texts,row_positions,set_up_window,0)

                # Entries
                entry_insert = [
                    self.set_up_parameters_lr ["tol"],
                    self.set_up_parameters_lr["c"],
                    self.set_up_parameters_lr ["intercept_scaling"],
                    self.set_up_parameters_lr ["max_iter"],
                    self.set_up_parameters_lr ["l1_ratio"],
                    self.set_up_parameters_lr ["n_jobs"]
                    ]
                row_positions = [2,3,5,8,10,11]        
                lr_entries = definition_of_entries_type_1 ("lr",entry_insert,row_positions,set_up_window,1) 
                
                # Combobox
                combobox_insert = [
                    ["l1","l2","elasticnet","No"],
                    ["True","False"],
                    ["True","False"],
                    ["No","balanced"],
                    ["lbfgs","liblinear","newtow-cg","newton-cholesky","sag","saga"],
                    ["auto","ovr","multimodal"],
                    ]
                row_positions = [0,1,4,6,7,9]
                selected_element = ["l2","False","True","No","lbfgs","auto"]
                lr_comboboxes =definition_of_combobox_type_1 ("lr",combobox_insert,row_positions, selected_element,set_up_window,1)
                
                # Buttons  
                _=definition_ok_cancel_buttons_type_1("lr",[lambda: on_ok_button_click(algo),None],12,set_up_window,1)   
                    
            elif algo=="Auto Machine Learning":
          
                # Labels
                label_texts = [
                    "Number of generations:",
                    "Population size:",
                    "Mutation rate:",
                    "Crossover rate:",
                    "Function used to evaluate the quality:",
                    "Cross-validation:",
                    "Maximum total time (mins):",
                    "Maximum time per evaluation (mins):",
                    "Number of generations without improvement:",
                ]
                
                row_positions = [0,1,2,3,4,5,6,7,8]        
                definition_of_labels_type_1 ("aml",label_texts,row_positions,set_up_window,1)  
                
                entry_insert = [
                    self.set_up_parameters_aml["generations"],
                    self.set_up_parameters_aml["population_size"],
                    self.set_up_parameters_aml["mutation_rate"],
                    self.set_up_parameters_aml["crossover_rate"],
                    self.set_up_parameters_aml["cv"],
                    self.set_up_parameters_aml["max_time_mins"],
                    self.set_up_parameters_aml["max_eval_time_mins"],
                    self.set_up_parameters_aml["early_stop"]                  
                    ]
                row_positions = [0,1,2,3,5,6,7,8]        
                aml_entries = definition_of_entries_type_1 ("aml",entry_insert,row_positions,set_up_window,2) 
                
                # Combobox
                combobox_insert = [
                    ["balanced_accuracy","accuracy","f1","f1_weighted","precision","precision_weighted"]
                    ]
                row_positions = [4]
                selected_element = [self.set_up_parameters_aml["scoring"]]
                aml_comboboxes =definition_of_combobox_type_1 ("aml",combobox_insert,row_positions, selected_element,set_up_window,2)
                                              
                # Buttons  
                _=definition_ok_cancel_buttons_type_1("aml",[lambda: on_ok_button_click(algo),None],9,set_up_window,1) 

            # Map the owned dialog only after all its controls exist. Waiting
            # for visibility before setting the grab avoids Windows placing it
            # behind its already-modal owner.
            set_up_window.update_idletasks()
            set_up_window.deiconify()
            set_up_window.wait_visibility()
            set_up_window.lift(root)
            set_up_window.grab_set()
            set_up_window.focus_force()
            root.wait_window(set_up_window)

            # Opening the nested dialog replaces the owner's grab. Restore it
            # after the dialog closes so the SEG4D main window remains blocked.
            try:
                if root.winfo_exists():
                    root.grab_set()
                    root.lift()
                    root.focus_force()
            except tk.TclError:
                pass
            
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Supervised Machine Learning segmentation")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button 
        root.protocol("WM_DELETE_WINDOW", lambda: destroy(self))
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")   
        
        tab1 = ttk.Frame(tab_control) # Create 3 tabs
        tab1.pack()
        tab_control.add(tab1, text='Feature selection')
        tab2 = ttk.Frame(tab_control) # Create 3 tabs
        tab2.pack() 
        tab_control.add(tab2, text='Classification')
        tab3 = ttk.Frame(tab_control) # Create 3 tabs
        tab3.pack()
        tab_control.add(tab3, text='Prediction')
        tab_control.pack(expand=1, fill="both")
       
        
        # TAB1 = FEATURE SELECTION      
        
        # Some lines to start the tab      
        listbox = tk.Listbox(tab1, selectmode=tk.MULTIPLE, height=len(self.set_up_parameters_of["selectors"]))
        for value in self.set_up_parameters_of["selectors"]:
            listbox.insert(tk.END, value)
            
        # Select first six elements
        for i in range(6):
            listbox.selection_set(i)
        listbox.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # Get the selected parameters from the Listbox
        selected_params = [self.set_up_parameters_of["selectors"][i] for i in listbox.curselection()]
        
        # Labels
        label_texts = [
            "Choose point cloud:",
            "Selectors:",
            "Features to include:",
            "Number of features to consider:",
            "Folds for cross-validation:",
            "Choose output directory:",
        ]
        row_positions = [0,1,2,3,4,5]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,0) 
            
        # Combobox
        t1_combo_point_cloud=ttk.Combobox (tab1,values=name_list)
        t1_combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        t1_combo_point_cloud.set("Select the point cloud used for feature selection:")

        # Entry
        t1_entry_percentage = ttk.Entry(tab1, width=10)
        t1_entry_percentage.insert(0,self.set_up_parameters_of["percentage"])
        t1_entry_percentage.grid(row=3, column=1, sticky="e", pady=2)
        
        t1_entry_cv = ttk.Entry(tab1, width=10)
        t1_entry_cv.insert(0,self.set_up_parameters_of["cv"])
        t1_entry_cv.grid(row=4, column=1, sticky="e", pady=2)
        
        t1_entry_widget = ttk.Entry(tab1, width=30)
        t1_entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.output_directory)

        # Buttons
        row_buttons=[1,5,2]  
        button_names=["OK","...","..."]  
        _=definition_of_buttons_type_1("tab1",
                                       button_names,
                                       row_buttons,
                                       [on_ok_button_click,lambda:save_file_dialog(1),lambda: show_features_window(self,name_list,t1_combo_point_cloud.get())],
                                       tab1,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("tab1",
                                     [lambda:run_algorithm_1(self,name_list,t1_combo_point_cloud.get(),listbox.curselection(),int(t1_entry_percentage.get()),int(t1_entry_cv.get())),lambda:destroy(self)],
                                     6,
                                     tab1,
                                     1
                                     ) 
        # TAB2 = CLASSIFICATION

        # Labels
        label_texts = [
            "Choose point cloud for training:",
            "Choose point cloud for testing:",
            "Select machine learning algorithm:",
            "Select the features to include:",
            "Choose output directory:",
            "Remove isolated labels (salt-and-pepper):",
            "Spatial neighbors:",
            "Minimum neighbor agreement (0.5-1.0):",
            "Smoothing iterations:",
            "Classify as supervoxels (Random Forest):",
        ]
        row_positions = [0,1,2,3,4,5,6,7,8,9]
        definition_of_labels_type_1 ("t2",label_texts,row_positions,tab2,0)     

        # Combobox
        t2_combo_point_cloud_training=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud_training.grid(column=1, row=0, sticky="e", pady=2)
        t2_combo_point_cloud_training.set("Select the point cloud used for training:")
        
        t2_combo_point_cloud_testing=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud_testing.grid(column=1, row=1, sticky="e", pady=2)
        t2_combo_point_cloud_testing.set("Select the point cloud used for testing:")
        
        algorithms=["Random Forest","Support Vector Machine", "Logistic Regression", "Auto Machine Learning"]
        t2_combo_algo=ttk.Combobox (tab2,values=algorithms, state="readonly")
        t2_combo_algo.grid(column=1, row=2, sticky="e", pady=2)
        t2_combo_algo.set("Not selected")
        
        # Entry
        t2_entry_widget = ttk.Entry(tab2, width=30)
        t2_entry_widget.grid(row=4, column=1, sticky="e", pady=2)
        t2_entry_widget.insert(0, self.output_directory)

        t2_postprocess_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab2,
            variable=t2_postprocess_var,
        ).grid(row=5, column=1, sticky="e", pady=2)
        t2_postprocess_neighbors = ttk.Entry(tab2, width=10)
        t2_postprocess_neighbors.insert(
            0, self.postprocessing_parameters["neighbors"]
        )
        t2_postprocess_neighbors.grid(row=6, column=1, sticky="e", pady=2)
        t2_postprocess_agreement = ttk.Entry(tab2, width=10)
        t2_postprocess_agreement.insert(
            0, self.postprocessing_parameters["minimum_agreement"]
        )
        t2_postprocess_agreement.grid(row=7, column=1, sticky="e", pady=2)
        t2_postprocess_iterations = ttk.Entry(tab2, width=10)
        t2_postprocess_iterations.insert(
            0, self.postprocessing_parameters["iterations"]
        )
        t2_postprocess_iterations.grid(row=8, column=1, sticky="e", pady=2)
        t2_supervoxel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab2,
            variable=t2_supervoxel_var,
        ).grid(row=9, column=1, sticky="e", pady=2)
               
        # Buttons  
        row_buttons=[2,3,4]  
        button_names=["Set-up","...","..."]  
        _=definition_of_buttons_type_1("tab2",
                                       button_names,
                                       row_buttons,
                                       [lambda: show_set_up_window(self,t2_combo_algo.get()),lambda: show_features_window(self,name_list,t2_combo_point_cloud_training.get()),lambda:save_file_dialog(2)],
                                       tab2,
                                       2
                                       ) 
        
        _=definition_run_cancel_buttons_type_1("tab2",
                                      [lambda:run_algorithm_2(self,t2_combo_algo.get(),t2_combo_point_cloud_training.get(),t2_combo_point_cloud_testing.get()),lambda:destroy(self)],
                                     10,
                                     tab2,
                                     1
                                     ) 
    
       
        # TAB3 = PREDICTION

        # Labels
        label_texts = [
            "Choose point cloud for prediction:",
            "Load feature file:",
            "Load pkl file:",
            "Choose a directory for temporal files:",
            "Remove isolated labels (salt-and-pepper):",
            "Spatial neighbors:",
            "Minimum neighbor agreement (0.5-1.0):",
            "Smoothing iterations:",
            "Predict as supervoxels:",
        ]
        row_positions = [0,1,2,3,4,5,6,7,8]
        definition_of_labels_type_1 ("t3",label_texts,row_positions,tab3,0)  
        
        # Combobox
        t3_combo_1=ttk.Combobox (tab3,values=name_list)
        t3_combo_1.grid(column=1, row=0, sticky="e", pady=2)
        t3_combo_1.set("Select the point cloud used for prediction:")

        # Entry
        t3_entry_features_widget = ttk.Entry(tab3, width=30)
        t3_entry_features_widget.grid(row=1, column=1, sticky="e", pady=2)
        t3_entry_features_widget.insert(0, self.file_path_features)
        t3_entry_pkl_widget = ttk.Entry(tab3, width=30)
        t3_entry_pkl_widget.grid(row=2, column=1, sticky="e", pady=2)
        t3_entry_pkl_widget.insert(0, self.file_path_pkl)
        t3_entry_widget = ttk.Entry(tab3, width=30)
        t3_entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        t3_entry_widget.insert(0, self.output_directory)

        t3_postprocess_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab3,
            variable=t3_postprocess_var,
        ).grid(row=4, column=1, sticky="e", pady=2)
        t3_postprocess_neighbors = ttk.Entry(tab3, width=10)
        t3_postprocess_neighbors.insert(
            0, self.postprocessing_parameters["neighbors"]
        )
        t3_postprocess_neighbors.grid(row=5, column=1, sticky="e", pady=2)
        t3_postprocess_agreement = ttk.Entry(tab3, width=10)
        t3_postprocess_agreement.insert(
            0, self.postprocessing_parameters["minimum_agreement"]
        )
        t3_postprocess_agreement.grid(row=6, column=1, sticky="e", pady=2)
        t3_postprocess_iterations = ttk.Entry(tab3, width=10)
        t3_postprocess_iterations.insert(
            0, self.postprocessing_parameters["iterations"]
        )
        t3_postprocess_iterations.grid(row=7, column=1, sticky="e", pady=2)
        t3_supervoxel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tab3,
            variable=t3_supervoxel_var,
        ).grid(row=8, column=1, sticky="e", pady=2)

        # Buttons
        row_buttons=[1,2,3]  
        button_names=["...","...","..."]  
        _=definition_of_buttons_type_1("tab3",button_names,row_buttons,[lambda: load_features_dialog(),lambda: load_configuration_dialog(),lambda:save_file_dialog(3)],tab3,2 ) 
        
        _=definition_run_cancel_buttons_type_1("tab3",
                                     [lambda:run_algorithm_3(self,t3_combo_1.get(),load_features,load_configuration),lambda:destroy(self)],
                                     9,
                                     tab3,
                                     1
                                     ) 
        
        # To run the optimal flow   
        def run_algorithm_1 (self,name_list,pc_training_name,selected_indices,f,cv): 
            # Update de data
            self.set_up_parameters_of["percentage"]=f
            self.set_up_parameters_of["cv"]=cv
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            
            # Transform the point cloud into a dataframe and select only the interseting columns
            feature_selection_pcd=P2p_getdata(pc_training,False,True,True)

            # # Save the point cloud
            # feature_selection_pcd[self.features2include].to_csv(os.path.join(self.output_directory, 'input_features.txt'),sep=' ',header=True,index=False)
            
            # Add the 'Classification' column to features2include
            features_to_save = self.features2include.copy()
            if 'Classification' not in features_to_save:
                features_to_save.append('Classification')
            
            # Save the point cloud
            feature_selection_pcd[features_to_save].to_csv(os.path.join(self.output_directory, 'input_features.txt'), sep=' ', header=True, index=False)
            
            # Save the selectors of the optimal flow algorithm             
            selected_items = [self.set_up_parameters_of["selectors"][i] for i in selected_indices]
            with open(os.path.join(self.output_directory,'selected_params.txt'), "w") as output_file:
                output_file.write('\n'.join(selected_items))
                
            # YAML file
            yaml_of = {
                'ALGORITHM': "Optimal-flow",
                'INPUT_POINT_CLOUD': os.path.join(self.output_directory,'input_features.txt'),               
                'SELECTORS_FILE': os.path.join(self.output_directory,'selected_params.txt'),
                'OUTPUT_DIRECTORY': self.output_directory,
                'CONFIGURATION': {
                    'cv': self.set_up_parameters_of["cv"],
                    'f': self.set_up_parameters_of["percentage"]
                }
            }            
            
            
            write_yaml_file (self.output_directory,yaml_of)
            
            # RUN THE COMMAND LINE      
            command = [
                path_optimal_flow,
                "--input",
                os.path.join(self.output_directory, "algorithm_configuration.yaml"),
                "--output",
                self.output_directory,
            ]
            completed, log_path = run_logged_command(
                command,
                self.output_directory,
                "optimal_flow_execution.log",
                "Optimal Flow feature selection",
                working_directory=os.path.dirname(path_optimal_flow),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                details={
                    "Algorithm": "Optimal Flow",
                    "Input shape": feature_selection_pcd.shape,
                    "Selected features": ", ".join(features_to_save),
                    "Configuration": yaml_of,
                },
            )
            if completed.returncode != 0:
                messagebox.showerror(
                    "Optimal Flow error",
                    execution_failure_message(
                        "Optimal Flow failed.",
                        log_path,
                        completed.returncode,
                    ),
                    parent=root,
                )
                return
            
            def read_features_and_print():
                features_file = os.path.join(self.output_directory, "features.txt")
            
                # Esperar hasta que el archivo exista
                while not os.path.exists(features_file):
                    time.sleep(1)  # Esperar 1 segundo antes de volver a verificar
            
                # Leer y imprimir las características seleccionadas
                with open(features_file, "r") as file:
                    features = file.read()
            
                print("Best features selected:", features)
                print("The process has been finished")    
            
            read_features_and_print()

        # To run the machine learning segmentation
        def run_algorithm_2 (self,algo,pc_training_name,pc_testing_name):
            if self._classifier_process is not None:
                if self._classifier_process.poll() is None:
                    messagebox.showinfo(
                        "Classifier running",
                        "Wait for the current classifier to finish or cancel it.",
                        parent=root,
                    )
                    return
                self._classifier_process = None

            if algo == "Not selected":
                messagebox.showerror(
                    "Missing algorithm",
                    "Please select an algorithm for the training.",
                    parent=root,
                )
                return
            if not self.features2include:
                messagebox.showerror(
                    "Missing features",
                    "Select at least one feature before running the classifier.",
                    parent=root,
                )
                return

            supervoxel_mode = t2_supervoxel_var.get()
            if supervoxel_mode and algo != "Random Forest":
                messagebox.showerror(
                    "Supervoxel classification",
                    "Supervoxel classification is currently available for Random Forest.",
                    parent=root,
                )
                return
            if supervoxel_mode and "Supervoxel ID" in self.features2include:
                messagebox.showerror(
                    "Invalid feature",
                    "Supervoxel ID identifies the processing unit and must not be "
                    "used as a predictive feature.",
                    parent=root,
                )
                return

            postprocessing = None
            if t2_postprocess_var.get():
                try:
                    postprocessing = {
                        "neighbors": int(t2_postprocess_neighbors.get()),
                        "minimum_agreement": float(
                            t2_postprocess_agreement.get()
                        ),
                        "iterations": int(t2_postprocess_iterations.get()),
                    }
                    if postprocessing["neighbors"] < 1:
                        raise ValueError("Spatial neighbors must be at least 1.")
                    if not 0.5 <= postprocessing["minimum_agreement"] <= 1.0:
                        raise ValueError(
                            "Minimum neighbor agreement must be between 0.5 and 1.0."
                        )
                    if postprocessing["iterations"] < 1:
                        raise ValueError("Smoothing iterations must be at least 1.")
                except ValueError as error:
                    messagebox.showerror(
                        "Invalid postprocessing configuration",
                        str(error),
                        parent=root,
                    )
                    return
                self.postprocessing_parameters.update(postprocessing)

            selected_output_directory = t2_entry_widget.get().strip()
            if not selected_output_directory:
                messagebox.showerror(
                    "Missing output directory",
                    "Choose an output directory.",
                    parent=root,
                )
                return
            self.output_directory = os.path.abspath(
                os.path.expanduser(selected_output_directory)
            )

            if algo == "Auto Machine Learning":
                parameters = self.set_up_parameters_aml
                validation_errors = []
                if parameters["generations"] < 1:
                    validation_errors.append("Generations must be at least 1.")
                if parameters["population_size"] < 1:
                    validation_errors.append("Population size must be at least 1.")
                if not 0 <= parameters["mutation_rate"] <= 1:
                    validation_errors.append("Mutation rate must be between 0 and 1.")
                if not 0 <= parameters["crossover_rate"] <= 1:
                    validation_errors.append("Crossover rate must be between 0 and 1.")
                if (
                    parameters["mutation_rate"]
                    + parameters["crossover_rate"]
                    > 1
                ):
                    validation_errors.append(
                        "Mutation rate plus crossover rate cannot exceed 1."
                    )
                if parameters["cv"] < 2:
                    validation_errors.append("Cross-validation must be at least 2.")
                if parameters["max_time_mins"] < 1:
                    validation_errors.append("Maximum total time must be at least 1 minute.")
                if parameters["max_eval_time_mins"] < 1:
                    validation_errors.append(
                        "Maximum evaluation time must be at least 1 minute."
                    )
                if parameters["early_stop"] < 0:
                    validation_errors.append(
                        "Early stop must be 0 (disabled) or a positive integer."
                    )
                if validation_errors:
                    messagebox.showerror(
                        "Invalid AutoML configuration",
                        "\n".join(validation_errors),
                        parent=root,
                    )
                    return

            self._classifier_cancel_requested = False
            self._show_classifier_progress(algo)
            self._set_classifier_progress(0, "Preprocessing the point cloud...")

            testing_point_coordinates = None
            testing_point_supervoxel_labels = None
            testing_supervoxel_ids = None
            try:
                os.makedirs(self.output_directory, exist_ok=True)

                # Check if the selection is a point cloud
                pc_training=check_input(name_list,pc_training_name)
                pc_testing=check_input(name_list,pc_testing_name)

                # Convert to a pandas dataframe
                pcd_training=P2p_getdata(pc_training,False,True,True)
                pcd_testing=P2p_getdata(pc_testing,False,True,True)
                self._set_classifier_progress(5, "Validating selected fields...")

                required_columns = set(self.features2include) | {"Classification"}
                if supervoxel_mode:
                    required_columns.add("Supervoxel ID")
                missing_training = sorted(required_columns - set(pcd_training.columns))
                missing_testing = sorted(required_columns - set(pcd_testing.columns))
                if missing_training or missing_testing:
                    raise ValueError(
                        "The selected fields are missing. "
                        f"Training: {missing_training}; testing: {missing_testing}"
                    )

                if supervoxel_mode:
                    self._set_classifier_progress(
                        7, "Aggregating training and testing supervoxels..."
                    )
                    testing_point_coordinates = pcd_testing[
                        ["X", "Y", "Z"]
                    ].copy()
                    testing_point_supervoxel_labels = np.asarray(
                        pcd_testing["Supervoxel ID"].values, dtype=int
                    )
                    model_columns = list(
                        dict.fromkeys(["X", "Y", "Z", *self.features2include])
                    )

                    training_ids, training_values = reduce_features_by_supervoxel(
                        pcd_training[model_columns].values,
                        pcd_training["Supervoxel ID"].values,
                    )
                    classification_ids, training_classes = (
                        majority_labels_by_supervoxel(
                            pcd_training["Classification"].values,
                            pcd_training["Supervoxel ID"].values,
                        )
                    )
                    if not np.array_equal(training_ids, classification_ids):
                        raise RuntimeError(
                            "Training supervoxel aggregation produced inconsistent IDs."
                        )
                    pcd_training = pd.DataFrame(
                        training_values, columns=model_columns
                    )
                    pcd_training["Classification"] = training_classes

                    testing_supervoxel_ids, testing_values = (
                        reduce_features_by_supervoxel(
                            pcd_testing[model_columns].values,
                            pcd_testing["Supervoxel ID"].values,
                        )
                    )
                    testing_class_ids, testing_classes = (
                        majority_labels_by_supervoxel(
                            pcd_testing["Classification"].values,
                            pcd_testing["Supervoxel ID"].values,
                        )
                    )
                    if not np.array_equal(
                        testing_supervoxel_ids, testing_class_ids
                    ):
                        raise RuntimeError(
                            "Testing supervoxel aggregation produced inconsistent IDs."
                        )
                    pcd_testing = pd.DataFrame(
                        testing_values, columns=model_columns
                    )
                    pcd_testing["Classification"] = testing_classes

                # Create the features file
                comma_separated = ','.join(self.features2include)
                with open(
                    os.path.join(self.output_directory, 'features.txt'),
                    'w',
                    encoding="utf-8",
                ) as file:
                    file.write(comma_separated)
                # Save the point clouds and the features
                pcd_training.to_csv(os.path.join(self.output_directory, 'input_point_cloud_training.txt'),sep=' ',header=True,index=False)
                pcd_testing.to_csv(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),sep=' ',header=True,index=False)
                self._set_classifier_progress(10, "Preparing classifier configuration...")
            except Exception as error:
                preprocessing_log = start_execution_log(
                    self.output_directory,
                    (
                        "automl_execution.log"
                        if algo == "Auto Machine Learning"
                        else "supervised_execution.log"
                    ),
                    f"Classifier preprocessing - {algo}",
                    details={
                        "Algorithm": algo,
                        "Selected features": ", ".join(self.features2include),
                    },
                )
                append_execution_exception(
                    preprocessing_log, "preprocessing the point clouds", error
                )
                self._close_classifier_progress()
                messagebox.showerror(
                    "Classifier error",
                    execution_failure_message(str(error), preprocessing_log),
                    parent=root,
                )
                return

            if algo=="Random Forest":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'), 
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Random_forest",
                    'CONFIGURATION': 
                        {
                        'n_estimators': self.set_up_parameters_rf["n_estimators"],
                        'criterion': self.set_up_parameters_rf["criterion"],
                        'max_depth': self.set_up_parameters_rf["max_depth"],
                        'min_samples_split': self.set_up_parameters_rf["min_samples_split"],
                        'min_samples_leaf': self.set_up_parameters_rf["min_samples_leaf"],
                        'min_weight_fraction_leaf': self.set_up_parameters_rf["min_weight_fraction_leaf"],
                        'max_features': self.set_up_parameters_rf["max_features"],
                        'max_leaf_nodes': self.set_up_parameters_rf["max_leaf_nodes"],
                        'min_impurity_decrease': self.set_up_parameters_rf["min_impurity_decrease"],
                        'bootstrap': self.set_up_parameters_rf["bootstrap"],
                        'class_weight': self.set_up_parameters_rf["class_weight"],
                        'ccp_alpha': self.set_up_parameters_rf["ccp_alpha"],
                        'max_samples': self.set_up_parameters_rf["max_samples"],
                        'n_jobs': self.set_up_parameters_rf["n_jobs"]
                        }
                }                          
                write_yaml_file (self.output_directory,yaml)
                executable_path = path_random_forest
                
            elif algo=="Support Vector Machine":

                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'), 
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Support Vector Machine",
                    'CONFIGURATION': 
                        {
                        'C': self.set_up_parameters_svm_svc["c"],
                        'kernel': self.set_up_parameters_svm_svc["kernel"],
                        'degree': self.set_up_parameters_svm_svc["degree"],
                        'gamma': self.set_up_parameters_svm_svc["gamma"],
                        'coef0': self.set_up_parameters_svm_svc["coef()"],
                        'shrinking': self.set_up_parameters_svm_svc["shrinking"],
                        'probability': self.set_up_parameters_svm_svc["probability"],
                        'tol': self.set_up_parameters_svm_svc["tol"],
                        'class_weight': self.set_up_parameters_svm_svc["class_weight"],
                        'max_iter': self.set_up_parameters_svm_svc["max_iter"],
                        'decision_function_shape': self.set_up_parameters_svm_svc["decision_function_shape"],
                        'break_ties': self.set_up_parameters_svm_svc["break_ties"]
                        }
                } 
                write_yaml_file (self.output_directory,yaml)
                executable_path = path_support_vector_machine
           
            elif algo=="Logistic Regression":
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'), 
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Logistic Regression",
                    'CONFIGURATION': 
                        {
                        'penalty': self.set_up_parameters_lr["penalty"],
                        'dual': self.set_up_parameters_lr["dual"],
                        'tol': self.set_up_parameters_lr["tol"],
                        'c': self.set_up_parameters_lr["c"],
                        'fit_intercept': self.set_up_parameters_lr["fit_intercept"],
                        'intercept_scaling': self.set_up_parameters_lr["intercept_scaling"],
                        'class_weight': self.set_up_parameters_lr["class_weight"],
                        'solver': self.set_up_parameters_lr["solver"],
                        'max_iter': self.set_up_parameters_lr["max_iter"],
                        'multi_class': self.set_up_parameters_lr["multi_class"],
                        'l1_ratio': self.set_up_parameters_lr["l1_ratio"],
                        'nj': self.set_up_parameters_lr["n_jobs"]
                        }
                }
                write_yaml_file (self.output_directory,yaml)
                executable_path = path_linear_regression
           
            elif algo=="Auto Machine Learning":                
                # Join the list items with commas to create a comma-separated string
                comma_separated = ','.join(self.features2include)    
                # Write the comma-separated string to a text file
                with open(os.path.join(self.output_directory, 'features.txt'), 'w') as file:
                    file.write(comma_separated)  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_POINT_CLOUD_TESTING': os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "TPOT",
                    'CONFIGURATION': 
                        {
                        'generations': self.set_up_parameters_aml["generations"],
                        'population_size': self.set_up_parameters_aml["population_size"],
                        'mutation_rate': self.set_up_parameters_aml["mutation_rate"],
                        'crossover_rate': self.set_up_parameters_aml["crossover_rate"],
                        'scoring': self.set_up_parameters_aml["scoring"],
                        'cv': self.set_up_parameters_aml["cv"],
                        'max_time_mins': self.set_up_parameters_aml["max_time_mins"],
                        'max_eval_time_mins': self.set_up_parameters_aml["max_eval_time_mins"],
                        'early_stop': self.set_up_parameters_aml["early_stop"],
                        }
                } 
                write_yaml_file (self.output_directory,yaml)
                executable_path = path_aml

            configuration_path = os.path.join(
                self.output_directory, 'algorithm_configuration.yaml'
            )
            command = [
                executable_path,
                "--i",
                configuration_path,
                "--o",
                self.output_directory,
            ]

            def close_classifier_log():
                if self._classifier_log_handle is not None:
                    self._classifier_log_handle.close()
                    self._classifier_log_handle = None

            def postprocess_predictions():
                predictions_path = os.path.join(
                    self.output_directory, 'predictions.txt'
                )
                if not os.path.isfile(predictions_path):
                    raise FileNotFoundError(
                        "The classifier finished without creating predictions.txt."
                    )

                self._set_classifier_progress(
                    95, "Postprocessing the point cloud..."
                )
                pcd_prediction = pd.read_csv(predictions_path, sep=',')
                required_prediction_columns = {"X", "Y", "Z", "Predictions"}
                missing_columns = sorted(
                    required_prediction_columns - set(pcd_prediction.columns)
                )
                if missing_columns:
                    raise ValueError(
                        f"predictions.txt is missing columns: {missing_columns}"
                    )

                raw_predictions = np.asarray(
                    pcd_prediction['Predictions'].values
                ).copy()
                if postprocessing is not None:
                    self._set_classifier_progress(
                        96, "Removing isolated class labels..."
                    )
                    coordinates = np.asarray(
                        pcd_prediction[['X', 'Y', 'Z']].values,
                        dtype=np.float64,
                    )
                    pcd_prediction['Predictions'] = spatial_majority_filter(
                        coordinates,
                        raw_predictions,
                        **postprocessing,
                    )

                if supervoxel_mode:
                    if len(pcd_prediction) != len(testing_supervoxel_ids):
                        raise ValueError(
                            "The classifier prediction count does not match the "
                            "testing supervoxel count."
                        )
                    filtered_predictions = broadcast_supervoxel_predictions(
                        testing_supervoxel_ids,
                        pcd_prediction['Predictions'].values,
                        testing_point_supervoxel_labels,
                    )
                    raw_predictions = broadcast_supervoxel_predictions(
                        testing_supervoxel_ids,
                        raw_predictions,
                        testing_point_supervoxel_labels,
                    )
                    pcd_prediction = testing_point_coordinates.copy()
                    pcd_prediction['Predictions'] = filtered_predictions

                pc_results_prediction = pycc.ccPointCloud(
                    pcd_prediction['X'],
                    pcd_prediction['Y'],
                    pcd_prediction['Z'],
                )
                pc_results_prediction.setName("Results_from_segmentation")
                if supervoxel_mode:
                    supervoxel_index = pc_results_prediction.addScalarField(
                        "Supervoxel ID", testing_point_supervoxel_labels
                    )
                    pc_results_prediction.getScalarField(
                        supervoxel_index
                    ).computeMinAndMax()
                if postprocessing is not None:
                    raw_index = pc_results_prediction.addScalarField(
                        "Classification (raw)", raw_predictions
                    )
                    pc_results_prediction.getScalarField(
                        raw_index
                    ).computeMinAndMax()
                idx = pc_results_prediction.addScalarField(
                    "Classification", pcd_prediction['Predictions']
                )
                CC = pycc.GetInstance()
                CC.addToDB(pc_results_prediction)
                pc_results_prediction.setCurrentDisplayedScalarField(idx)
                pc_results_prediction.getScalarField(
                    pc_results_prediction.getScalarFieldIndexByName(
                        "Classification"
                    )
                ).computeMinAndMax()
                CC.updateUI()
                self._set_classifier_progress(100, "Classification completed.")

                for temporary_name in (
                    'input_point_cloud_training.txt',
                    'input_point_cloud_testing.txt',
                    'predictions.txt',
                    'automl_progress.json',
                ):
                    temporary_path = os.path.join(
                        self.output_directory, temporary_name
                    )
                    if os.path.isfile(temporary_path):
                        os.remove(temporary_path)
                print("The process has been finished")
                self._close_classifier_progress()
                root.destroy()

            log_filename = (
                "automl_execution.log"
                if algo == "Auto Machine Learning"
                else "supervised_execution.log"
            )
            log_path = os.path.join(self.output_directory, log_filename)

            def classifier_failed(return_code):
                self._close_classifier_progress()
                messagebox.showerror(
                    "Classifier error",
                    execution_failure_message(
                        f"{algo} failed.",
                        log_path,
                        return_code,
                    ),
                    parent=root,
                )

            self._set_classifier_progress(
                12, f"Starting {algo}..."
            )

            if algo not in {"Random Forest", "Auto Machine Learning"}:
                try:
                    completed, log_path = run_logged_command(
                        command,
                        self.output_directory,
                        log_filename,
                        f"Supervised classification - {algo}",
                        working_directory=os.path.dirname(executable_path),
                        creationflags=getattr(
                            subprocess, "CREATE_NO_WINDOW", 0
                        ),
                        details={
                            "Algorithm": algo,
                            "Configuration file": configuration_path,
                            "Configuration": yaml,
                            "Selected features": ", ".join(
                                self.features2include
                            ),
                            "Training shape": pcd_training.shape,
                            "Testing shape": pcd_testing.shape,
                        },
                    )
                    if completed.returncode != 0:
                        classifier_failed(completed.returncode)
                        return
                    postprocess_predictions()
                except Exception as error:
                    if not os.path.isfile(log_path):
                        log_path = start_execution_log(
                            self.output_directory,
                            log_filename,
                            f"Supervised classification - {algo}",
                            command=command,
                        )
                    append_execution_exception(
                        log_path, "running or postprocessing the classifier", error
                    )
                    self._close_classifier_progress()
                    messagebox.showerror(
                        "Classifier error",
                        execution_failure_message(str(error), log_path),
                        parent=root,
                    )
                return

            is_automl = algo == "Auto Machine Learning"
            status_path = (
                os.path.join(self.output_directory, "automl_progress.json")
                if is_automl
                else None
            )
            if status_path and os.path.isfile(status_path):
                os.remove(status_path)
            log_path = start_execution_log(
                self.output_directory,
                log_filename,
                f"Supervised classification - {algo}",
                command=command,
                working_directory=os.path.dirname(executable_path),
                details={
                    "Algorithm": algo,
                    "Configuration file": configuration_path,
                    "Configuration": yaml,
                    "Selected features": ", ".join(self.features2include),
                    "Training shape": pcd_training.shape,
                    "Testing shape": pcd_testing.shape,
                },
            )
            self._classifier_log_handle = open(
                log_path, "a", encoding="utf-8", errors="replace"
            )
            creation_flags = (
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            )
            try:
                self._classifier_process = subprocess.Popen(
                    command,
                    cwd=os.path.dirname(executable_path),
                    stdout=self._classifier_log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                )
            except Exception as error:
                close_classifier_log()
                append_execution_exception(
                    log_path, f"starting the {algo} process", error
                )
                self._close_classifier_progress()
                messagebox.showerror(
                    "Classifier error",
                    execution_failure_message(str(error), log_path),
                    parent=root,
                )
                return

            started_at = time.monotonic()
            if is_automl:
                maximum_seconds = max(
                    60, self.set_up_parameters_aml["max_time_mins"] * 60
                )
                self._set_classifier_progress(
                    12, "Optimizing the classification pipeline..."
                )
            else:
                maximum_seconds = None
                self._set_classifier_progress(
                    message="Training the Random Forest model...",
                    pulse=True,
                )

            def poll_classifier():
                process = self._classifier_process
                if process is None:
                    return

                return_code = process.poll()
                if return_code is None:
                    if is_automl:
                        elapsed = time.monotonic() - started_at
                        displayed_percent = 10 + min(
                            78, 78 * elapsed / maximum_seconds
                        )
                        status_message = (
                            "Optimizing the classification pipeline..."
                        )
                        try:
                            with open(
                                status_path, "r", encoding="utf-8"
                            ) as status_file:
                                status = json.load(status_file)
                            displayed_percent = max(
                                displayed_percent,
                                float(
                                    status.get("percent", displayed_percent)
                                ),
                            )
                            status_message = status.get(
                                "message", status_message
                            )
                        except (OSError, ValueError, TypeError):
                            pass
                        self._set_classifier_progress(
                            min(displayed_percent, 94), status_message
                        )
                    root.after(500, poll_classifier)
                    return

                close_classifier_log()
                self._classifier_process = None
                append_execution_result(log_path, return_code)
                if self._classifier_cancel_requested:
                    self._classifier_cancel_requested = False
                    self._close_classifier_progress()
                    return
                if return_code != 0:
                    classifier_failed(return_code)
                    return
                try:
                    postprocess_predictions()
                except Exception as error:
                    append_execution_exception(
                        log_path, f"postprocessing the {algo} result", error
                    )
                    self._close_classifier_progress()
                    messagebox.showerror(
                        "Classifier postprocessing error",
                        execution_failure_message(str(error), log_path),
                        parent=root,
                    )

            root.after(250, poll_classifier)


        def run_algorithm_3 (self,pc_prediction_name,path_features,path_pickle):
            supervoxel_mode = t3_supervoxel_var.get()
            postprocessing = None
            if t3_postprocess_var.get():
                try:
                    postprocessing = {
                        "neighbors": int(t3_postprocess_neighbors.get()),
                        "minimum_agreement": float(
                            t3_postprocess_agreement.get()
                        ),
                        "iterations": int(t3_postprocess_iterations.get()),
                    }
                    if postprocessing["neighbors"] < 1:
                        raise ValueError("Spatial neighbors must be at least 1.")
                    if not 0.5 <= postprocessing["minimum_agreement"] <= 1.0:
                        raise ValueError(
                            "Minimum neighbor agreement must be between 0.5 and 1.0."
                        )
                    if postprocessing["iterations"] < 1:
                        raise ValueError("Smoothing iterations must be at least 1.")
                except ValueError as error:
                    messagebox.showerror(
                        "Invalid postprocessing configuration",
                        str(error),
                        parent=root,
                    )
                    return
                self.postprocessing_parameters.update(postprocessing)
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Preprocessing the pointcloud...")
            
            # Check if the selection is a point cloud
            pc_prediction=check_input(name_list,pc_prediction_name)
            
            # Convert to a pandasdataframe
            pcd_prediction=P2p_getdata(pc_prediction,False,True,True)

            prediction_point_coordinates = None
            prediction_point_supervoxel_labels = None
            prediction_supervoxel_ids = None
            if supervoxel_mode:
                if "Supervoxel ID" not in pcd_prediction.columns:
                    progress.stop()
                    messagebox.showerror(
                        "Missing supervoxels",
                        "Run Supervoxel computation on the prediction cloud first.",
                        parent=root,
                    )
                    return
                try:
                    with open(path_features, "r", encoding="utf-8") as feature_file:
                        prediction_features = [
                            name.strip()
                            for name in feature_file.read().split(',')
                            if name.strip()
                        ]
                    if "Supervoxel ID" in prediction_features:
                        raise ValueError(
                            "Supervoxel ID must not be used as a predictive feature."
                        )
                    model_columns = list(
                        dict.fromkeys(["X", "Y", "Z", *prediction_features])
                    )
                    missing = sorted(
                        set(model_columns) - set(pcd_prediction.columns)
                    )
                    if missing:
                        raise ValueError(
                            f"The prediction cloud is missing fields: {missing}"
                        )
                    prediction_point_coordinates = pcd_prediction[
                        ["X", "Y", "Z"]
                    ].copy()
                    prediction_point_supervoxel_labels = np.asarray(
                        pcd_prediction["Supervoxel ID"].values, dtype=int
                    )
                    prediction_supervoxel_ids, prediction_values = (
                        reduce_features_by_supervoxel(
                            pcd_prediction[model_columns].values,
                            prediction_point_supervoxel_labels,
                        )
                    )
                    pcd_prediction = pd.DataFrame(
                        prediction_values, columns=model_columns
                    )
                except Exception as error:
                    progress.stop()
                    messagebox.showerror(
                        "Supervoxel prediction error", str(error), parent=root
                    )
                    return
            
            # Change the status of the progress bar
            progress.setMethodTitle("Saving the point cloud for processing...")
            
            # Save the point cloud
            pcd_prediction.to_csv(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'),sep=' ',header=True,index=False)
            # YAML file
            yaml = {
                'INPUT_POINT_CLOUD': os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'),  
                'OUTPUT_DIRECTORY': self.output_directory,
                'ALGORITHM': "Prediction",
                'CONFIGURATION': 
                    {
                    'f': path_features,
                    'p': path_pickle,
                    }
            } 
            write_yaml_file (self.output_directory,yaml)    
            
            # Change the status of the progress bar
            progress.setMethodTitle("Runing the classifier...")
            
            # RUN THE COMMAND LINE
            command = [
                path_prediction,
                "--input",
                os.path.join(self.output_directory, "algorithm_configuration.yaml"),
                "--output",
                self.output_directory,
            ]
            completed, log_path = run_logged_command(
                command,
                self.output_directory,
                "supervised_prediction_execution.log",
                "Supervised model prediction",
                working_directory=os.path.dirname(path_prediction),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                messagebox.showerror(
                    "Prediction error",
                    execution_failure_message(
                        "The supervised prediction failed.",
                        log_path,
                        completed.returncode,
                    ),
                    parent=root,
                )
                return
            
            # CREATE THE RESULTING POINT CLOUD 
            progress.setMethodTitle("Postprocessing the point cloud...")
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       

            raw_predictions = np.asarray(
                pcd_prediction['Predictions'].values
            ).copy()
            if postprocessing is not None:
                progress.setMethodTitle("Removing isolated class labels...")
                coordinates = np.asarray(
                    pcd_prediction[['X', 'Y', 'Z']].values,
                    dtype=np.float64,
                )
                pcd_prediction['Predictions'] = spatial_majority_filter(
                    coordinates,
                    raw_predictions,
                    **postprocessing,
                )

            if supervoxel_mode:
                if len(pcd_prediction) != len(prediction_supervoxel_ids):
                    progress.stop()
                    messagebox.showerror(
                        "Supervoxel prediction error",
                        "The prediction count does not match the supervoxel count.",
                        parent=root,
                    )
                    return
                filtered_predictions = broadcast_supervoxel_predictions(
                    prediction_supervoxel_ids,
                    pcd_prediction['Predictions'].values,
                    prediction_point_supervoxel_labels,
                )
                raw_predictions = broadcast_supervoxel_predictions(
                    prediction_supervoxel_ids,
                    raw_predictions,
                    prediction_point_supervoxel_labels,
                )
                pcd_prediction = prediction_point_coordinates.copy()
                pcd_prediction['Predictions'] = filtered_predictions

            # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_segmentation")
            if supervoxel_mode:
                supervoxel_index = pc_results_prediction.addScalarField(
                    "Supervoxel ID", prediction_point_supervoxel_labels
                )
                pc_results_prediction.getScalarField(
                    supervoxel_index
                ).computeMinAndMax()
            if postprocessing is not None:
                raw_index = pc_results_prediction.addScalarField(
                    "Classification (raw)", raw_predictions
                )
                pc_results_prediction.getScalarField(
                    raw_index
                ).computeMinAndMax()
            idx = pc_results_prediction.addScalarField("Classification",pcd_prediction['Predictions']) 
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            pc_results_prediction.setCurrentDisplayedScalarField(idx)
            pc_results_prediction.getScalarField(pc_results_prediction.getScalarFieldIndexByName("Classification")).computeMinAndMax()
            CC.updateUI() 
            root.destroy()
            
            # Revome files
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'))
            os.remove(os.path.join(self.output_directory,'algorithm_configuration.yaml'))
            os.remove(os.path.join(self.output_directory,'predictions.txt'))
            
            # Stop the progress_Bar
            progress = pycc.ccProgressDialog()
            print("The process has been finished")
            
    def show_frame(self,root):
        self.main_frame(root)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
         
#%% RUN THE GUI
if __name__ == "__main__":        
    try:
        # START THE MAIN WINDOW        
        root = tk.Tk()
        app = GUI_mls()
        app.main_frame(root)
        root.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        root.destroy()
