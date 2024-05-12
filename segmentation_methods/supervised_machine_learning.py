# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:58:50 2024

@author: LuisJa
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
            root.destroy ()
        
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

                set_up_window.destroy()  # Close the window after saving parameters
                
            # Setup window
            set_up_window = tk.Toplevel(root)
            set_up_window.title("Set Up the algorithm")
            set_up_window.resizable (False, False)
            # Remove minimize and maximize button 
            set_up_window.attributes ('-toolwindow',-1)
               
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
                selected_element = ["accuracy"]
                aml_comboboxes =definition_of_combobox_type_1 ("aml",combobox_insert,row_positions, selected_element,set_up_window,2)
                                              
                # Buttons  
                _=definition_ok_cancel_buttons_type_1("aml",[lambda: on_ok_button_click(algo),None],9,set_up_window,1) 
                
            
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Supervised Machine Learning segmentation")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button 
        
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
            "Choose output directory:"
        ]
        row_positions = [0,1,2,3,4]        
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
                                     5,
                                     tab2,
                                     1
                                     ) 
    
       
        # TAB3 = PREDICTION

        # Labels
        label_texts = [
            "Choose point cloud for prediction:",
            "Load feature file:",
            "Load pkl file:",
            "Select the features to include:",
            "Choose a directory for temporal files:"
        ]
        row_positions = [0,1,2,3]        
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

        # Buttons
        row_buttons=[1,2,3]  
        button_names=["...","...","..."]  
        _=definition_of_buttons_type_1("tab3",button_names,row_buttons,[lambda: load_features_dialog(),lambda: load_configuration_dialog(),lambda:save_file_dialog(3)],tab3,2 ) 
        
        _=definition_run_cancel_buttons_type_1("tab3",
                                     [lambda:run_algorithm_3(self,t3_combo_1.get(),load_features,load_configuration),lambda:destroy(self)],
                                     4,
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
            command = path_optimal_flow + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            print (command)
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            os.system(command)
            os.chdir(current_directory)
            
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
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Preprocessing the pointcloud...")
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            pc_testing=check_input(name_list,pc_testing_name)
            

            # Convert to a pandasdataframe
            pcd_training=P2p_getdata(pc_training,False,True,True)
            pcd_testing=P2p_getdata(pc_testing,False,True,True)            

            # Error control to prevent not algorithm for the training
            if algo=="Not selected":
                raise RuntimeError ("Please select and algorithm for the training")
            else: 
                
                # Create the features file
                comma_separated = ','.join(self.features2include)    
                with open(os.path.join(self.output_directory, 'features.txt'), 'w') as file:
                    file.write(comma_separated)
                # Save the point clouds and the features
                pcd_training.to_csv(os.path.join(self.output_directory, 'input_point_cloud_training.txt'),sep=' ',header=True,index=False)
                pcd_testing.to_csv(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),sep=' ',header=True,index=False)   

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
                command = path_random_forest + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
                
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
                command = path_support_vector_machine + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
           
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
                command = path_linear_regression + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
           
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
                command = path_aml + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            
            # RUN THE COMMAND LINE
            progress.setMethodTitle("Running the classifier...")
            os.system(command)    
            # CREATE THE RESULTING POINT CLOUD 
            progress.setMethodTitle("Postprocessing the point cloud...")
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       
            # # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_segmentation")
            idx = pc_results_prediction.addScalarField("Classification",pcd_prediction['Predictions']) 
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            pc_results_prediction.setCurrentDisplayedScalarField(idx)
            pc_results_prediction.getScalarField(pc_results_prediction.getScalarFieldIndexByName("Classification")).computeMinAndMax()
            CC.updateUI()
            root.destroy()
            # Revome files
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_training.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'))
            os.remove(os.path.join(self.output_directory,'predictions.txt'))
            
            # Stop the progress_Bar
            progress = pycc.ccProgressDialog()
            print("The process has been finished")    


        def run_algorithm_3 (self,pc_prediction_name,path_features,path_pickle):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Preprocessing the pointcloud...")
            
            # Check if the selection is a point cloud
            pc_prediction=check_input(name_list,pc_prediction_name)
            
            # Convert to a pandasdataframe
            pcd_prediction=P2p_getdata(pc_prediction,False,True,True)
            
            # Change the status of the progress bar
            progress.setMethodTitle("Saving the point cloud for processing...")
            
            # Save the point cloud
            pcd_prediction.to_csv(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'),sep=' ',header=True,index=False)
            command= path_prediction

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
            command = path_prediction + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            os.system(command) 
            
            # CREATE THE RESULTING POINT CLOUD 
            progress.setMethodTitle("Postprocessing the point cloud...")
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       

            # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_segmentation")
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