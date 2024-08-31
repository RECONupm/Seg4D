# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:05:13 2023

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

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)
additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'
additional_modules_directory_2=script_directory
sys.path.insert(0, additional_modules_directory)
sys.path.insert(0, additional_modules_directory_2)
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input, write_yaml_file
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1, definition_of_entries_type_1


#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))
config_file=os.path.join(current_directory,r'..\configs\executables.yml')

# Read the configuration from the YAML file for the set-up
with open(config_file, 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)
path_kmeans= os.path.join(current_directory,config_data['K_MEANS'])
path_fuzzykmeans= os.path.join(current_directory,config_data['FUZZY_K_MEANS'])
path_hierarchical_clustering= os.path.join(current_directory,config_data['HIERARCHICAL_CLUSTERING'])
path_dbscan= os.path.join(current_directory,config_data['DBSCAN'])
path_optics= os.path.join(current_directory,config_data['OPTICS'])
path_prediction= os.path.join(current_directory,config_data['PREDICTION'])

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()

#%% GUI
class GUI_mlu(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
        
        # Initial features2include
        self.features2include=[]
       
        # Initial paramertes for the different algorithms
        # K-means
        self.set_up_parameters_km= {
            "clusters": 5,
            "iterations": 200,
        }
        # Fuzzy k-means
        self.set_up_parameters_fkm= {
            "clusters": 5,
            "iterations": 200,
        }
        # Hierarchical clustering   
        self.set_up_parameters_hc= {
            "n_clusters": 2,
            "metric": "euclidean",
            "compute_full_tree": "auto",
            "linkage": "ward",
            "ldt": 0.1,
            "dist_clusters": "false",
        }           
        #DBSCAN
        self.set_up_parameters_dbscan= {
            "epsilon": 0.5,
            "min_samples": 5,
        }
        #OPTICS
        self.set_up_parameters_optics= {
            "min_samples": 5,
            "epsilon": 200,
            "dist_computation": "minkowski",
            "extraction_method": "xi",
            "min_steepness": 0.05,
            "min_cluster_size": 10
        }
        
        # Variable to save if we want a cluster optimization or not
        self.optimization_strategy= (0,0,0) # 0 not optimization strategy, 1 elbow method, 2 silhouette method, 3 Calinski-Harabasz index, 4 Davies-Bouldin index,
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        
        self.file_path=os.path.join(current_directory, self.output_directory)
        self.file_path_features=os.path.join(current_directory, self.output_directory)
        self.file_path_pkl=os.path.join(current_directory, self.output_directory)
        
        #Directory to load the features file (prediction)
        self.load_features=os.path.join(current_directory, self.output_directory)
        
        #Directory to load the configuration file (prediction)
        self.load_configuration=os.path.join(current_directory, self.output_directory)
        
        #List with the features loaded (prediction)
        self.features_prediction=[]
        
    def main_frame (self, root):
               
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
                elif tab==2:
                    t2_entry_widget.delete(0, tk.END)
                    t2_entry_widget.insert(0, self.output_directory)
                    
        def destroy (self):
            root.destroy ()
        
        def save_setup_parameters (self,algo,*params): 
            if algo=="K-means":
                self.set_up_parameters_km["clusters"]=params[0]
                self.set_up_parameters_km["iterations"]=params[1]
            
            elif algo=="Fuzzy-K-means":
                self.set_up_parameters_fkm["clusters"]=params[0]
                self.set_up_parameters_fkm["iterations"]=params[1]
            
                
            elif algo=="Hierarchical-clustering":
                self.set_up_parameters_hc ["n_clusters"]=params[0]
                self.set_up_parameters_hc ["mcdbi"]=params[1]
                self.set_up_parameters_hc ["mcdbi2"]=params[2]
                self.set_up_parameters_hc ["criterion"]=params[3]
                self.set_up_parameters_hc ["ldt"]=params[4]
                self.set_up_parameters_hc ["dist_clusters"]=params[5]
                
            elif algo=="DBSCAN":
                self.set_up_parameters_dbscan["epsilon"]=params[0]
                self.set_up_parameters_dbscan["min_samples"]=params[1]
                
            elif algo== "OPTICS":
                self.set_up_parameters_optics ["n_samples"]=params[0]
                self.set_up_parameters_optics ["epsilon"]=params[1]
                self.set_up_parameters_optics ["dist_computation"]=params[2]
                self.set_up_parameters_optics ["extraction_method"]=params[3]
                self.set_up_parameters_optics ["min_steepness"]=params[4]
                self.set_up_parameters_optics ["min_cluster_size"]=params[5]
            pass
                
        # Load the configuration files for prediction
        def load_configuration_dialog():
           global load_configuration   
           file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
           if file_path:
               load_configuration = file_path 
               self.file_path_pkl=file_path 
               t2_entry_pkl_widget.delete(0, tk.END)
               t2_entry_pkl_widget.insert(0, self.file_path_pkl)
                    
        def load_features_dialog():
            global load_features 
            file_path = filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
            if file_path:
                load_features = file_path 
                self.file_path_features=file_path
                t2_entry_features_widget.delete(0, tk.END)
                t2_entry_features_widget.insert(0, self.file_path_features)
        
        def show_set_up_window (self,algo):
            def on_ok_button_click(algo):
                if algo=="K-means":
                    save_setup_parameters(self,algo, int(entry_param1_km.get()), int(entry_param2_km.get()))  
                elif algo=="Fuzzy-K-means":
                    save_setup_parameters(self,algo, int(entry_param1_fkm.get()), int(entry_param2_fkm.get()))
                elif algo=="Hierarchical-clustering":
                    if entry_param6_hc.get()=="false":
                        bool_dist_hc=False
                    else:
                       bool_dist_hc=True
                    save_setup_parameters(self,algo, int(entry_param1_hc.get()), str(entry_param2_hc.get()),str(entry_param3_hc.get()), str(entry_param4_hc.get()),float(entry_param5_hc.get()),bool_dist_hc)                         
                elif algo=="DBSCAN":
                    save_setup_parameters(self,algo, float(entry_param1_dbscan.get()), int(entry_param2_dbscan.get()))
                elif algo=="OPTICS":
                    save_setup_parameters(self,algo, int(entry_param1_optics.get()), float(entry_param2_optics.get()), str(entry_param3_optics.get()), str(entry_param4_optics.get()), float(entry_param5_optics.get()),int(entry_param6_optics.get()))                
                
                if algo=="K-means" or algo=="Fuzzy-K-means": 
                    if entry_opt.get()=="Elbow method" and var1.get()==1: # Elbow method is selected
                        self.optimization_strategy= (1,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                    elif entry_opt.get()=="Silhouette coefficient" and var1.get()==1: # Silhouette coefficient is selected
                        self.optimization_strategy= (2,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                    elif entry_opt.get()=="Calinski-Harabasz-index" and var1.get()==1: # Calinski-Harabasz-index is selected
                        self.optimization_strategy= (3,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                    elif entry_opt.get()=="Davies-Bouldin-index" and var1.get()==1: # Davies-Bouldin-index is selected
                        self.optimization_strategy= (4,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                    else:
                        self.optimization_strategy= (0,int(entry_max_clusters.get()),int(entry_min_clusters.get()))
                
                set_up_window.destroy()  # Close the window after saving parameters
                
            set_up_window = tk.Toplevel(root)
            set_up_window.title("Set Up the algorithm")
            set_up_window.resizable (False, False)
            
            # Remove minimize and maximize button 
            set_up_window.attributes ('-toolwindow',-1)
            def toggle_row():
                if var1.get() == 1:
                    label_max_clusters.config(state=tk.NORMAL)
                    entry_max_clusters.config(state=tk.NORMAL)
                    label_min_clusters.config(state=tk.NORMAL)
                    entry_min_clusters.config(state=tk.NORMAL)
                    label_opt.config(state=tk.NORMAL)
                    entry_opt.config(state=tk.NORMAL)
                else:
                    label_max_clusters.config(state=tk.DISABLED)
                    entry_max_clusters.config(state=tk.DISABLED)
                    label_min_clusters.config(state=tk.DISABLED)
                    entry_min_clusters.config(state=tk.DISABLED)
                    label_opt.config(state=tk.DISABLED)
                    entry_opt.config(state=tk.DISABLED)
            def check_uncheck_1():
                var1.set(1)

               
            if algo=="K-means":
                
                # Labels            
                label_param1_km = tk.Label(set_up_window, text="Number of clusters:")
                label_param1_km.grid(row=0, column=0, sticky=tk.W)
                label_param2_km = tk.Label(set_up_window, text="Number of iterations:")
                label_param2_km.grid(row=1, column=0, sticky=tk.W) 
                label_max_clusters = tk.Label(set_up_window, text="Maximum number of clusters:")
                label_max_clusters.grid(row=4, column=0, sticky=tk.W)
                label_max_clusters.config(state=tk.DISABLED)
                label_min_clusters = tk.Label(set_up_window, text="Minimum number of clusters:")
                label_min_clusters.grid(row=5, column=0, sticky=tk.W)   
                label_min_clusters.config(state=tk.DISABLED)
                
                # Entries
                entry_param1_km= tk.Entry(set_up_window)
                entry_param1_km.insert(0,self.set_up_parameters_km["clusters"])
                entry_param1_km.grid(row=0, column=1, sticky=tk.W) 
                
                entry_param2_km= tk.Entry(set_up_window)
                entry_param2_km.insert(0,self.set_up_parameters_km["iterations"])
                entry_param2_km.grid(row=1, column=1, sticky=tk.W)
                
                entry_max_clusters= tk.Entry(set_up_window)
                entry_max_clusters.insert(0,10)
                entry_max_clusters.grid(row=4, column=1, sticky=tk.W)
                entry_max_clusters.config(state=tk.DISABLED)
                entry_min_clusters= tk.Entry(set_up_window)
                entry_min_clusters.insert(0,1)
                entry_min_clusters.config(state=tk.DISABLED)
                entry_min_clusters.grid(row=5,column=1, sticky=tk.W)
                
                # Checkbox
                var1 = tk.IntVar()
                checkbox1 = tk.Checkbutton(set_up_window, text="Optimize the number of clusters", variable=var1, command=lambda: [check_uncheck_1(),toggle_row()])
                checkbox1.grid(row=2, column=0, sticky=tk.W)
                
                # Entries
                label_opt = tk.Label(set_up_window, text="Optimization strategy:")
                label_opt.grid(row=3, column=0, sticky=tk.W)
                label_opt.config(state=tk.DISABLED)
                features_opt = ["Elbow method", "Silhouette coefficient","Calinski-Harabasz-index","Davies-Bouldin-index"]
                entry_opt = ttk.Combobox(set_up_window,values=features_opt, state="readonly")
                entry_opt.current(0) 
                entry_opt.grid(row=3, column=1) 
                entry_opt.config(state=tk.DISABLED)


                # Buttons
                button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
                button_ok.grid(row=6, column=1)
                
            elif algo=="Fuzzy-K-means":
                
                # Labels            
                label_param1_fkm = tk.Label(set_up_window, text="Number of clusters:")
                label_param1_fkm.grid(row=0, column=0, sticky=tk.W)
                label_param2_fkm = tk.Label(set_up_window, text="Number of iterations:")
                label_param2_fkm.grid(row=1, column=0, sticky=tk.W) 
                label_max_clusters = tk.Label(set_up_window, text="Maximum number of clusters:")
                label_max_clusters.grid(row=4, column=0, sticky=tk.W)
                label_max_clusters.config(state=tk.DISABLED)
                label_min_clusters = tk.Label(set_up_window, text="Minimum number of clusters:")
                label_min_clusters.grid(row=5, column=0, sticky=tk.W)   
                label_min_clusters.config(state=tk.DISABLED)
                
                # Entries
                entry_param1_fkm= tk.Entry(set_up_window)
                entry_param1_fkm.insert(0,self.set_up_parameters_fkm["clusters"])
                entry_param1_fkm.grid(row=0, column=1, sticky=tk.W) 
                
                entry_param2_fkm= tk.Entry(set_up_window)
                entry_param2_fkm.insert(0,self.set_up_parameters_fkm["iterations"])
                entry_param2_fkm.grid(row=1, column=1, sticky=tk.W)
                
                entry_max_clusters= tk.Entry(set_up_window)
                entry_max_clusters.insert(0,10)
                entry_max_clusters.grid(row=4, column=1, sticky=tk.W)
                entry_max_clusters.config(state=tk.DISABLED)
                entry_min_clusters= tk.Entry(set_up_window)
                entry_min_clusters.insert(0,1)
                entry_min_clusters.config(state=tk.DISABLED)
                entry_min_clusters.grid(row=5,column=1, sticky=tk.W)
                
                # Checkbox
                var1 = tk.IntVar()
                checkbox1 = tk.Checkbutton(set_up_window, text="Optimize the number of clusters", variable=var1, command=lambda: [check_uncheck_1(),toggle_row()])
                checkbox1.grid(row=2, column=0, sticky=tk.W)
                
                # Entries
                label_opt = tk.Label(set_up_window, text="Optimization strategy:")
                label_opt.grid(row=3, column=0, sticky=tk.W)
                label_opt.config(state=tk.DISABLED)
                features_opt = ["Elbow method", "Silhouette coefficient","Calinski-Harabasz-index","Davies-Bouldin-index"]
                entry_opt = ttk.Combobox(set_up_window,values=features_opt, state="readonly")
                entry_opt.current(0) 
                entry_opt.grid(row=3, column=1) 
                entry_opt.config(state=tk.DISABLED)

                # Buttons
                button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
                button_ok.grid(row=6, column=1)      
            elif algo=="Hierarchical-clustering":
                label_param1_hc = tk.Label(set_up_window, text="Number of clusters:")
                label_param1_hc.grid(row=0, column=0, sticky=tk.W)            
                entry_param1_hc = tk.Entry(set_up_window)
                entry_param1_hc.insert(0,self.set_up_parameters_hc["n_clusters"])
                entry_param1_hc.grid(row=0, column=1)
                
                label_param2_hc = tk.Label(set_up_window, text="Metric for calculating the distance between istances:")
                label_param2_hc.grid(row=1, column=0, sticky=tk.W)            
                features_param2_hc = ["None","euclidean","l1","l2","manhattan","cosine"]
                entry_param2_hc = ttk.Combobox(set_up_window,values=features_param2_hc, state="readonly")
                entry_param2_hc.current(0) 
                entry_param2_hc.grid(row=1, column=1)
                
                label_param3_hc = tk.Label(set_up_window, text="Stop early the construction of the tree:")
                label_param3_hc.grid(row=2, column=0, sticky=tk.W)
                features_param3_hc = ["auto","true","false"]
                entry_param3_hc = ttk.Combobox(set_up_window,values=features_param3_hc, state="readonly")
                entry_param3_hc.current(0) 
                entry_param3_hc.grid(row=2, column=1)   
                
                label_param4_hc = tk.Label(set_up_window, text="Linkage criterion:")
                label_param4_hc.grid(row=3, column=0, sticky=tk.W)            
                features_param4_hc = ["ward","complete","average","single"]
                entry_param4_hc = ttk.Combobox(set_up_window,values=features_param4_hc, state="readonly")
                entry_param4_hc.current(0) 
                entry_param4_hc.grid(row=3, column=1)
                
                label_param5_hc = tk.Label(set_up_window, text="Linkage distance threshold:")
                label_param5_hc.grid(row=4, column=0, sticky=tk.W)            
                entry_param5_hc = tk.Entry(set_up_window)
                entry_param5_hc.insert(0,self.set_up_parameters_hc["ldt"])
                entry_param5_hc.grid(row=4, column=1)
                
                label_param6_hc = tk.Label(set_up_window, text="Compute distance between clusters:")
                label_param6_hc.grid(row=5, column=0, sticky=tk.W)            
                features_param6_hc = ["false","true"]
                entry_param6_hc = ttk.Combobox(set_up_window,values=features_param6_hc, state="readonly")
                entry_param6_hc.current(0) 
                entry_param6_hc.grid(row=5, column=1)    
                    
                # Buttons
                button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
                button_ok.grid(row=6, column=1)              
            elif algo=="DBSCAN":
               
                label_param1_dbscan = tk.Label(set_up_window, text="Epsilon (minimum distance between points of a cluster):")
                label_param1_dbscan.grid(row=0, column=0, sticky=tk.W)
                
                entry_param1_dbscan = tk.Entry(set_up_window)
                entry_param1_dbscan.insert(0,self.set_up_parameters_dbscan["epsilon"])
                entry_param1_dbscan.grid(row=0, column=1)
                
                label_param2_dbscan = tk.Label(set_up_window, text="Minimum number of points to create a cluster:")
                label_param2_dbscan.grid(row=1, column=0, sticky=tk.W)
                
                entry_param2_dbscan = tk.Entry(set_up_window)
                entry_param2_dbscan.insert(0,self.set_up_parameters_dbscan["min_samples"])
                entry_param2_dbscan.grid(row=1, column=1)    
                # Buttons
                button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
                button_ok.grid(row=2, column=1)
            elif algo=="OPTICS":
                label_param1_optics = tk.Label(set_up_window, text="Number of samples in a neighborhood to be considered as cluster:")
                label_param1_optics.grid(row=0, column=0, sticky=tk.W)
                
                entry_param1_optics = tk.Entry(set_up_window)
                entry_param1_optics.insert(0,self.set_up_parameters_optics["min_samples"])
                entry_param1_optics.grid(row=0, column=1)
                
                label_param2_optics = tk.Label(set_up_window, text="Epsilon (maximum distance between poins of a cluster):")
                label_param2_optics.grid(row=1, column=0, sticky=tk.W)
                
                entry_param2_optics = tk.Entry(set_up_window)
                entry_param2_optics.insert(0,self.set_up_parameters_optics["epsilon"])
                entry_param2_optics.grid(row=1, column=1) 


                label_param3_optics = tk.Label(set_up_window, text="Metric for distance computation:")
                label_param3_optics.grid(row=2, column=0, sticky=tk.W)
                features_optics = ["minkowski", "cityblock","cosine","euclidean", "l1","l2","manhattan", "braycurtis","canberra","chebyshev", "correlation","dice","hamming", "jaccard","kulsinski","mahalanobis", "rogerstanimoto","russellrao","seuclidean", "sokalmichener","sokalsneath","sqeuclidean","yule"]
                entry_param3_optics = ttk.Combobox(set_up_window,values=features_optics, state="readonly")
                entry_param3_optics.current(0) 
                entry_param3_optics.grid(row=2, column=1) 

                label_param4_optics = tk.Label(set_up_window, text="Extraction method:")
                label_param4_optics.grid(row=3, column=0, sticky=tk.W)
                features_2_optics = ["xi","dbscan"]
                entry_param4_optics = ttk.Combobox(set_up_window,values=features_2_optics, state="readonly")
                entry_param4_optics.current(0) 
                entry_param4_optics.grid(row=3, column=1) 


                label_param5_optics = tk.Label(set_up_window, text="Minimum steepness:")
                label_param5_optics.grid(row=4, column=0, sticky=tk.W)
                
                entry_param5_optics = tk.Entry(set_up_window)
                entry_param5_optics.insert(0,self.set_up_parameters_optics["min_steepness"])
                entry_param5_optics.grid(row=4, column=1)

                label_param6_optics = tk.Label(set_up_window, text="Minimum cluster size:")
                label_param6_optics.grid(row=5, column=0, sticky=tk.W)
                
                entry_param6_optics = tk.Entry(set_up_window)
                entry_param6_optics.insert(0,self.set_up_parameters_optics["min_cluster_size"])
                entry_param6_optics.grid(row=5, column=1)
                # Buttons
                button_ok = tk.Button(set_up_window, text="OK", command=lambda: on_ok_button_click(algo))
                button_ok.grid(row=6, column=1)                         
       
        # GENERAL CONFIGURATION OF THE GUI
        
        root.title ("Unsupervised clustering")
        root.resizable (False, False)
        
        # Remove minimize and maximize button 
        root.attributes ('-toolwindow',-1)
        
        # Create two tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")
        
        tab1 = ttk.Frame(tab_control)
        tab1.pack()
        
        tab2 = ttk.Frame(tab_control)
        tab2.pack()
        
        tab_control.add(tab1, text='Training')
        tab_control.add(tab2, text='Prediction')
        
        tooltip = tk.Label(root, text="", relief="solid", borderwidth=1)
        tooltip.place_forget()
        
        # TAB1= TRAINING
        
        # Labels
        label_texts = [
            "Choose point cloud for training:",
            "Select a clustering algorithm:",
            "Select the features to include:",
            "Choose output directory:"
        ]
        row_positions = [0,1,2,3]        
        definition_of_labels_type_1 ("t2",label_texts,row_positions,tab1,0)
        
        # Combobox
        t1_combo_point_cloud=ttk.Combobox (tab1,values=name_list)
        t1_combo_point_cloud.grid(row=0,column=1, sticky="e", pady=2)
        t1_combo_point_cloud.set("Not selected")
        
        algorithms = ["K-means", "Fuzzy-K-means","Hierarchical-clustering","DBSCAN","OPTICS"]
        t1_combo_algo=ttk.Combobox (tab1,values=algorithms, state="readonly")
        t1_combo_algo.current(0)
        t1_combo_algo.grid(column=1, row=1, sticky="e", pady=2)
        t1_combo_algo.set("Not selected")
        
        # Entry
        t1_entry_widget = ttk.Entry(tab1, width=30)
        t1_entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.file_path)
        
        # Buttons  
        row_buttons=[1,2,3]  
        button_names=["Set-up","...","..."]  
        _=definition_of_buttons_type_1("tab2",
                                       button_names,
                                       row_buttons,
                                       [lambda: show_set_up_window(self,t1_combo_algo.get()),lambda: show_features_window(self,name_list,t1_combo_point_cloud.get()),lambda:save_file_dialog(1)],
                                       tab1,
                                       2
                                       ) 
        
        _=definition_run_cancel_buttons_type_1("tab2",
                                     [lambda: run_algorithm_1(self,t1_combo_algo.get(),t1_combo_point_cloud.get()),lambda:destroy(self)],
                                     4,
                                     tab1,
                                     1
                                     )
        
        # TAB2= PREDICTION
        
        # Labels 
        label_texts = [
            "Choose point cloud for prediction:",
            "Load feature file:",
            "Load configuration file:",
            "Choose output directory:"
        ]
        row_positions = [0,1,2,3]        
        definition_of_labels_type_1 ("t2",label_texts,row_positions,tab2,0)

        # Combobox
        t2_combo_point_cloud=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud.grid(row=0,column=1, sticky="e", pady=2)
        t2_combo_point_cloud.set("Not selected")
        
        # Entry
        t2_entry_features_widget = ttk.Entry(tab2, width=30)
        t2_entry_features_widget.grid(row=1, column=1, sticky="e", pady=2)
        t2_entry_features_widget.insert(0, self.file_path_features)
        t2_entry_pkl_widget = ttk.Entry(tab2, width=30)
        t2_entry_pkl_widget.grid(row=2, column=1, sticky="e", pady=2)
        t2_entry_pkl_widget.insert(0, self.file_path_pkl)
        t2_entry_widget = ttk.Entry(tab2, width=30)
        t2_entry_widget.grid(row=3, column=1, sticky="e", pady=2)
        t2_entry_widget.insert(0, self.file_path)
        
        # Buttons
        row_buttons=[1,2,3]  
        button_names=["...","...","..."]  
        _=definition_of_buttons_type_1("tab2",
                                       button_names,
                                       row_buttons,
                                       [lambda: load_features_dialog(),lambda: load_configuration_dialog(),lambda: save_file_dialog(2)],
                                       tab2,
                                       2 ) 
        
        _=definition_run_cancel_buttons_type_1("tab2",
                                     [lambda:run_algorithm_2(self,t2_combo_point_cloud.get(),load_features,load_configuration),lambda:destroy(self)],
                                     4,
                                     tab2,
                                     1
                                     ) 
        
        
    
            
        def run_algorithm_1 (self,algo,pc_training_name): 
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Segmenting the point cloud...")
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            
            # Convert to a pandasdataframe
            pcd_training=P2p_getdata(pc_training,False,True,True)
            
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

            if algo=="K-means":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "K-means",
                    'OPTIMIZATION_STRATEGY': self.optimization_strategy,
                    'CONFIGURATION': 
                        {
                        'clusters': self.set_up_parameters_km["clusters"],
                        'iterations': self.set_up_parameters_km["iterations"]
                        }
                }                          
                write_yaml_file (self.output_directory,yaml)
                command = path_kmeans + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory +'"'
            
            elif algo=="Fuzzy-K-means":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Fuzzy-K-means",
                    'OPTIMIZATION_STRATEGY': self.optimization_strategy,
                    'CONFIGURATION': 
                        {
                        'clusters': self.set_up_parameters_fkm["clusters"],
                        'iterations': self.set_up_parameters_fkm["iterations"]
                        }
                }                          
                write_yaml_file (self.output_directory,yaml)
                command = path_fuzzykmeans + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory +'"'
                
            elif algo=="Hierarchical-clustering":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "Hierarchical-clustering",
                    'CONFIGURATION': 
                        {
                        'n_clusters': self.set_up_parameters_hc["n_clusters"],
                        'metric': self.set_up_parameters_hc["metric"],
                        'compute_full_tree': self.set_up_parameters_hc["compute_full_tree"],
                        'linkage': self.set_up_parameters_hc["linkage"],
                        'ldt': self.set_up_parameters_hc["ldt"],
                        'dist_clusters': self.set_up_parameters_hc["dist_clusters"],
                        }
                }                           
                write_yaml_file (self.output_directory,yaml)
                command = path_hierarchical_clustering + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory +'"'
                
            elif algo=="DBSCAN":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "DBSCAN",
                    'CONFIGURATION': 
                        {
                        'epsilon': self.set_up_parameters_dbscan["epsilon"],
                        'min_samples': self.set_up_parameters_dbscan["min_samples"]
                        }
                }                          
                write_yaml_file (self.output_directory,yaml)
                command = path_dbscan + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory +'"'
                
            elif algo=="OPTICS":
                  
                # YAML file
                yaml = {
                    'INPUT_POINT_CLOUD_TRAINING': os.path.join(self.output_directory, 'input_point_cloud_training.txt'),  
                    'INPUT_FEATURES': os.path.join(self.output_directory, 'features.txt'),
                    'OUTPUT_DIRECTORY': self.output_directory,
                    'ALGORITHM': "OPTICS",
                    'CONFIGURATION': 
                        {
                        'min_samples': self.set_up_parameters_optics["min_samples"],
                        'epsilon': self.set_up_parameters_optics["epsilon"],
                        'dist_computation': self.set_up_parameters_optics["dist_computation"],
                        'extraction_method': self.set_up_parameters_optics["extraction_method"],
                        'min_steepness': self.set_up_parameters_optics["min_steepness"],
                        'min_cluster_size': self.set_up_parameters_optics["min_cluster_size"],
                        }
                }                    
                write_yaml_file (self.output_directory,yaml)
                command = path_optics + ' --i "' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + '" --o "' + self.output_directory + '"'
            # RUN THE COMMAND LINE
            os.system(command)            

            # CREATE THE RESULTING POINT CLOUD 
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       
            # # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_clustering")
            idx = pc_results_prediction.addScalarField("Clusters",pcd_prediction['Predictions']) 
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            pc_results_prediction.setCurrentDisplayedScalarField(idx)
            pc_results_prediction.getScalarField(pc_results_prediction.getScalarFieldIndexByName("Clusters")).computeMinAndMax()
            CC.updateUI() 
            root.destroy()
            # Revome files
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_training.txt'))
            os.remove(os.path.join(self.output_directory,'algorithm_configuration.yaml'))
            os.remove(os.path.join(self.output_directory,'predictions.txt'))
            # Stop the progress bar
            progress.stop()
            print("The process has been finished")
    

        def run_algorithm_2 (self,pc_prediction_name,path_features,path_pickle):
            
            # Start the progress bar
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Predicting the clusters of the point cloud...")
            
            # Check if the selection is a point cloud
            pc_prediction=check_input(name_list,pc_prediction_name)
            
            # Convert to a pandasdataframe
            pcd_prediction=P2p_getdata(pc_prediction,False,True,True)
            
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
            
            # RUN THE COMMAND LINE
            command = path_prediction + ' --i ' + os.path.join(self.output_directory,'algorithm_configuration.yaml') + ' --o ' + self.output_directory
            os.system(command)

            
            # CREATE THE RESULTING POINT CLOUD 
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'predictions.txt'), sep=',')  # Use sep='\t' for tab-separated files       

            # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_clustering")
            idx = pc_results_prediction.addScalarField("Clusters",pcd_prediction['Predictions']) 
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            pc_results_prediction.setCurrentDisplayedScalarField(pc_results_prediction.getScalarFieldIndexByName("Clusters"))
            pc_results_prediction.getScalarField(pc_results_prediction.getScalarFieldIndexByName("Clusters")).computeMinAndMax()
            CC.updateUI() 
            root.destroy()
            
            # Revome files
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_prediction.txt'))
            os.remove(os.path.join(self.output_directory,'algorithm_configuration.yaml'))
            # Stop the progress bar
            progress.stop()
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
        app = GUI_mlu()
        app.main_frame(root)
        root.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        root.destroy()


