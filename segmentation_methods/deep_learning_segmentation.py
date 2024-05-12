# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:54:24 2023

@author: Digi_2
"""
#%% LIBRARIES
import os
import subprocess
import sys

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
from main import P2p_getdata,get_istance,get_point_clouds_name, check_input
from main_gui import show_features_window, definition_of_labels_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

#%% ADDING PATHS FROM THE CONFIGS FILES
current_directory= os.path.dirname(os.path.abspath(__file__))
directory_bat=os.path.join(current_directory,'point_transformer')

#%% INITIAL OPERATIONS
name_list=get_point_clouds_name()
path_icon_ico= os.path.join(current_directory,'..','assets',r"logo.ico")

#%% GUI
class GUI_dl(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)
       
        # Features2include
        self.features2include=[] 
        self.values_list=[]
        self.features=[]
        
        
        # Point Transformer
        self.set_up_parameters_pt= {
            "gpu": 20000,
            "iterations": 10,
            "point_cloud":"input_point_cloud.txt"
        }
        
        # Directory to save the files (output)
        self.output_directory=os.getcwd() # The working directory
        self.file_path_pt=os.getcwd() # The working directory
        self.feature_path=os.getcwd() # The working directory
    
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
                elif tab==2:
                    t2_entry_widget.delete(0, tk.END)
                    t2_entry_widget.insert(0, self.output_directory)       
                
        # Destroy the window
        def destroy (self): 
            root.destroy ()
        
        # Load the configuration files for prediction
        def load_pytorch_dialog():
           global load_pytorch   
           file_path = filedialog.askopenfilename(filetypes=[("Model", "*.pt")])
           if file_path:
               load_pytorch = file_path  
               self.file_path_pt=file_path 
               t2_entry_pt_widget.delete(0, tk.END)
               t2_entry_pt_widget.insert(0, self.file_path_pt)   
     
        def load_features_dialog():
            global load_features 
            file_path = filedialog.askopenfilename(filetypes=[("Feature file", "*.txt")])
            if file_path:
                load_features = file_path
                self.feature_path=load_features
                t2_entry_feature_widget.delete(0, tk.END)
                t2_entry_feature_widget.insert(0, self.feature_path)                  
             
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root.title ("Supervised Deep Learning segmentation")
        root.resizable (False, False)     
        root.attributes ('-toolwindow',-1) # Remove minimize and maximize button 
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root)
        tab_control.pack(expand=1, fill="both")   
        
        tab1 = ttk.Frame(tab_control) # Create 2 tabs
        tab1.pack()
        tab_control.add(tab1, text='Training')
        tab2 = ttk.Frame(tab_control) # Create 2 tabs
        tab2.pack() 
        tab_control.add(tab2, text='Prediction')
        tab_control.pack(expand=1, fill="both")
        
        # TAB1 = TRAINING
        
        # Labels
        label_texts = [
            "Choose point cloud for training:",
            "Choose point cloud for evaluation:",
            "Features to include:",
            "GPU comsumption:",
            "Number of iterations:",
            "Choose output directory:",
        ]
        row_positions = [0,1,2,3,4,5]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,0)

        # Combobox
        t1_combo_point_cloud_training=ttk.Combobox (tab1,values=name_list)
        t1_combo_point_cloud_training.grid(column=1, row=0, sticky="e", pady=2)
        t1_combo_point_cloud_training.set("Select the point cloud used for training:")
        
        t1_combo_point_cloud_testing=ttk.Combobox (tab1,values=name_list)
        t1_combo_point_cloud_testing.grid(column=1, row=1, sticky="e", pady=2)
        t1_combo_point_cloud_testing.set("Select the point cloud used for testing:")

        # Entries        
        t1_entry_gpu = ttk.Entry(tab1, width=10)
        t1_entry_gpu.insert(0,self.set_up_parameters_pt["gpu"])
        t1_entry_gpu.grid(row=3, column=1, sticky="e", pady=2)
        
        t1_entry_iterations = ttk.Entry(tab1, width=10)
        t1_entry_iterations.insert(0,self.set_up_parameters_pt["iterations"])
        t1_entry_iterations.grid(row=4, column=1, sticky="e", pady=2)
        
        t1_entry_widget = ttk.Entry(tab1, width=30)
        t1_entry_widget.grid(row=5, column=1, sticky="e", pady=2)
        t1_entry_widget.insert(0, self.output_directory)
        
        # Buttons
        row_buttons=[5,2]  
        button_names=["...","..."]  
        _=definition_of_buttons_type_1("tab1",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog(1), lambda: show_features_window(self,name_list,t1_combo_point_cloud_training.get())],
                                       tab1,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("tab1",
                                     [lambda:check_and_display_advertisement_run_algorithm_1(self, t1_combo_point_cloud_training.get(),t1_combo_point_cloud_testing.get(), int(t1_entry_gpu.get()), int(t1_entry_iterations.get())),lambda:destroy(self)],
                                     6,
                                     tab1,
                                     1
                                     )
        
        # TAB2 = CLASSIFICATION
        
        # Labels
        label_texts = [
            "Choose the trained model (.pt model):",
            "Load the feature file:",
            "Choose point cloud for classifying:",
            "GPU comsumption:",
            "Choose output directory:",
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels_type_1 ("t2",label_texts,row_positions,tab2,0)

        # Combobox
        t2_combo_point_cloud_classification=ttk.Combobox (tab2,values=name_list)
        t2_combo_point_cloud_classification.grid (row=2,column=1,sticky="e",pady=2)
        t2_combo_point_cloud_classification.set("Select the point cloud used you want to clasify")

        # Entry
        t2_entry_gpu = ttk.Entry(tab2, width=10)
        t2_entry_gpu.insert(0,self.set_up_parameters_pt["gpu"])
        t2_entry_gpu.grid(row=3, column=1, sticky="e", pady=2)

        t2_entry_widget = ttk.Entry(tab2, width=30)
        t2_entry_widget.grid(row=4, column=1, sticky="e", pady=2)
        t2_entry_widget.insert(0, self.output_directory)
        
        t2_entry_pt_widget = ttk.Entry(tab2, width=30)
        t2_entry_pt_widget.grid(row=0, column=1, sticky="e", pady=2)
        t2_entry_pt_widget.insert(0, self.file_path_pt)
        
        t2_entry_feature_widget = ttk.Entry(tab2, width=30)
        t2_entry_feature_widget.grid(row=1, column=1, sticky="e", pady=2)
        t2_entry_feature_widget.insert(0, self.feature_path)
                
        # Buttons
        row_buttons=[4,1,0]  
        button_names=["...","...","..."]  
        _=definition_of_buttons_type_1("tab2",
                                       button_names,
                                       row_buttons,
                                       [lambda:save_file_dialog(2), lambda:load_features_dialog(), lambda:load_pytorch_dialog()],
                                       tab2,
                                       2
                                       ) 
        _=definition_run_cancel_buttons_type_1("tab2",
                                     [lambda:check_and_display_advertisement_run_algorithm_2(self,t2_combo_point_cloud_classification.get(), load_features, load_pytorch,t2_entry_gpu.get()),lambda:destroy(self)],
                                     5,
                                     tab2,
                                     1
                                     )
        def check_and_display_advertisement_run_algorithm_1(self,pc_training_name,pc_testing_name,gpu_value,epoch_value):
            
            # Start the progress_Bar
            progress = pycc.ccProgressDialog()
            progress.start()            
            progress.setMethodTitle("Preprocessing the pointcloud...")
            
            # Check if the selection is a point cloud
            pc_training=check_input(name_list,pc_training_name)
            pc_testing=check_input(name_list,pc_testing_name)
            
            # Convert to a pandasdataframe
            pcd_training=P2p_getdata(pc_training,False,True,True)
            pcd_testing=P2p_getdata(pc_testing,False,True,True)
            
            # Replace spaces with underscores in column names. It is neccesary for the DL model
            new_names = {col: col.replace(' ', '_') for col in pcd_training.columns}
            pcd_training.rename(columns=new_names, inplace=True)        
            
            # Add a new column filled with the dummy value
            dummy_value=0
            pcd_training['Dummy'] = dummy_value

            new_names = {col: col.replace(' ', '_') for col in pcd_testing.columns}
            pcd_testing.rename(columns=new_names, inplace=True)
            pcd_testing['Dummy'] = dummy_value
            
            min_value_training = pcd_training['Classification'].min()
            min_value_testing = pcd_testing['Classification'].min()
            
            progress.stop()   
            if min_value_training != 0 or min_value_testing != 0:
                # Display advertisement (in a pop-up window)
                response = messagebox.askyesno("Advertisement", "Is 0 the first class of your input?")
                if response:  # If user clicks "yes"
                    run_algorithm_1(self,pc_training,pcd_training,pcd_testing,gpu_value,epoch_value)
                else:  # If user clicks "no"
                    # Take necessary action, such as closing the GUI
                    self.destroy()
            else:
                # Run algorithm directly without displaying advertisement
                run_algorithm_1(self,pc_training,pcd_training,pcd_testing,gpu_value,epoch_value)
        
        # To run the tab1/Training of point transformer   
        def run_algorithm_1 (self,pc_training,pcd_training,pcd_testing,gpu_value,epoch_value):
            # Start the progress_Bar
            progress = pycc.ccProgressDialog()
            progress.start()            
            progress.setMethodTitle("Preprocessing the point cloud...")                                
            # Transform the point cloud into a dataframe and select only the interseting columns
            feature_selection_pcd=P2p_getdata(pc_training,False,True,True)

            # Create the features file
            comma_separated = ','.join(self.features2include)    
            with open(os.path.join(self.output_directory, 'feature_file.txt'), 'w') as file:
                file.write(comma_separated)
            # Save the point clouds and the features
            pcd_training.to_csv(os.path.join(self.output_directory, 'input_point_cloud_training.txt'),sep=' ',header=True,index=False, float_format='%.6f') 
            pcd_testing.to_csv(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'),sep=' ',header=True,index=False, float_format='%.6f')    
            
            # Save the point cloud with the features selected
            input_path_point_cloud=os.path.join(self.output_directory,"input_point_cloud.txt")
            feature_selection_pcd[self.features2include].to_csv(input_path_point_cloud,sep=' ',header=True,index=False)
            
            # Directories for the command
            input_features= os.path.join(self.output_directory,'feature_file.txt')
            output_log_file=os.path.join(self.output_directory, 'output.log')
            training_script = os.path.join(current_directory,'point_transformer','training.py ')
            output_directory_training=self.output_directory
            train_file= os.path.join(self.output_directory, 'input_point_cloud_training.txt')
            test_file= os.path.join(self.output_directory, 'input_point_cloud_testing.txt')
            gpu=gpu_value
            epoch=epoch_value
            
            # Make list with the numeric features for P.T. classification
            pcd_training=P2p_getdata(pc_training,False,True,True)
            values_list_1=[col for col in pcd_training.columns if col != 'Classification']
            df = pd.DataFrame(self.features2include, columns=["Columns"])
            values_list = df["Columns"].tolist()
            selected_indices = [str(values_list_1.index(column)) for column in self.features2include if column in values_list]
            indices_as_string = ",".join(selected_indices)
            with open(input_features, "w") as output_file:
                output_file.write(indices_as_string)
            
            # Selected features to command
            try:
                with open(input_features, "r") as file:
                    content=file.read()
            except FileNotFoundError:
                print("The file was not found or the path is incorrect.")
            features= content
            
            values_list=[col for col in pcd_training.columns]
            class_numeric = None
            for index, column in enumerate(values_list):
                if column == "Classification":
                    class_numeric = index
                    break
            classification= class_numeric
            progress.setMethodTitle("Running the classifier...")  
            os.chdir(directory_bat)            
            command = f'{directory_bat}/env/point_transformer_v1/Scripts/activate.bat && python -u "{training_script}" --train "{train_file}" --test "{test_file}" --features {features} --labels {classification} --size {gpu} --epoch {epoch} --output "{output_directory_training}" > "{output_log_file}"'
            os.system(command)
            progress.setMethodTitle("Postprocessing the point cloud...") 
            os.remove(os.path.join(self.output_directory, 'input_point_cloud.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_testing.txt'))
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_training.txt'))
            progress.stop() 
            print("The process has been finished")
        
        def check_and_display_advertisement_run_algorithm_2(self,pc_classification_name, path_features, path_pytorch,gpu_value):
            
            # Start the progress_Bar
            progress = pycc.ccProgressDialog()
            progress.start()            
            progress.setMethodTitle("Preprocessing the point cloud...") 
            
            # Check if the selection is a point cloud
            pc_classification=check_input(name_list,pc_classification_name)
            
            # Convert to a pandasdataframe
            pcd_classification=P2p_getdata(pc_classification,False,True,True)
            
            # Replace spaces with underscores in column names. It is neccesary for the DL model
            new_names = {col: col.replace(' ', '_') for col in pcd_classification.columns}
            pcd_classification.rename(columns=new_names, inplace=True)         
            
            # Add a new column filled with the dummy value
            dummy_value=0
            pcd_classification['Dummy'] = dummy_value
            
            min_value_classification =pcd_classification['Classification'].min()
            progress.stop()
            if min_value_classification != 0:
                # Display advertisement (in a pop-up window)
                response = messagebox.askyesno("Advertisement", "Is 0 the first class of your input?")
                if response:  # If user clicks "yes"
                    run_algorithm_2(self,pcd_classification,path_features, path_pytorch,gpu_value) 
                else:  # If user clicks "no"
                    # Take necessary action, such as closing the GUI
                    self.destroy()
            else:
                # Run algorithm directly without displaying advertisement
                run_algorithm_2(self,pcd_classification,path_features, path_pytorch,gpu_value)
            
        # To run the tab1/Training of point transformer   
        def run_algorithm_2 (self,pcd_classification, path_features, path_pytorch,gpu_value): 
            # Start the progress_Bar
            progress = pycc.ccProgressDialog()
            progress.start()            
            progress.setMethodTitle("Preprocessing the point cloud...") 
            
            # Save the point cloud
            pcd_classification.to_csv(os.path.join(self.output_directory, 'input_point_cloud_classification.txt'),sep=' ',header=True,index=False, float_format='%.6f') 

            
            # Directories for the command
            additional_modules_directory=os.path.sep.join(self.output_directory)+ '\cloud_classified.txt'
            sys.path.insert(0, additional_modules_directory)
            output_directory_classification=os.path.join(self.output_directory, 'cloud_classified.txt')
            output_log_file=os.path.join(self.output_directory, 'output.log')
            inference_script = os.path.join(current_directory,'point_transformer','inference.py')
            classification_file= os.path.join(self.output_directory, 'input_point_cloud_classification.txt')
            gpu=gpu_value
            
            # Selected features to command
            try:
                with open(path_features, "r") as file:
                    content=file.read()
            except FileNotFoundError:
                print("The file was not found or the path is incorrect.")
            features= content
            progress.setMethodTitle("Running the classifier...") 
            os.chdir(directory_bat)
            command = f'{directory_bat}/env/point_transformer_v1/Scripts/activate.bat && python -u "{inference_script}" --model "{path_pytorch}" --features {features} --input "{classification_file}" --size {gpu} --output "{output_directory_classification}" > "{output_log_file}"'
            os.system(command)
            progress.setMethodTitle("Postprocessing the point cloud...") 
            # CREATE THE RESULTING POINT CLOUD 
            # Load the predictions
            pcd_prediction = pd.read_csv(os.path.join(self.output_directory,'cloud_classified.txt'), sep=' ')  # Use sep='\t' for tab-separated files       

            # Select only the 'Predictions' column
            pc_results_prediction = pycc.ccPointCloud(pcd_prediction['X'], pcd_prediction['Y'], pcd_prediction['Z'])
            pc_results_prediction.setName("Results_from_prediction")
            idx = pc_results_prediction.addScalarField("Classification",pcd_prediction['prediction']) 
            
            # STORE IN THE DATABASE OF CLOUDCOMPARE
            CC = pycc.GetInstance()
            CC.addToDB(pc_results_prediction)
            pc_results_prediction.setCurrentDisplayedScalarField(idx)
            pc_results_prediction.getScalarField(pc_results_prediction.getScalarFieldIndexByName("Classification")).computeMinAndMax()
            CC.updateUI() 
            root.destroy()
            
            os.remove(os.path.join(self.output_directory, 'input_point_cloud_classification.txt'))
            os.remove(os.path.join(self.output_directory, 'cloud_classified.txt'))
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
        app = GUI_dl()
        app.main_frame(root)
        root.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        # Optionally, print detailed traceback
        traceback.print_exc()
        root.destroy()