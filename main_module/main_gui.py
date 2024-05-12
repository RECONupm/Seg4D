# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:19:01 2024

@author: LuisJa
"""
#%% LIBRARIES
import os
import sys
#CloudCompare Python Plugin
import cccorelib
import pycc

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-2])+ '\main_module'

sys.path.insert(0, additional_modules_directory)
from main import P2p_getdata,get_istance,get_point_clouds_name

#%% ADD ICON
current_directory= os.path.dirname(os.path.abspath(__file__))
path_icon_ico= os.path.join(current_directory,'..','assets',r"logo.ico")

#%% FUNCTIONS

def show_features_window(self,name_list,training_pc_name="Not selected",excluded_feature="Classification", single_selection_mode=False):
    """
    This fuction allows to render a form for selecting the features of the point cloud
        
    Parameters
    ----------
    self (self): allow to store the data outside this window. Examples: self.features2include
    
    name_list (list) : list of available point clouds
    
    training_pc_name (str): target point cloud. Default: "Not selected"   
    
    preselected_features (list): list of the preselected features. To load it while starting the window
    
    excluded_feature (str): name of the feeature that need to be excluded from the selection. Default: "Classification"
    
    single_aselection_mode (bool): true if the window just only allow to select one feature. False the other side.
   
    Returns
    -------

    """
    # Functions
    def update_checkbuttons(selected_var):
        if single_selection_mode:
            for var in checkbuttons_vars:
                if var != selected_var:
                    var.set(False)
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def on_features_frame_configure(event):
        # Update the scroll region to encompass the size of the inner frame
        canvas.configure(scrollregion=canvas.bbox("all"))         
 
    def select_all_checkbuttons(checkbuttons_vars): 
        for var in checkbuttons_vars:
            var.set(True)
    def unselect_all_checkbuttons(checkbuttons_vars):
        for var in checkbuttons_vars:
            var.set(False)
            
    def ok_features_window():
        features2include = [value for value, var in zip(values_list, checkbuttons_vars) if var.get()]
        if single_selection_mode and len(features2include) > 1:
            print("Error: You need to select only one feature for the compation of the indexes")
            return
        features2include = [value for value, var in zip(values_list, checkbuttons_vars) if var.get()]
        if len(features2include) == 1:
            print("The feature " + str(features2include) + " has been included for the training")
        elif len(features2include)==0:
            print("There are not features selected")
        else:
            print("The features " + str(features2include) + " have been included for the training")    
        self.features2include = features2include
        feature_window.destroy()    
    
    def cancel_features_window():
        feature_window.destroy()
    
    def load_features_from_file():
        
        # Open a file dialog to select the file
        filename = filedialog.askopenfilename(title="Select a file", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if filename:
            with open(filename, "r") as file:
                # Read the file and split by comma
                features_from_file = file.read().split(',')
    
                # Update checkbuttons based on the loaded features
                for var, value in zip(checkbuttons_vars, values_list):
                    if value in features_from_file:
                        var.set(True)
                    else:
                        var.set(False)
                        
    def save_selected_features():
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return  # User cancelled, just return
    
        selected_features = [value for value, var in zip(values_list, checkbuttons_vars) if var.get()]
        with open(filepath, 'w') as file:
            file.write(','.join(selected_features))
        
    # Some lines of code for ensuring thee proper selection of the point cloud
    if training_pc_name=="Not selected":
        raise RuntimeError("Please select a point cloud to evaluate the features")
    CC = pycc.GetInstance()
    entities = CC.getSelectedEntities()[0]    
    type_data, number = get_istance()
    if type_data=='point_cloud':
        pc_training=entities
    else:
        for ii, item in enumerate(name_list):
            if item == training_pc_name:
                pc_training = entities.getChild(ii)
                break
            
    # Transform the point cloud to a pandas dataframe
    pcd_training = P2p_getdata(pc_training, False, True, True)
    
    #GUI
    feature_window = tk.Toplevel()
    feature_window.title("Features of the point cloud")
    feature_window.iconbitmap(path_icon_ico)
    feature_window.resizable (False, False) 
    
    # Set up the protocol for the window close button (the "X" button)
    feature_window.protocol("WM_DELETE_WINDOW", ok_features_window)
    
    # Checkbutton
    checkbutton_frame = tk.Frame(feature_window)
    checkbutton_frame.pack(side="left", fill="y")

    # Canvas
    canvas = tk.Canvas(checkbutton_frame)
    features_frame = tk.Frame(canvas)
    # Bind the event to the features frame
    features_frame.bind("<Configure>", on_features_frame_configure)  
    
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=features_frame, anchor="nw")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Scrollbar
    scrollbar = tk.Scrollbar(checkbutton_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Buttons  
    
    if single_selection_mode is False:
        button_text =["Select All","Unselect All","Load Features","Save Features"]
        command_list = [lambda: select_all_checkbuttons(checkbuttons_vars),
                        lambda: unselect_all_checkbuttons(checkbuttons_vars),
                        load_features_from_file,
                        save_selected_features
                        ]    
        button_frame = tk.Frame(feature_window)
        button_frame.pack(side="right", fill="y")
        _=definition_of_buttons_type_2("features_windows", button_text, command_list, button_frame) 
    
    # Render the features within the check button and set initial selection
    values_list = [col for col in pcd_training.columns if col != excluded_feature]
    checkbuttons_vars = []
    for value in values_list:
        var = tk.BooleanVar(value=value in self.features2include)
        checkbuttons_vars.append(var)
        ttk.Checkbutton(features_frame, text=value, variable=var).pack(anchor="w")
        
def definition_of_labels_type_1(header, label_texts, row_positions, window, column=0, disabled_index=None):
    """
    This function allows to create the labels of a tab

    Parameters
    ----------
    header (str): name of the label. It will be as: header_label_idx. Where idx is the row on which the label appears.
                  I.e t1_label_1 the header is t1 and the row is 1 for this element

    label_texts (list): a list with the name of each label

    row_positions (list): a list with the position (rows) of each label text

    window (tk window): the window on which the information will be rendered

    column (int): the column to place the labels. Default: 0

    disable_labels (bool): If True, the labels will be disabled. Default: False

    Returns
    -------

    """
    labels = {}  # Dictionary to store labels
    for idx, (text, row) in enumerate(zip(label_texts, row_positions)):
        label_name = f"{header}_label_{idx}"
        labels[label_name] = ttk.Label(window, text=text)
        labels[label_name].grid(column=column, row=row, pady=2, sticky="w")
        
        if idx == disabled_index:
            labels[label_name].config(state=tk.DISABLED)
    
    return labels

def definition_of_entries_type_1 (header,entry_insert,row_positions,window,column=1):
    
    """
    This fuction allows to create the entries of a tab
        
    Parameters
    ----------
    header (str): name of the entry. It will be as: header_entry_idx. Where idx is the row on which the label appears. I.e t1_entry_1 the header is t1 and the row is 1 for this element

    entry_insert (list): a list with the insert of each entry
    
    row_positions (list): a list with the position (rows) of each entry text

    window (tk window): the window on which the information will be rendered
    
    column (int): the column to place the labels. Default: 1
      
    Returns
    -------
    
    entry_list (dict): dictionary with the name of the elements. This is because if you have rows (0,2,4) you can access as entry_dict [2] for the second element.

    """      
    entry_dict = {}  # Dictionary to store instances of entry mapped to row positions

    for row_idx, (row_data, insert_value) in enumerate(zip(row_positions, entry_insert)):        
        entry = tk.Entry(window, name=f"{header}_entry_{row_idx}",width=10) 
        entry.grid(column=column, row=row_data, sticky="e")
        entry.insert(0,insert_value)
        entry_dict[row_data] = entry  # Map row position to the entry instance
    return entry_dict

def definition_of_combobox_type_1 (header,combobox_insert,row_positions, selected_element,window,column=1):
    
    """
    This fuction allows to create the comboboxes of a tab
        
    Parameters
    ----------
    header (str): name of the combobox. It will be as: header_combobox_idx. Where idx is the row on which the combobox appears. I.e t1_combobox_1 the header is t1 and the row is 1 for this element

    combobox_insert (list): a list with the insert of each combobox. This list could include another list with the options of the combobox
    
    row_positions (list): a list with the position (rows) of each combobox text

    window (tk window): the window on which the information will be rendered
    
    column (int): the column to place the combobox. Default: 1
      
    Returns
    -------
    
    combobox_list (dict): dictionary with the name of the elements. This is because if you have rows (0,2,4) you can access as comboboxes_dict [2] for the second element.

    """      
  
    comboboxes_dict = {}  # Dictionary to store instances of ComboBoxes mapped to row positions
    
    for row_idx, (row_data, options) in enumerate(zip(row_positions, combobox_insert)):
        combobox = ttk.Combobox(window, name=f"{header}_combobox_{row_idx}", width=10, values=options)
        combobox.grid(column=column, row=row_data, sticky="e")
        initial_selection = selected_element[row_idx] if row_idx < len(selected_element) else options[0]
        combobox.current(options.index(initial_selection))  # Set initial selection
        comboboxes_dict[row_data] = combobox  # Map row position to the ComboBox instance
    return comboboxes_dict

def definition_of_checkbutton_type_1(header, row_positions, window, initial_states=None, column=1):
    """
    This function allows to create the checkbuttons of a tab.
    
    Parameters
    ----------
    header (str): Name of the checkbutton.
    checkbutton_insert (list): A list with the text for each checkbutton.
    row_positions (list): A list with the position (rows) of each checkbutton.
    window (tk window): The window on which the information will be rendered.
    initial_states (list of int): A list with the initial state (0 or 1) of each checkbutton. Default: None.
    column (int): The column to place the checkbutton. Default: 1.
      
    Returns
    -------
    checkbutton_dict (dict): Dictionary with the checkbutton instances, keyed by row position.
    checkbutton_vars(dict): Dictionary to store BooleanVar instances mapped to row positions
    """      
    checkbutton_dict = {}
    checkbutton_vars = {} 

    if initial_states is None:
        initial_states = [0] * len(row_positions)

    if len(initial_states) != len(row_positions):
        raise ValueError("Initial states and row positions must be of the same length.")

    for row_idx, row_data in enumerate(row_positions):
        state_var = tk.BooleanVar(value=bool(initial_states[row_idx]))
    
        checkbutton = tk.Checkbutton(window, variable=state_var)
        checkbutton.grid(column=column, row=row_data, sticky="e")

        checkbutton_vars[row_data] = state_var
        checkbutton_dict[row_data] = checkbutton

    return checkbutton_dict, checkbutton_vars

def definition_of_buttons_type_1 (header, button_texts, row_positions, command_list, window, column=1):
    """
    This function allows to create buttons of a tab.
    
    Parameters
    ----------
    header (str): Name of the button. It will be as: header_button_idx. Where idx is the row on which the button appears.
    
    button_texts (list): A list with the text for each button.
    
    row_positions (list): A list with the position (rows) of each button.
    
    command_list (list): A list of functions (commands) that are executed when buttons are clicked.
    
    window (tk window): The window on which the information will be rendered.
    
    column (int): The column to place the button. Default: 1
    
    Returns
    -------
    buttons_dict (dict): Dictionary with the name of the buttons mapped to their row positions.
    """

    buttons_dict = {}  # Dictionary to store instances of Buttons mapped to row positions

    for row_idx, button_text in enumerate(button_texts):
        button_name = f"{header}_button_{row_idx}"
        command = command_list[row_idx] if row_idx < len(command_list) else None
        button = tk.Button(window, name=button_name, text=button_text, command=command)
        button.grid(column=column, row=row_positions[row_idx], sticky="e")
        buttons_dict[row_positions[row_idx]] = button  # Map row position to the Button instance
    
    return buttons_dict

def definition_ok_cancel_buttons_type_1(header, commands, row_position, window, column=1):
    """
    Create OK and Cancel buttons with specific behavior.
    
    Parameters
    ----------
    header (str): Prefix for the button names. The buttons are named as header_ok_button and header_cancel_ok_button
    
    commands (list): A list with two elements, the first is the command for the OK button,and the second is for the Cancel button. If any is None, the button is not created.
    
    row_position (int): The row position for the buttons.
    
    window (tk window): The window on which the buttons will be rendered.
    
    column (int): The column to place the buttons. Default: 1
       
    Returns
    -------
    
    dict: A dictionary with the created button objects.
    """
    
    buttons_dict = {}
    ok_command, cancel_command = commands
    
    # Create OK button if command is not None
    if ok_command is not None:
        ok_button_name = f"{header}_ok_button"
        ok_button = tk.Button(window, text="OK", command=ok_command, name=ok_button_name )
        ok_button.grid(column=column, row=row_position, sticky="e", padx=100)
        buttons_dict[ok_button_name] = ok_button

    # Create Cancel button if command is not None
    if cancel_command is not None:
        cancel_button_name=f"{header}_cancel_button"
        cancel_button = tk.Button(window, text="Cancel", command=cancel_command, name=cancel_button_name)
        cancel_button.grid(column=column, row=row_position, sticky="e")
        buttons_dict[cancel_button_name] = cancel_button

    return buttons_dict

def definition_run_cancel_buttons_type_1(header, commands, row_position, window, column=1):
    """
    Create Run and Cancel buttons with specific behavior.
    
    Parameters
    ----------
    header (str): Prefix for the button names. The buttons are named as header_run_button and header_cancel_ok_button
    
    commands (list): A list with two elements, the first is the command for the Run button,and the second is for the Cancel button. If any is None, the button is not created.
    
    row_position (int): The row position for the buttons.
    
    window (tk window): The window on which the buttons will be rendered.
    
    column (int): The column to place the buttons. Default: 1
       
    Returns
    -------
    
    dict: A dictionary with the created button objects.
    """
    
    buttons_dict = {}
    ok_command, cancel_command = commands
    
    # Create OK button if command is not None
    if ok_command is not None:
        run_button_name = f"{header}_run_button"
        run_button = tk.Button(window, text="Run", command=ok_command, name=run_button_name )
        run_button.grid(column=column, row=row_position, sticky="e", padx=100)
        buttons_dict[run_button_name] = run_button

    # Create Cancel button if command is not None
    if cancel_command is not None:
        cancel_button_name=f"{header}_cancel_button"
        cancel_button = tk.Button(window, text="Cancel", command=cancel_command, name=cancel_button_name)
        cancel_button.grid(column=column, row=row_position, sticky="e")
        buttons_dict[cancel_button_name] = cancel_button

    return buttons_dict

def definition_of_buttons_type_2 (header, button_texts, command_list, window):
    """
    This function allows to create buttons of a tab in pack mode.
    
    Parameters
    ----------
    header (str): Name of the button. It will be as: header_button_idx. Where idx is the row on which the button appears.
    
    button_texts (list): A list with the text for each button.    
    
    command_list (list): A list of functions (commands) that are executed when buttons are clicked.
    
    window (tk window): The frame on which the information will be rendered.
    
    Returns
    -------
    """    
    for row_idx, button_text in enumerate(button_texts):
        button_name = f"{header}_button_{row_idx}"
        command = command_list[row_idx] if row_idx < len(command_list) else None
        button = tk.Button(window, name=button_name, text=button_text, command=command)
        button.pack(side="top")
