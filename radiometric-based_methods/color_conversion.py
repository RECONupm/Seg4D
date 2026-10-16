# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:00:29 2023

@author: Luisja
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cccorelib
import pycc
import numpy as np
import os
import sys
import traceback

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS

script_directory = os.path.abspath(__file__)
additional_modules_directory = os.path.join(
    os.path.dirname(os.path.dirname(script_directory)),
    'main_module'
)
sys.path.insert(0, additional_modules_directory)
from main import get_point_clouds_name, check_input
from main_gui import show_features_window, definition_of_labels_type_1,definition_of_entries_type_1, definition_of_combobox_type_1,definition_ok_cancel_buttons_type_1,definition_run_cancel_buttons_type_1, definition_of_buttons_type_1

CC = pycc.GetInstance()

#%% GUI

class GUI_cc(tk.Frame):
    def __init__(self, master=None, **kwargs): # Initial parameters. It is in self because we can update during the interaction with the user
        super().__init__(master, **kwargs)      
        self.output_directory=os.getcwd() # Directory to save the files (output)
    
    def get_rgb(self, pc):
        """Return RGB values as an N x 3 float array.

        Native CloudCompare colors are preferred. Scalar fields named R, G and
        B are retained as a fallback for older workflows.
        """
        colors = pc.colors()
        if colors is not None:
            rgb = np.asarray(colors, dtype=np.float32)
            if rgb.ndim != 2 or rgb.shape[0] != pc.size() or rgb.shape[1] < 3:
                raise ValueError("The native color table has an invalid shape")
            return rgb[:, :3].copy()

        channels = []
        for name in ("R", "G", "B"):
            index = pc.getScalarFieldIndexByName(name)
            if index < 0:
                raise ValueError(
                    "The selected point cloud has neither native RGB colors "
                    "nor scalar fields named R, G and B"
                )
            channel = np.asarray(
                pc.getScalarField(index).asArray(),
                dtype=np.float32
            ).reshape(-1)
            channels.append(channel.copy())

        rgb = np.column_stack(channels)
        if rgb.shape[0] != pc.size():
            raise ValueError("The RGB scalar fields do not match the point-cloud size")
        return rgb

    def rgb_to_hsv(self, rgb):
        """Vectorized RGB-to-HSV conversion.

        Hue and saturation are returned in [0, 1]. Value retains the same
        numeric scale as the input RGB values.
        """
        maximum = np.max(rgb, axis=1)
        minimum = np.min(rgb, axis=1)
        delta = maximum - minimum

        value = maximum
        saturation = np.zeros_like(maximum)
        nonzero_value = maximum != 0
        saturation[nonzero_value] = delta[nonzero_value] / maximum[nonzero_value]

        hue = np.zeros_like(maximum)
        chromatic = delta != 0

        red = chromatic & (maximum == rgb[:, 0])
        green = chromatic & (maximum == rgb[:, 1])
        blue = chromatic & (maximum == rgb[:, 2])

        hue[red] = ((rgb[red, 1] - rgb[red, 2]) / delta[red]) % 6.0
        hue[green] = (rgb[green, 2] - rgb[green, 0]) / delta[green] + 2.0
        hue[blue] = (rgb[blue, 0] - rgb[blue, 1]) / delta[blue] + 4.0
        hue /= 6.0

        return hue, saturation, value

    def add_or_replace_scalar_field(self, pc, name, values):
        """Create a scalar field using the API supported by PythonRuntime."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != pc.size():
            raise ValueError(f"Scalar field '{name}' does not match the cloud size")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Scalar field '{name}' contains non-finite values")

        existing_index = pc.getScalarFieldIndexByName(name)
        if existing_index >= 0:
            pc.deleteScalarField(existing_index)

        index = pc.addScalarField(name)
        if index < 0:
            raise RuntimeError(f"CloudCompare could not create scalar field '{name}'")

        scalar_field = pc.getScalarField(index)
        scalar_field.asArray()[:] = values
        scalar_field.computeMinAndMax()
        return index

    # Create scalar fields for the selected color conversions
    def color_conversion(self, selected_algorithms, pc, rgb, progress):
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]
        last_index = -1
        total_steps = len(selected_algorithms) * 4
        completed_steps = 0

        def advance(info):
            nonlocal completed_steps
            completed_steps += 1
            progress.setInfo(info)
            progress.update(15.0 + 80.0 * completed_steps / total_steps)

        for algorithm in selected_algorithms:
            print(f"Converting to: {algorithm}")
            progress.setInfo(f"Calculating {algorithm} values...")

            if algorithm == 'HSV':
                converted = self.rgb_to_hsv(rgb)
                names = ("H(HSV)", "S(HSV)", "V(HSV)")
            elif algorithm == 'YCbCr':
                y = 0.299 * r + 0.587 * g + 0.114 * b
                converted = (y, 0.564 * (b - y), 0.713 * (r - y))
                names = ("Y(YCbCr)", "Cb(YCbCr)", "Cr(YCbCr)")
            elif algorithm == 'YIQ':
                converted = (
                    0.299 * r + 0.587 * g + 0.114 * b,
                    0.596 * r - 0.274 * g - 0.322 * b,
                    0.211 * r - 0.523 * g + 0.312 * b
                )
                names = ("Y(YIQ)", "I(YIQ)", "Q(YIQ)")
            elif algorithm == 'YUV':
                converted = (
                    0.299 * r + 0.587 * g + 0.114 * b,
                    -0.147 * r - 0.289 * g + 0.436 * b,
                    0.615 * r - 0.515 * g - 0.100 * b
                )
                names = ("Y(YUV)", "U(YUV)", "V(YUV)")
            else:
                continue

            advance(f"{algorithm} values calculated")
            for name, values in zip(names, converted):
                progress.setInfo(f"Creating scalar field {name}...")
                last_index = self.add_or_replace_scalar_field(pc, name, values)
                advance(f"Scalar field {name} created")

        if last_index >= 0:
            pc.setCurrentDisplayedScalarField(last_index)
            pc.showSF(True)
    
    def main_frame (self,window):
        name_list = get_point_clouds_name()
        if isinstance(name_list, str):
            name_list = [name_list]
        
        def destroy(self):
            window.destroy()  # Close the window
  
        window.title("Color maps")        
        window.resizable(False, False) # Disable resizing the window        
        window.attributes('-toolwindow', 1) # Remove minimize and maximize buttons (title bar only shows close button)       
        form_frame = tk.Frame(window, padx=10, pady=10)  # Create a frame for the form
        form_frame.pack()
    
        # Control variables
        algorithm1_var = tk.BooleanVar()
        algorithm2_var = tk.BooleanVar()
        algorithm3_var = tk.BooleanVar()
        algorithm4_var = tk.BooleanVar()
    
        # Labels
        label_texts = [
            "Select a point cloud:",
            "HSV (Hue-Saturation-Value)",
            "YCbCr (Luma-Chrominance)",
            "YIQ (Brightness-Chrominance-Quadr-phase chrominance)",
            "YUV (Brightness-Chrominance"
        ]
        row_positions = [0,1,2,3,4]        
        definition_of_labels_type_1 ("form_frame",label_texts,row_positions,form_frame,0)
        
        # Combobox
        combo_point_cloud=ttk.Combobox (form_frame,values=name_list,state="readonly")
        combo_point_cloud.grid(column=1, row=0, sticky="e", pady=2)
        if name_list:
            combo_point_cloud.current(0)
    
        # Checkboxes
        algorithm1_checkbox = tk.Checkbutton(form_frame, variable=algorithm1_var)
        algorithm1_checkbox.grid(row=1, column=1, sticky="e")
        
        algorithm2_checkbox = tk.Checkbutton(form_frame, variable=algorithm2_var)
        algorithm2_checkbox.grid(row=2, column=1, sticky="e")       
        
        algorithm3_checkbox = tk.Checkbutton(form_frame, variable=algorithm3_var)
        algorithm3_checkbox.grid(row=3, column=1, sticky="e")       
        
        algorithm4_checkbox = tk.Checkbutton(form_frame, variable=algorithm4_var)
        algorithm4_checkbox.grid(row=4, column=1, sticky="e")        
        
        # Buttons
        _=definition_run_cancel_buttons_type_1("form_frame",
                                     [lambda:run_algorithm_1(self,name_list,combo_point_cloud.get(),bool(algorithm1_var.get()),bool(algorithm2_var.get()),bool(algorithm3_var.get()),bool(algorithm4_var.get())),lambda:destroy(self)],
                                     5,
                                     form_frame,
                                     1
                                     )
            
        def run_algorithm_1(self,name_list,pc_name,hsv,ycbcr,yiq,yuv):
            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Generating the new color scales...")
            progress.setInfo("Validating the selection...")
            progress.update(0)
            try:
                selected_algorithms = []
                if hsv:
                    selected_algorithms.append("HSV")
                if ycbcr:
                    selected_algorithms.append("YCbCr")
                if yiq:
                    selected_algorithms.append("YIQ")
                if yuv:
                    selected_algorithms.append("YUV")
                if not selected_algorithms:
                    raise ValueError("Select at least one color conversion")

                progress.update(5)
                progress.setInfo("Reading the selected point cloud...")
                pc = check_input(name_list,pc_name)
                progress.update(10)
                progress.setInfo("Reading RGB colors...")
                rgb = self.get_rgb(pc)
                progress.update(15)
                self.color_conversion(selected_algorithms,pc,rgb,progress)
                progress.setInfo("Updating CloudCompare...")
                CC.updateUI()
                progress.update(100)

                print('The color scales have been added to the point cloud')
                window.destroy()
            except Exception as error:
                traceback.print_exc()
                messagebox.showerror(
                    "Color conversion",
                    str(error),
                    parent=window
                )
            finally:
                progress.stop()

    def show_frame(self,window):
        self.main_frame(window)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()
        

#%% START THE GUI        
if __name__ == "__main__":
    try:        
        window = tk.Tk()
        app = GUI_cc()
        app.main_frame(window)
        window.mainloop()    
    except Exception as e:
        print("An error occurred during the computation of the algorithm:", e)
        
        # Optionally, print detailed traceback
        traceback.print_exc()
        window.destroy()
