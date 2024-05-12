# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:10:24 2024

@author: Digi_2
"""
import os
import subprocess
import sys
import traceback

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkPDFViewer import tkPDFViewer_1 as pdf_1
from tkPDFViewer import tkPDFViewer_2 as pdf_2
from tkPDFViewer import tkPDFViewer_3 as pdf_3
import webbrowser
from PIL import Image, ImageTk

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\main_module'
sys.path.insert(0, additional_modules_directory)

from main_gui import definition_of_labels_type_1, definition_of_buttons_type_1

#%% ADDING GUIS

# Segmentation methods
additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\segmentation_methods'
sys.path.insert(0, additional_modules_directory)

from supervised_machine_learning import GUI_mls
from unsupervised_machine_learning import GUI_mlu
from deep_learning_segmentation import GUI_dl
from bim_integration_css import GUI_bicss
from bim_integration_damage import GUI_bid

# Geometric based methods
additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\geometric-based_methods'
sys.path.insert(0, additional_modules_directory)

from geometrical_features import GUI_gf
from analysis_of_arches import GUI_arches
from analysis_of_deflections import GUI_deflection
from analysis_of_inclinations import GUI_inclination
from analysis_of_vaults import GUI_vaults

# Radiometric based methods
additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\\radiometric-based_methods'
sys.path.insert(0, additional_modules_directory)

from color_conversion import GUI_cc
from stadistical_features import GUI_sf

# Other methods
additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\other_methods'
sys.path.insert(0, additional_modules_directory)

from anisotropic_denoising import GUI_ad
from potree_converter import GUI_potree_converter
from voxelize import GUI_voxelize

#%% ADDING assets
current_directory= os.path.dirname(os.path.abspath(__file__))
path_guide_1= os.path.join(current_directory,'assets',r"guide_1.pdf")
path_guide_2= os.path.join(current_directory,'assets',r"guide_2.pdf")
path_guide_3= os.path.join(current_directory,'assets',r"guide_3.pdf")
path_license= os.path.join(current_directory,'assets',r"license.pdf")
path_icon_ico= os.path.join(current_directory,'assets',r"logo.ico")
path_icon_png= os.path.join(current_directory,'assets',r"logo_high_res.png")
path_about_logo_1= os.path.join(current_directory,'assets',r"logo_1.png")
path_about_logo_2= os.path.join(current_directory,'assets',r"logo_2.png")
path_about_logo_3= os.path.join(current_directory,'assets',r"logo_3.png")
path_about_logo_4= os.path.join(current_directory,'assets',r"logo_4.png")

#%% ADDING contributors faces
contributors_directory = (current_directory,'assets', 'contributors')
extensions = ["jpg", "jpeg", "png"]
# Initiate the list of the contributors images
contributor_images = [os.path.join(*contributors_directory, f"i{i}.{ext}") 
                      for i in range(1, 18) 
                      for ext in extensions 
                      if os.path.exists(os.path.join(*contributors_directory, f"i{i}.{ext}"))]

#%% GUI

class main_GUI(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
    
    def toggle_window(self, window_name, window_class):
        window_attr = f'{window_name}_window'
        app_attr = f'{window_name}_app'
        
        if not hasattr(self, window_attr) or not getattr(self, window_attr).winfo_exists():
            setattr(self, window_attr, tk.Toplevel(self.master))
            setattr(self, app_attr, window_class(getattr(self, window_attr)))
            getattr(self, app_attr).main_frame(getattr(self, window_attr))
        else:
            getattr(self, window_attr).destroy()
            
    def toggle_gf(self,root_gui):
        self.toggle_window('gf', GUI_gf)

    def toggle_sf(self,root_gui):
        self.toggle_window('sf', GUI_sf)
    
    def toggle_cc(self,root_gui):
        self.toggle_window('cc', GUI_cc)

    def toggle_mls(self,root_gui):
        self.toggle_window('mls', GUI_mls)

    def toggle_mlu(self,root_gui):
        self.toggle_window('mlu', GUI_mlu)

    def toggle_dl(self,root_gui):
        self.toggle_window('dl', GUI_dl)

    def toggle_arches(self,root_gui):
        self.toggle_window('arches', GUI_arches)

    def toggle_deflection(self,root_gui):
        self.toggle_window('deflection', GUI_deflection)

    def toggle_inclination(self,root_gui):
        self.toggle_window('inclination', GUI_inclination)

    def toggle_vaults(self,root_gui):
        self.toggle_window('vaults', GUI_vaults)

    def toggle_bicss(self,root_gui):
        self.toggle_window('bicss', GUI_bicss)

    def toggle_bid(self,root_gui):
        self.toggle_window('bid', GUI_bid)

    def toggle_anisotropic(self,root_gui):
        self.toggle_window('anisotropic', GUI_ad)

    def toggle_voxelize(self,root_gui):
        self.toggle_window('voxelize', GUI_voxelize)

    def toggle_pc(self,root_gui):
        self.toggle_window('pc', GUI_potree_converter)
    
    def open_guide_1(self, event):
        webbrowser.open(path_guide_1, new=2)
    
    def open_guide_2(self, event):
        webbrowser.open(path_guide_2, new=2)
    
    def open_guide_3(self, event):
        webbrowser.open(path_guide_3, new=2)
        
    def open_license(self, event):
        webbrowser.open(path_license, new=2)
    
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
                
    def main_frame_gui(self, root_gui):
                
        # # GENERAL CONFIGURATION OF THE GUI
        # Configuration of the window
        root_gui.title("SEG4D: Segmentation for CH Diagnosis v.1.0")
        root_gui.iconbitmap(path_icon_ico)
        root_gui.resizable(False, False)     
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root_gui)
        tab_control.pack(expand=1, fill="both")
        
        tab1 = ttk.Frame(tab_control) 
        tab1.pack()
        tab_control.add(tab1, text='Construction Systems Segmentation')
        tab2 = ttk.Frame(tab_control)  
        tab2.pack() 
        tab_control.add(tab2, text='Damage evaluation')
        tab3 = ttk.Frame(tab_control)
        tab3.pack()
        tab_control.add(tab3, text='Other')
        tab4 = ttk.Frame(tab_control)
        tab4.pack()
        tab_control.add(tab4, text='About')
        tab_control.pack(expand=1, fill="both")
        
        # TAB1 = CONSTRUCTION SYSTEMS SEGMENTATION
        
        frames_and_buttons = [
            ("Features Computation", [("Geometrical features", self.toggle_gf), ("Stadistical features", self.toggle_sf)]),
            ("Color Conversion", [("Open", self.toggle_cc)]),
            ("Machine Learning", [("Supervised", self.toggle_mls), ("Unsupervised", self.toggle_mlu)]),
            ("Deep Learning", [("Point Transformer", self.toggle_dl)]),
            ("Synthetic point clouds", [("Timber slabs", self.toggle_deflection), ("Mansory vaults", self.toggle_vaults)]),
            ("BIM Integration", [("Open", self.toggle_bicss)])
        ]

        for row, (frame_text, buttons) in enumerate(frames_and_buttons, start=1):
            frame = tk.LabelFrame(tab1, text=frame_text, padx=5, pady=4)
            frame.grid(row=row, column=0, sticky="nsew", columnspan=2, padx=5, pady=4)
            for col, (button_text, command) in enumerate(buttons):
                button = ttk.Button(frame, text=button_text, command=lambda cmd=command: cmd(root_gui))
                button.grid(row=row, column=col, sticky="nsew", pady=4, padx=5)
                
        # # Disabled button
        bim_integration_button = next(button for frame_text, buttons in frames_and_buttons if frame_text == "BIM Integration" for button_text, button_command in buttons if button_text == "Open")
        bim_integration_button.config(state="disabled")
        
        synthetic_clouds_buttons = []
        for frame_text, buttons in frames_and_buttons:
            if frame_text == "Synthetic point clouds":
                for button_text, command in buttons:
                    frame = next((child for child in tab1.winfo_children() if isinstance(child, tk.LabelFrame) and child.cget("text") == frame_text), None)
                    if frame:
                        button = next((child for child in frame.winfo_children() if isinstance(child, ttk.Button) and child["text"] == button_text), None)
                        if button:
                            synthetic_clouds_buttons.append(button)
        
        # Desactivar los botones encontrados
        for button in synthetic_clouds_buttons:
            button.config(state="disabled")

        # Label with hyperlink
        label_text = "For more information: open the guide in your default PDF visualizer"
        label = tk.Label(tab1, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=11, column=4, sticky="w", pady=1, padx=1)
        label.bind("<Button-1>", lambda event: self.open_guide_1(event))
        
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(tab1, orient="vertical")
        separator.grid(row=0, column=3, rowspan=4, sticky="ns", padx=15, pady=10)
        
        # PDF visualizer in tkinter
        v1 = pdf_1.ShowPdf_1()
        v1.img_object_li.clear()
        v1.pdf_view(tab1, pdf_location=path_guide_1, width=55, height=25).grid(row=0, column=4, rowspan=10, padx=5, pady=5, sticky="nsew")

        
        # TAB2 = DAMAGE EVALUATION
        
        frames_and_buttons = [
            ("Features Computation", [("Geometrical features", self.toggle_gf), ("Stadistical features", self.toggle_sf)]),
            ("Color Conversion", [("Open", self.toggle_cc)]),
            ("Machine Learning", [("Supervised", self.toggle_mls), ("Unsupervised", self.toggle_mlu)]),
            ("Deep Learning", [("Point Transformer", self.toggle_dl)]),
            ("Arches and Vaults", [("Analysis of arches", self.toggle_arches), ("Analysis of vaults", self.toggle_vaults)]),
            ("Slabs", [("Analysis of deflection", self.toggle_deflection)]),
            ("Pilars and Buttresses", [("Analysis of inclination", self.toggle_inclination)]),
            ("BIM Integration", [("Open", self.toggle_bid)])
        ]

        for row, (frame_text, buttons) in enumerate(frames_and_buttons, start=1):
            frame = tk.LabelFrame(tab2, text=frame_text, padx=5, pady=2)
            frame.grid(row=row, column=0, sticky="nsew", columnspan=2, padx=5, pady=2)
            for col, (button_text, command) in enumerate(buttons):
                button = ttk.Button(frame, text=button_text, command=lambda cmd=command: cmd(root_gui))
                button.grid(row=row, column=col, sticky="nsew", pady=2, padx=5)
        
        # Disabled button
        bim_integration_button = next(button for frame_text, buttons in frames_and_buttons if frame_text == "BIM Integration" for button_text, _ in buttons if button_text == "Open")
        bim_integration_button.config(state="disabled")
        
        # Label with hyperlink
        label_text = "For more information: open the guide in your default PDF visualizer"
        label = tk.Label(tab2, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=11, column=4, sticky="w", pady=10)
        label.bind("<Button-1>", lambda event: self.open_guide_2(event))
   
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(tab2, orient="vertical")
        separator.grid(row=0, column=3, rowspan=5, sticky="ns", padx=15 , pady=10)
        
        # PDF visualizer in tkinter
        v2 = pdf_2.ShowPdf_2()
        v2.img_object_li.clear()
        v2.pdf_view(tab2, pdf_location=path_guide_2, width=55, height=25).grid(row=0, column=4, rowspan=10, padx=5, pady=5, sticky="nsew")
        
        
        # TAB3 = OTHER
        
        frames_and_buttons = [
            ("Noise Reduction", [("Open", self.toggle_anisotropic)]),
            ("Point Cloud Voxelization", [("Open", self.toggle_voxelize)]),
            ("Potree Converter", [("Open", self.toggle_pc)])
        ]

        for row, (frame_text, buttons) in enumerate(frames_and_buttons, start=1):
            frame = tk.LabelFrame(tab3, text=frame_text, padx=5, pady=5)
            frame.grid(row=row, column=0, sticky="nsew", columnspan=2, padx=5, pady=5)
            for col, (button_text, command) in enumerate(buttons):
                button = ttk.Button(frame, text=button_text, command=lambda cmd=command: cmd(root_gui))
                button.grid(row=row, column=col, sticky="nsew", pady=2, padx=5)
        
        # Label with hyperlink
        label_text = "For more information: open the guide in your default PDF visualizer"
        label = tk.Label(tab3, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=41, column=4, sticky="w", pady=10)
        label.bind("<Button-1>", lambda event: self.open_guide_3(event))
        
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(tab3, orient="vertical")
        separator.grid(row=0, column=2, rowspan=5, sticky="ns", padx=85 , pady=10)
        
        # PDF visualizer in tkinter
        v3 = pdf_3.ShowPdf_3()
        v3.img_object_li.clear()
        v3.pdf_view(tab3, pdf_location=path_guide_3, width=55, height=25).grid(row=0, column=4, rowspan=10, padx=5, pady=5, sticky="nsew")
        
        
        # TAB4 = ABOUT
        
        # Create a fram inside tab4 to positionate Canvas
        canvas_frame = tk.Frame(tab4)
        canvas_frame.grid(row=0, column=0, sticky="nsew")

        # Create Canvas inside frame
        canvas = tk.Canvas(canvas_frame)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        # Create frame inside canvas to add widgets
        inner_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
        
        # Add logos
        self.load_and_display_image(path_icon_png, 80, 80, inner_frame, column=0, row=0, rowspan=2,columnspan=2,sticky="nsew")
        self.load_and_display_image(path_about_logo_1, 270, 140, inner_frame, column=0, row=4, columnspan=3, rowspan=4, sticky="w", padx=5)
        self.load_and_display_image(path_about_logo_2, 100, 80, inner_frame, column=4, row=0,rowspan=2,sticky="e")
        self.load_and_display_image(path_about_logo_3, 80, 80, inner_frame, column=5, row=0, rowspan=2,sticky="e")
        self.load_and_display_image(path_about_logo_4, 100, 80, inner_frame, column=6, row=0, rowspan=2,sticky="e")
        
        # Labels
        label_title = tk.Label(inner_frame, text="SEG4D", font=("Helvetica", 16, "bold"), padx=5, pady=5)
        label_title.grid(row=0, column=2, sticky="w", padx=1, pady=0)
        
        label_title = tk.Label(inner_frame, text="Segmentation for Cultural Heritage Diagnosis", font=("Helvetica", 10, "bold"), padx=5, pady=5)
        label_title.grid(row=1, column=2, columnspan=3, sticky="w", padx=0, pady=1)
        
        label_copyright= tk.Label(inner_frame, text="Copyright © 2024 Luis Javier Sánchez, Pablo Sanz & Rubén Santamaría", font=("Helvetica", 10, "bold"), padx=5)
        label_copyright.grid(row=2, column=0, columnspan=5, sticky="w", padx=1, pady=1)
        
        # Labels with information about the software and the authors
        label_text = ("This software has been funded by the Comunidad de Madrid through the call Research Grants for Young Investigators from Universidad\n"
                        "Politécnica de Madrid. It has been developed by the AIPA research team."
                      )
        
        label_about = tk.Label(inner_frame, text=label_text, justify="left", padx=5, pady=5)
        label_about.grid(row=3, column=0, columnspan=7, sticky="w", padx=1, pady=1)
        
        label_text = ("For help using the software, consult the user guide.\n"
                        "For technical support, contact us at:\n"
                        "lj.sanchez@upm.es\n"
                        "p.sanzh@upm.es\n"
                        "ruben.santamaria.maestro@upm.es\n\n"
                        "Version: 1.0")
        
        label_about = tk.Label(inner_frame, text=label_text, justify="right", padx=5, pady=5)
        label_about.grid(row=4, column=4, columnspan=3,rowspan=4, sticky="e", padx=5, pady=5)

        # Label with hyperlink
        label_text = "See LICENSE"
        label = tk.Label(inner_frame, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=4, column=3, sticky="w", pady=2)
        label.bind("<Button-1>", lambda event: self.open_license(event))
        
        # Label with hyperlink
        label_text = "Open the user guide"
        label = tk.Label(inner_frame, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=5, column=3, sticky="w", pady=2)
        label.bind("<Button-1>", lambda event: self.open_guide_1(event))
        
        # Label with hyperlink
        label_text = "Visit our website"
        label = tk.Label(inner_frame, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=6, column=3, sticky="w", pady=2)
        label.bind("<Button-1>", lambda event: webbrowser.open("https://www.dcta.upm.es/regional-careen/", new=2))
        
        # Label with hyperlink
        label_text = "Contact technical support"
        label = tk.Label(inner_frame, text=label_text, fg="blue", cursor="hand2")
        label.grid(row=7, column=3, sticky="w", pady=2)
        label.bind("<Button-1>", lambda event: webbrowser.open("lj.sanchez@upm.es", new=2))
        
        # Create a list of label texts
        label_texts = [
            "Luis Javier Sánchez Aparicio (lj.sanchez@upm.es). PhD in Geoinformatics. Universidad Politécnica de Madrid. Department\n"
            "of Construction and Architectural Technologies. ORCID: 0000-0001-6758-2234.",
            "Pablo Sanz Honrado (p.sanzh@upm.es). Pre-doctoral fellow. Universidad Politécnica de Madrid. Department of\n"
            "Construction and Architectural Technologies. ORCID: 0000-0002-8090-1794.",
            "Rubén Santamaría Maestro (ruben.santamaria.maestro@upm.es). Pre-doctoral fellow. Universidad Politécnica de\n"
            "Madrid. Department of Construction and Architectural Technologies. ORCID: 0009-0001-0141-2002.",
            "Paula Villanueva Llauradó (paula.villanueva@upm.es). PhD in Structural Engineering. Universidad Politécnica de Madrid.\n"
            "Department of Structures. ORCID: 0009-0001-0141-2002.",
            "Jose Ramón Aira Zunzunegui (joseramon.aira@upm.es). PhD in Mechanics of Materials. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0002-4598-5259.",
            "Jesús María García Gago (jesusmgg@usal.es). PhD in Cartographic and Terrain Engineering. University of Salamanca.\n"
            "Department of Construction and Agronomy. ORCID: 0000-0001-9100-7600.",
            "Federico Luis del Blanco García (federicoluis.delblanco@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0002-7907-6643.",
            "David Sanz Aráuz (david.sanz.arauz@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0003-4793-169X.",
            "Roberto Pierdicca (r.pierdicca@univpm.it). PhD in Information Engineering. Università Politecnica delle Marche.\n"
            "Department of Civil Building Engineering and Architecture. ORCID: 0000-0002-9160-834X.",
            "Soledad García Morales (soledad.garcia@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0003-1106-1063.",
            "Miguel Ángel Maté González (mategonzalez@usal.es). PhD in Geotechnologies Applied to Construction, Energy and\n"
            "Industry. University of Salamanca. Department of Cartographic and Terrain Engineering. ORCID: 0000-0001-5721-346X.",
            "Javier Pinilla Meló (javier.pinilla@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0002-9390-2083.",
            "Esther Moreno Fernández (esther.moreno@upm.es). PhD in Materials Engineering and Chemical Engineering. Universidad\n"
            "Politécnica de Madrid. Department of Construction and Architectural Technologies. ORCID: 0000-0001-6625-7093.",
            "Cristina Mayo Corrochano (info@estudiomayo.com). PhD in Architecture. ESTUDIO MAYO. Madrid\n",
            "Beatriz del Río Calleja (b.delrio@alumnos.upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0002-9998-8789.",
            "David Mencías Carrizosa (d.mencias@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Structures.",
            "Milagros Palma Crespo (m.palma@upm.es). PhD in Architecture. Universidad Politécnica de Madrid.\n"
            "Department of Construction and Architectural Technologies. ORCID: 0000-0002-6982-2374."
        ]
        
        # Load images and display them dynamically
        for i, (image_path, label_text) in enumerate(zip(contributor_images, label_texts), start=8):
            self.load_and_display_image(image_path, 60, 60, inner_frame, column=0, row=i, sticky="w", padx=10)
            label_about = tk.Label(inner_frame, text=label_text, justify="left", padx=5, pady=5)
            label_about.grid(row=i, column=1, columnspan=6, sticky="nw", padx=1, pady=1)
        
        # Configurate Canvas to expand with the size change
        inner_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        # Add scroll bar
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.config(yscrollcommand=scrollbar.set)
        
        def scroll_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", scroll_wheel)
        
        # Configurate the Canvas behaviour with te window size
        canvas.grid_rowconfigure(0, weight=1)
        canvas.grid_columnconfigure(0, weight=1)
        
        # Configurate the expand of inner_frame
        tab4.grid_rowconfigure(0, weight=1)
        tab4.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
try:
    # START THE MAIN WINDOW        
    root_gui = tk.Tk()
    app = main_GUI()
    app.main_frame_gui(root_gui)
    root_gui.mainloop()    
except Exception as e:
    print("An error occurred during the computation of the algorithm:", e)
    # Optionally, print detailed traceback
    traceback.print_exc()
    root_gui.destroy()

