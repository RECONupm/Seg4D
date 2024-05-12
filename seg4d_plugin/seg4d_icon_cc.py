import pycc

from typing import List
from pathlib import Path

import os
import sys
import tkinter as tk

MAIN_ICON_PATH = str(Path(__file__).parent / "logo.png")


class Seg4D(pycc.PythonPluginInterface):
    def __init__(self):
        pycc.PythonPluginInterface.__init__(self)
        self.count = 0
        self.app = pycc.GetInstance()

    def getIcon(self):
        return MAIN_ICON_PATH

    def getActions(self):
        return [pycc.Action(name="Seg4D", target=self.launch_application, icon=MAIN_ICON_PATH)]
    
    def launch_application(self):
        # Obtener la ruta del directorio del script seg4d.py
        script_directory = os.path.abspath(__file__)
        path_parts = script_directory.split(os.path.sep)
        main_app_directory=os.path.sep.join(path_parts[:-2])+ '\plugins\Python\Lib\site-packages\seg4dinst'
        sys.path.insert(0, main_app_directory)

        # Importar la clase main_GUI desde seg4d.py
        from seg4d import main_GUI

        # Crear una instancia de la clase main_GUI y mostrar la ventana principal
        try:
            root_gui = tk.Tk()
            app = main_GUI()
            app.main_frame_gui(root_gui)
            root_gui.mainloop()
        except Exception as e:
            print("An error occurred during the computation of the algorithm:", e)
            # Optionally, print detailed traceback
            traceback.print_exc()
            root_gui.destroy()
