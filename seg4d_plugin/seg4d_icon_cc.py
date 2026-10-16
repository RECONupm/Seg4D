import pycc

from pathlib import Path

import sys
import tkinter as tk
import traceback

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
        cloudcompare_directory = Path(__file__).resolve().parents[2]
        site_packages_directory = (
            cloudcompare_directory
            / "plugins"
            / "Python"
            / "Lib"
            / "site-packages"
        )
        seg4d_runtime_directory = site_packages_directory / "seg4d_runtime"
        main_app_directory = (
            site_packages_directory
            / "seg4d"
        )
        sys.path.insert(0, str(seg4d_runtime_directory))
        sys.path.insert(0, str(main_app_directory))

        root_gui = None
        try:
            from seg4d import main_GUI

            root_gui = tk.Tk()
            app = main_GUI(master=root_gui)
            app.main_frame_gui(root_gui)
            root_gui.mainloop()
        except Exception:
            traceback.print_exc()
            if root_gui is not None:
                root_gui.destroy()


# Keep the public name expected by seg4d_plugin/__init__.py.
seg4d_plugin = Seg4D
