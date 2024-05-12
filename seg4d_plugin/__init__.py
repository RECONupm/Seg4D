# importing our class in the top level __init__ is
# mandatory, so that after importing the module
# the `seg4d_plugin` class exists in the Python type system
# and the PythonPlugin can import & instantiate it
from .seg4d_icon_cc import seg4d_plugin