# Example
import PyQSpecFit
import pandas as pd
import numpy as np



############
# Switches #
############

# Continuum Fitting Windows #
MgII_XQ100_windows = np.array([[2200., 2740.], [2840., 3300.], [3500., 3650.]]) # Default


# Filepath with Lines #
lines_path = 'Lines/Lines_MgII.csv'
line_fit_MgII = [[2750., 2850.]]



example = PyQSpecFit.PyQSpecFit()
example.runFit(lines_path, MgII_XQ100_windows, line_fit_MgII, useFe=True)














