# Example
import PyQSpecFit
import pandas as pd
import numpy as np



############
# Switches #
############

# Continuum Fitting Windows #
MgII_XQ100_windows = np.array([[2200., 2740.], [2840., 3300.], [3500., 3650.]]) # Default

# Filepath with Lines Definition #
line_path = 'Lines/Lines_MgII.csv'

# Line Fitting Windows #
line_fit_MgII = [[2750., 2850.]]


example = PyQSpecFit.PyQSpecFit()
#example.runFit(line_path, MgII_XQ100_windows, line_fit_MgII, N_fits = 10, syntheticFits=True, useFe=True)
example.evalLineProperties(line_path, 'Line_Params/example.csv', 4.0, useFe=True)













