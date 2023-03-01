# Example
import PyQSpecFit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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
#example.evalLineProperties(line_path, 'Line_Params/example.csv', 0.83, useFe=True)

fig, axs = plt.subplots(2,1, figsize=(10, 8))
example.plotLineFits(axs[0], axs[1], line_path, 'data/example.csv', 'Line_Params/example.csv', 0.83, plotWindow=[2500, 3100], useFe=True)
plt.show()









