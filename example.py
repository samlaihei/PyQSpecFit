# Example
import PyQSpecFit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



############
# Switches #
############

# Continuum Fitting Windows #
conti_windows = np.array([[1973., 1983.], [2060., 2340.], [2600., 2740.], [2840., 3100.]])

# Filepath with Lines Definition #
line_path = 'Lines/Lines_MgII.csv'

# Line Fitting Windows #
line_fit_MgII = [[2750., 2850.]]


example = PyQSpecFit.PyQSpecFit()
#example.runFit(line_path, conti_windows, line_fit_MgII, N_fits = 10, syntheticFits=True, useFe=True)
#example.evalLineProperties(line_path, 'Line_Params/example.csv', 0.83, useFe=True)


fig, axs = plt.subplots(2,1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3,1], width_ratios=[1]))
plt.subplots_adjust(wspace= 0.30, hspace= 0.00)
example.plotLineFits(axs[0], axs[1], line_path, 'data/example.csv', 'Line_Params/example.csv', 0.83, plotWindow=[2500, 3100], useFe=True)
plt.show()









