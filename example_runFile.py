# Example of running PyQSpecFit from special parameter file
import PyQSpecFit
import pandas as pd
import matplotlib.pyplot as plt

file = 'Run_Files/example.csv'

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)

# Evaluate line #
line_path = pd.read_csv(file)['LineFile'].to_numpy()[0]
example.evalLineProperties(line_path, 'Line_Params/example.csv', 0.83, lineCompInd=0)

# Create plots #
fig, axs = plt.subplots(2,1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3,1], width_ratios=[1]), sharex=True)
plt.subplots_adjust(wspace= 0.30, hspace= 0.00)
example.plotLineFits(axs[0], axs[1], line_path, 'data/example.csv', 'Line_Params/example.csv', 0.83, plotWindow=[2500, 3100])
plt.savefig('Fit_Figs/example.png', dpi=200, bbox_inches='tight')







