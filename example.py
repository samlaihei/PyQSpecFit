# Example of running PyQSpecFit from special parameter file
import PyQSpecFit
import pandas as pd
import matplotlib.pyplot as plt

file = 'Run_Files/runFile.csv'
pdata = pd.read_csv(file)

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)


for ind, line_path in enumerate(pdata['LineFile'].to_numpy()):
    runName = pdata['runName'].to_numpy()[ind]
    dataFile = pdata['DataFile'].to_numpy()[ind]
    paramFile = 'Line_Params/'+runName+'.csv'
    z = float(pdata['redshift'].to_numpy()[ind])
    lineCompInd = int(pdata['lineComplexInd'].to_numpy()[ind])
    plotWindow = example.strToArray(pdata['plotWindow'].to_numpy()[ind])[0]
    
    # Evaluate line #
    example.evalLineProperties(line_path, paramFile, z, lineCompInd=lineCompInd)

    # Create plots #
    fig, axs = plt.subplots(2,1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3,1], width_ratios=[1]), sharex=True)
    plt.subplots_adjust(wspace= 0.30, hspace= 0.00)
    example.plotLineFits(axs[0], axs[1], line_path, dataFile, paramFile, z, plotWindow=plotWindow)
    plt.savefig('Fit_Figs/'+runName, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()






