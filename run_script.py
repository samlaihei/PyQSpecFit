# Example of running PyQSpecFit from special parameter file
import PyQSpecFit
import pandas as pd
import matplotlib.pyplot as plt
import sys

file = sys.argv[1]  # 'Run_Files/run_v2.csv'
r_file = pd.read_csv(file)

example = PyQSpecFit.PyQSpecFit()

# Perform fits #
example.runFile(file)

line_name = ["Hbeta_br", "Hbeta_na", "OIII_left", "OIII_right"]
for _, pdata in r_file.iterrows():
    line_path = pdata['LineFile']
    runName = pdata['runName']
    print(f"Evaluating {runName}")
    dataFile = pdata['DataFile']
    paramFile = 'Line_Params/' + runName + '.csv'
    z = float(pdata['redshift'])

    new_df = pd.DataFrame()
    for cnt in [0, 1, 2, 3]:
        res_props_header, res_props, res_props_err = example.evalLineProperties(line_path, paramFile, z,
                                                                                lineCompInd=cnt)
        tmp_df = pd.DataFrame()
        tmp_df['line'] = [line_name[cnt]]
        for ind, val in enumerate(res_props_header):
            tmp_df[val] = [res_props[ind]]
            tmp_df['e' + val] = [res_props_err[ind]]
        new_df = pd.concat([new_df, tmp_df], ignore_index=True)

    new_df.to_csv(f"Evaluated_Lines/{runName}.csv", index=False)

    try:
        # Create plots #
        plotWindow = example.strToArray(pdata['plotWindow'])[0]
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3, 1], width_ratios=[1]),
                                sharex=True)
        plt.subplots_adjust(wspace=0.30, hspace=0.00)
        example.plotLineFits(axs[0], axs[1], line_path, dataFile, paramFile, z, plotWindow=plotWindow)
        plt.savefig('Fit_Figs/' + runName, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
    except:
        print(f"Could not plot {runName}")
