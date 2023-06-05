import hostDecomp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = pd.read_csv("Run_Files/run_v2.csv")
#  runName,DataFile,LineFile,redshift,N_Fits,ContiWindows,LineWindows,useBalmer
lll = len(files)
print("Decomposing hosts...")

with open("decomp_params.csv", 'w') as f:
    f.write('name,z,LogL3000,ebv,M_i,tbb,bbnorm,scal_emline,beslope,fragal,gplind,s0,sa,sb,sc,sd,blur')
    for ind, row in files.iterrows():
        file = row['DataFile'].split('/')[1]
        print(f"Decomposing file: {file}  --- {ind + 1}/{lll}", end='\r')
        hd = hostDecomp.HostDecomp("data_raw/" + file, row['redshift'], outDir='host_decomp_v2/')
        params = hd.run()

        pdata = pd.read_csv("data_raw/" + file)

        lams, flux, _ = pdata['Wavelength'], pdata['Flux'], pdata['eFlux']
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lams, flux, label='Data')
        de_comped_file = f'host_decomp_v2/{file}'
        pdata = pd.read_csv(de_comped_file)
        lams, flux, _ = pdata['Wavelength'], pdata['Flux'], pdata['eFlux']
        ax.plot(lams, flux, label='QSO')

        ax.plot(lams, hd.evalQSOGen(params, np.array(lams)).flux, label='Model')
        ax.plot(lams, hd.evalQSOGen(params, np.array(lams)).host_galaxy_flux, label='Host')
        ax.set_xlabel('Wavelength ($\\AA$)')
        ax.set_ylabel('Flux Density per Angstrom')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"host_decomp_figs_v2/{file[:-4]}.pdf")
        plt.close()
        param_str = f"{file[-4]},{','.join([str(p) for p in params])}\n"
        f.write(param_str)
    print("\nDone.")
