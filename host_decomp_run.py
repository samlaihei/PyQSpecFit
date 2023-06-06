import hostDecomp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files = pd.read_csv("Run_Files/run_v2.csv")
#  runName,DataFile,LineFile,redshift,N_Fits,ContiWindows,LineWindows,useBalmer
lll = len(files)
print("Decomposing hosts...")

in_dir = "data_raw/"
out_dir = "data/"
fig_dir = "host_decomp_figs/"


def subtract_host_6dfgs(name, host, wave_wif, raw_wif, outdir):
    dt = pd.read_csv(f"data_raw/{name}_6dFGS.csv")
    wave, flx, e_flx = np.array(dt['Wavelength']), np.array(dt['Flux']), np.array(dt['eFlux'])

    ind_o3_wifes = np.argmin(np.abs(wave_wif - 5007)) - 10
    ind_o3_6df = np.argmin(np.abs(wave - 5007)) - 10

    o3_peak_wifes = raw_wif[np.argmax(raw_wif[ind_o3_wifes:ind_o3_wifes + 20]) + ind_o3_wifes]
    o3_peak_6df = flx[np.argmax(flx[ind_o3_6df:ind_o3_6df + 20]) + ind_o3_6df]

    scaled_host = host / o3_peak_wifes * o3_peak_6df

    max_wav = np.nanmin([np.nanmax(wave), np.nanmax(wave_wif)])
    min_wav = np.nanmax([np.nanmin(wave), np.nanmin(wave_wif)])
    wave_ind = np.where(np.logical_and(wave >= min_wav, wave <= max_wav))

    gal_host = np.interp(wave[wave_ind], wave_wif, scaled_host)
    out_df = pd.DataFrame()
    out_df['Wavelength'] = wave[wave_ind]
    out_df['Flux'] = flx[wave_ind] - gal_host
    out_df['eFlux'] = e_flx[wave_ind]
    out_df.to_csv(f"{outdir}/{name}_6dFGS.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wave, flx, label='Data')

    ax.plot(out_df['Wavelength'], out_df['Flux'], label='QSO')

    ax.plot(out_df['Wavelength'], gal_host, label='Host')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Flux Density per Angstrom')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{obj}_6dFGS.pdf")
    plt.close()

    return o3_peak_6df / o3_peak_wifes


with open("decomp_params.csv", 'w') as f:
    f.write('name,z,LogL3000,ebv,M_i,tbb,bbnorm,scal_emline,beslope,fragal,gplind,s0,sa,sb,sc,sd,blur\n')
    for ind, row in files.iterrows():
        file = row['DataFile'].split('/')[1]
        obj = file[:-10]
        if "WiFeS" in file:
            print(f"\rDecomposing object: {obj}-WiFeS --- {ind + 1}/{lll}", end="")
            hd = hostDecomp.HostDecomp(in_dir + file, row['redshift'], outDir=out_dir)
            params = hd.run()

            pdata = pd.read_csv("data_raw/" + file)

            raw_lams, raw_flux = pdata['Wavelength'], pdata['Flux']
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(raw_lams, raw_flux, label='Data')
            de_comped_file = f'{out_dir}/{file}'
            pdata = pd.read_csv(de_comped_file)
            lams, flux = pdata['Wavelength'], pdata['Flux']
            ax.plot(lams, flux, label='QSO')

            mdl = hd.evalQSOGen(params, np.array(lams))
            host = mdl.host_galaxy_flux

            ax.plot(lams, mdl.flux, label='Model')
            ax.plot(lams, host, label='Host')
            ax.set_xlabel('Wavelength ($\\AA$)')
            ax.set_ylabel('Flux Density per Angstrom')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{fig_dir}/{file[:-4]}.pdf")
            plt.close()
            param_str = f"{file[-4]},{','.join([str(p) for p in params])}\n"
            f.write(param_str)

            print(f"\rDecomposing object: {obj}-6dFGS --- {ind + 2}/{lll}", end="")
            scale = subtract_host_6dfgs(obj, host, raw_lams, raw_flux, out_dir)

            param_str = f"{file[-4]},{','.join([str(p) for p in params])},{scale}\n"
            f.write(param_str)

print("\nDone.")
