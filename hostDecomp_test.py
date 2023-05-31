# Example
import hostDecomp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = 'data/decomp_test.csv'
test = hostDecomp.HostDecomp(file, 0.1)

params = test.run()

pdata = pd.read_csv(file)
lams, flux, eflux = pdata['Wavelength'], pdata['Flux'], pdata['eFlux']
fig, ax = plt.subplots()
ax.plot(lams, flux, label='Data')

file = 'data/hostDecomp/decomp_test.csv'
pdata = pd.read_csv(file)
lams, flux, eflux = pdata['Wavelength'], pdata['Flux'], pdata['eFlux']
ax.plot(lams, flux, label='QSO')

ax.plot(lams, test.evalQSOGen(params, np.array(lams)).flux, label='Model')
ax.plot(lams, test.evalQSOGen(params, np.array(lams)).host_galaxy_flux, label='Host')
ax.set_xlabel('Wavelength ($\\AA$)')
ax.set_ylabel('Flux Density per Angstrom')
ax.legend()
plt.show()