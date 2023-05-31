# hostDecomp: Partner for PyQSpecFit to decompose host component using qsogen
# Author: Samuel Lai, ANU
# Email: samuel.lai (AT) anu (DOT) edu (DOT) au

# Version 1.0
# 2023-03-31

###################
# Version History #
###################
# V1.0 - Initial file
#     
#
#

#########
# To Do #
#########
# TODO: ?


###################
# Import Packages #
###################
from qsosed import Quasar_sed
import pandas as pd
from lmfit import minimize, Parameters, report_fit
import numpy as np



class HostDecomp():
    def __init__(self, dataFile, z, outDir='data/hostDecomp/'):
        self.dataFile = dataFile
        self.z = z
        self.outDir = outDir
        

    def run(self):
        filename = self.dataFile.split('/')[-1]
        pdata = pd.read_csv(self.dataFile)
        lams = pdata['Wavelength'].to_numpy()
        flux = pdata['Flux'].to_numpy()
        eflux = pdata['eFlux'].to_numpy()
        ##############
        # Fit QSOGen #
        ##############
        """
        QSOGen components described by 10 parameters
         pp[0]: redshift
         pp[1]: Monochromatic luminosity at 3000A of (unreddened) quasar model
         pp[2]: Extinction E(B-V) applied to quasar model. Not applied to galaxy component. Default is zero.
         pp[3]: Absolute i-band magnitude (at z=2), as reported in SDSS DR16Q, used to control scaling of emission-line and host-galaxy contributions.
         pp[4]: Temperature of hot dust blackbody in Kelvin.
         pp[5]: Normalisation, relative to power-law continuum at 2 micron, of the hot dust blackbody.
         pp[6]: Overall scaling of emission line template. Negative values preserve relative equivalent widths while positive values preserve relative line fluxes. Default is -1.
         pp[7]: Baldwin effect slope, which controls the relationship between `emline_type` and luminosity `M_i`.
         pp[8]: Fractional contribution of the host galaxy to the rest-frame 4000-5000A region of the total SED, for a quasar with M_i = -23.
         pp[9]: Power-law index dependence of galaxy luminosity on M_i.
        """
        
        tmp_parinfo = [{'name':'z', 'limits': (0, 1E3), 'init_value': self.z, 'fixed':True}, 
                       {'name':'LogL3000', 'limits': (40, 50), 'init_value': 45., 'fixed':False}, 
                       {'name':'ebv', 'limits': (0., 5.), 'init_value': 0.01, 'fixed':False}, 
                       {'name':'M_i', 'limits': (-21, -29), 'init_value': -27, 'fixed':False}, 
                       {'name':'tbb', 'limits': (100, 1E4), 'init_value': 3000., 'fixed':False}, 
                       {'name':'bbnorm', 'limits': (0, 1E3), 'init_value': 1., 'fixed':False},
                       {'name':'scal_emline', 'limits': (-5., 5), 'init_value': -1., 'fixed':False},
                       {'name':'beslope', 'limits': (0, 1), 'init_value': 0.2, 'fixed':False}, 
                       {'name':'fragal', 'limits': (0, 1), 'init_value': 0.244, 'fixed':False},
                       {'name':'gplind', 'limits': (0., 1.), 'init_value': 0.684, 'fixed':False}]
        params = Parameters()
        for pars in tmp_parinfo:
            params.add(pars['name'], value=pars['init_value'], min=pars['limits'][0], max=pars['limits'][1], vary=not pars['fixed'])


        result = minimize(self.QSOGen_resid, params, args=[(lams, flux, eflux)])
        fitted_params = result.params
        fitted_values = fitted_params.valuesdict()
        fitted_array = np.array(list(fitted_values.values()))
    
        self.QSOgen_bestfit = fitted_array
        host_comp = self.evalQSOGen(self.QSOgen_bestfit, lams).host_galaxy_flux
        pdata = pd.DataFrame()
        pdata['Wavelength'] = lams
        pdata['Flux'] = flux - host_comp
        pdata['eFlux'] = eflux
        pdata.to_csv(self.outDir+filename, index=False)
        
        return self.QSOgen_bestfit
            

    def evalQSOGen(self, pp, xx):
        model = Quasar_sed(z=pp[0], LogL3000=pp[1], wavlen=xx, ebv=pp[2], M_i=pp[3], tbb=pp[4], 
                           bbnorm=pp[5], scal_emline=pp[6], beslope=pp[7], fragal=pp[8], gplind=pp[9])
        return model

    def QSOGen_resid(self, p, data):
        xx, yy, e_yy = data
        pp = np.array(list(p.valuesdict().values()))
        model = self.evalQSOGen(pp, xx)
        return (yy-model.flux)/e_yy