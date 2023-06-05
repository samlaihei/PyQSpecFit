# PyQSpecFit: Code for modelling emission lines in QSO spectra
# Author: Samuel Lai, ANU
# Email: samuel.lai (AT) anu (DOT) edu (DOT) au

# Version 1.0
# 2023-02-28

###################
# Version History #
###################
# V1.0 - Forked from PyQSpecFit_Functional_v2, basic functionality established
#     
#
#

#########
# To Do #
#########
# TODO: Cleanup
# TODO: Choice of using skewed Gaussians
# TODO: Custom cosmology
# TODO: Save used template to file, replace template input
# TODO: Observed frame input with redshift
# TODO: Units interpreting and handling
# TODO: Tie line complexes together
# TODO: Different line models: PL, Gauss Hermite, Lorentzian
# TODO: Write tests
#


###############################################
#  ___       ___  ___              ___ _ _    #
# | _ \_  _ / _ \/ __|_ __  ___ __| __(_) |_  #
# |  _/ || | (_) \__ \ '_ \/ -_) _| _|| |  _| #
# |_|  \_, |\__\_\___/ .__/\___\__|_| |_|\__| #
#      |__/          |_|                      #
###############################################


###################
# Import Packages #
###################
import glob
import os

# Numpy and Pandas and I/O
import numpy as np
import pandas as pd
import numpy.random as rand
import csv

# Astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from lmfit import minimize, Parameters, report_fit
from astropy.modeling.physical_models import BlackBody
from astropy.stats import sigma_clip

# Scipy
from scipy.stats import norm, skewnorm
from scipy import interpolate
from scipy.integrate import simps
import scipy.constants as con


# Specutils
from specutils import Spectrum1D
from specutils.analysis import line_flux, equivalent_width
from specutils.analysis import fwhm as specutils_fwhm
from specutils.manipulation import gaussian_smooth



# Uncertainty
from uncertainties import ufloat
from uncertainties.umath import *

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator

# QSOGen
from qsosed import Quasar_sed


class PyQSpecFit():
    def __init__(self, dataDir='data/'):
        """
        Get input data

        Parameters:
        -----------
        dataDir: string
            Points to location of csv datafiles setup with columns ['Wavelength', 'Flux', 'eFlux']
            Line models will follow units of the Flux column, but ideally it should be observed erg/s/cm2/Angstrom
            Wavelength should be in rest-frame.
        """
        
        self.dataDir = dataDir
        
        template_path = 'Fe_Templates/'
        VW01 = np.genfromtxt(template_path+'fe_uv_VW01.txt')
        VW01[:,0], VW01[:,1] = 10**VW01[:,0], VW01[:,1]*10**15
        T06 = np.genfromtxt(template_path+'Tsuzuki+2006.uvtemp.tab.txt')
        T06[:,0], T06[:,1] = T06[:,0], T06[:,1]*10**15
        M16 = np.genfromtxt(template_path+'feII_UV_Mejia-Restrepo_et_al_2016_2200-3646AA.data.txt')
        M16[:,0], M16[:,1] = M16[:,0], M16[:,1]*10
        BV08 = np.genfromtxt(template_path+'BruhweilerVerner2008/d11-m20-21-735_uv.txt')
        BV08[:,0], BV08[:,1] = BV08[:,0], BV08[:,1]*10**(-6)
        self.fe_uvs = [VW01, T06, M16, BV08]
        self.init_fe_uv_fwhms = [900., 900., 900., 800.]

        BG92 = np.genfromtxt(template_path + 'fe_optical_BG92.txt')
        BG92[:,0], BG92[:,1] = 10**BG92[:,0], BG92[:,1]*10**15
        T06_opt = np.genfromtxt(template_path + 'Tsuzuki+2006.opttemp.tab.txt')
        T06_opt[:,0], T06_opt[:,1] = T06_opt[:,0], T06_opt[:,1]*10**15
        P22 = np.genfromtxt(template_path + 'Park+2022.Mrk493_OpticalFeTemplate.txt')
        P22[:,0], P22[:,1] = P22[:,0], P22[:,1]*10.
        BV08_opt = np.genfromtxt(template_path+'BruhweilerVerner2008/d11-m20-21-735_opt.txt')
        BV08_opt[:,0], BV08_opt[:,1] = BV08_opt[:,0], BV08_opt[:,1]*10**(-6)
        self.fe_ops = [BG92, T06_opt, P22, BV08_opt]
        self.init_fe_op_fwhms = [1200., 1200., 800., 600.]
    
    def runFit(self, lineFile, contiWindow, lineWindow, N_fits=1,
               normSwitch=True, dataOut=True, syntheticFits=False, globalLineShift=0,
               smoothingSigma=0, clipSigma=0, clipBoxWidth=50, clipBufferWidth=3,
               useBalmer=False, useFe=False, Fe_uv_ind=0, Fe_opt_ind=0,
               runName=None, dataOutPath='Line_Params/'):
            
            """
            Run main fitting routine.

            Parameters:
            -----------
            lineFile: string
                Points to location of the file which determines the lines to be fit and their parameters.
                
            contiWindow: 1D array
                Set of rest-frame windows in Angstrom which should be used to fit the continuum model.
                
            lineWindow: 1D array
                Set of windows in Angstrom used to fit the emission-lines named in the lineFile.
                
            N_fits: integer
                Number of fits to perform, should generally be accompanied by syntheticFits=True
            
            normSwitch: bool
                Whether to normalise the spectrum. Should almost always be on.
                
            dataOut: bool
                Whether to output the data.
            
            syntheticFits: bool
                Toggle resampling of the spectrum based on error spectrum.
            
            globalLineShift: float
                Shift all lines in lineFile by some value in Angstroms. Useful if redshift is slightly incorrect.
                It's recommended to adjust the redshift prior to fitting, but this is used if you want to fit anyway.
                
            smoothingSigma: float
                Smooth spectrum with stddev
            
            clipSigma, clipBoxWidth, clipBufferWidth: floats
                Parameters used for sigma-clipping
            
            useBalmer, useFe: bool
                Toggles for using the Balmer and FeII continuum models
                
            Fe_uv_ind, Fe_opt_ind: integer
                Choice for the UV and optical FeII model, defined up to 3
                
            dataOutPath: string
                Determines where the output data file will be deposited.
                
            """
            self.lineFile = lineFile
            
            self.N_fits = N_fits
            self.syntheticFits = syntheticFits
            self.contiWindow = contiWindow
            self.lineWindow = lineWindow
            self.normSwitch = normSwitch
            
            self.useBalmer = useBalmer
            self.useFe = useFe
            self.Fe_uv_ind = Fe_uv_ind
            self.Fe_opt_ind = Fe_opt_ind
            
            self.globalLineShift = globalLineShift
            self.smoothingSigma = smoothingSigma
            self.clipSigma = clipSigma
            self.clipBoxWidth = clipBoxWidth
            self.clipBufferWidth = clipBufferWidth
            
            self.dataOut = dataOut
            self.dataOutPath = dataOutPath
            
            datafiles = glob.glob(self.dataDir+'*.csv')
            for datafile in datafiles:
                self.Fit(datafile, runName)
                
            return


    def runFile(self, file, 
                normSwitch=True, syntheticFits=False, 
                dataOut=True, dataOutPath='Line_Params/'):
        self.normSwitch = normSwitch
        self.syntheticFits = syntheticFits
        self.dataOut = dataOut
        self.dataOutPath = dataOutPath
        
        pdata = pd.read_csv(file)
        for ind, datafile in enumerate(pdata['DataFile'].to_numpy()):
            runName = pdata['runName'].to_numpy()[ind]
            
            self.lineFile = pdata['LineFile'].to_numpy()[ind]
            
            self.N_fits = int(pdata['N_Fits'].to_numpy()[ind])

            self.contiWindow = self.strToArray(pdata['ContiWindows'].to_numpy()[ind])
            self.lineWindow = self.strToArray(pdata['LineWindows'].to_numpy()[ind])
            
            self.useBalmer = int(pdata['useBalmer'].to_numpy()[ind])
            self.useFe = int(pdata['useFe'].to_numpy()[ind])
            self.Fe_uv_ind = int(pdata['Fe_uv'].to_numpy()[ind])
            self.Fe_opt_ind = int(pdata['Fe_opt'].to_numpy()[ind])
            
            self.globalLineShift = float(pdata['lineShift'].to_numpy()[ind])
            self.smoothingSigma = float(pdata['smoothStddev'].to_numpy()[ind])
            self.clipSigma = float(pdata['clipStddev'].to_numpy()[ind])
            self.clipBoxWidth = float(pdata['clipBoxWidth'].to_numpy()[ind])
            self.clipBufferWidth = int(pdata['clipBufferWidth'].to_numpy()[ind])
            
            self.Fit(datafile, runName)
            
        
        return
        
        
    def Fit(self, dataFile, runName=None):
        self.runName = runName
        for i in range(self.N_fits):
            if self.runName == None:
                self.runName = dataFile.split('/')[-1][:-4]
            print(self.runName)

            pdata = pd.read_csv(dataFile)
            lams = pdata['Wavelength'].to_numpy() # in angstrom
            flux = pdata['Flux'].to_numpy()
            eflux = pdata['eFlux'].to_numpy()

            temp_lams, temp_flux, temp_eflux = [], [], []
            for index, f in enumerate(flux):
                if not np.isnan(f):
                    temp_lams.append(lams[index])
                    temp_flux.append(f)
                    temp_eflux.append(eflux[index])

            lams = np.array(temp_lams)
            flux = np.array(temp_flux)
            eflux = np.array(temp_eflux)

            if self.syntheticFits or self.N_fits > 1:
                lams, flux, eflux = self.create_synthetic(lams, flux, eflux)

            if self.normSwitch: # almost always keep this toggled
                self.norm_median = np.nanmedian(flux)
                flux = flux/self.norm_median
                eflux = eflux/self.norm_median
                
            if self.smoothingSigma > 0:
                spec1 = Spectrum1D(spectral_axis=lams*u.angstrom, flux=flux*u.Jy)
                spec1 = gaussian_smooth(spec1, stddev=self.smoothingSigma)
                lams = spec1.spectral_axis
                lams = np.array([i.value for i in lams])
                flux = spec1.flux
                flux = np.array([i.value for i in flux])


            
            #################
            # Fit Continuum #
            #################
            qso_resid = flux
            """
            Continuum components described by 8 parameters
             pp[0]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
             pp[1]: slope for the power-law continuum
             pp[2]: norm_factor for the MgII Fe_template
             pp[3]: FWHM for the MgII Fe_template
             pp[4]: small shift of wavelength for the MgII Fe template
             pp[5:8]: same as pp[2:4] but for the Hbeta/Halpha Fe template
             pp[8:11]: norm, Te and Tau_e for the Balmer continuum at <3646 A
            """
        
            tmp_parinfo = [{'name':'PL_Norm', 'limits': (0, 1E3), 'init_value': 1., 'fixed':False}, 
                           {'name':'PL_Slope', 'limits': (-5, 3), 'init_value': 0., 'fixed':False}, 
                           {'name':'Fe_UV_Norm', 'limits': (0., 1E3), 'init_value': 0., 'fixed':True}, 
                           {'name':'Fe_UV_FWHM', 'limits': (1E3, 1E4), 'init_value': 3000., 'fixed':True}, 
                           {'name':'Fe_UV_del', 'limits': (-0.02, 0.02), 'init_value': 0., 'fixed':True},
                           {'name':'Fe_Opt_Norm', 'limits': (0., 1E3), 'init_value': 0., 'fixed':True}, 
                           {'name':'Fe_Opt_FWHM', 'limits': (1E3, 1E4), 'init_value': 3000., 'fixed':True}, 
                           {'name':'Fe_Opt_del', 'limits': (-0.02, 0.02), 'init_value': 0., 'fixed':True},
                           {'name':'Balmer_norm', 'limits': (0., 10.), 'init_value': 0., 'fixed':True}, 
                           {'name':'Balmer_Te', 'limits': (1E3, 5E4), 'init_value': 10000., 'fixed':True}, 
                           {'name':'Balmer_tau', 'limits': (0.1, 4.), 'init_value': 0.2, 'fixed':True}]
                           
            self.conti_header = np.array([i['name'] for i in tmp_parinfo])

            params = Parameters()
            for pars in tmp_parinfo:
                params.add(pars['name'], value=pars['init_value'], min=pars['limits'][0], max=pars['limits'][1], vary=not pars['fixed'])
                
            if self.useFe:
                params['Fe_UV_Norm'].value = 1E-4
                params['Fe_Opt_Norm'].value = 1E-4
                Fe_params_list = ['Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del']
                for i in Fe_params_list:
                    params[i].vary=True
            if self.useBalmer:
                params['Balmer_norm'].value = 1E-4
                Balmer_params_list = ['Balmer_norm', 'Balmer_Te', 'Balmer_tau']
                for i in Balmer_params_list:
                    params[i].vary=True

            #params.pretty_print()
            
            print('Fitting Continuum...')
            print()
            result = minimize(self.conti_residuals, params, args=[(lams, qso_resid, eflux, self.contiWindow)], calc_covar=False, xtol=1E-8, ftol=1E-8, gtol=1E-8)
            fitted_params = result.params
            fitted_values = fitted_params.valuesdict()
            fitted_array = np.array(list(fitted_values.values()))
            #report_fit(result)
            
            print("Best-fit parameters:  ", fitted_array)
            print("Function Evaluations: ", result.nfev)
            print("Reduced Chi^2:        ", result.redchi)
            print()

            self.conti_bestfit = fitted_array



            ################
            # Line Fitting #
            ################
            residual_flux = flux - self.eval_conti_all(self.conti_bestfit, lams)
            print('Fitting Lines...')
            print()

            pdata_lines = pd.read_csv(self.lineFile)
            line_names = pdata_lines['Name']
            line_wave = pdata_lines['Central Wavelength']
            line_norms = pdata_lines['Init_Norm']
            line_shifts = pdata_lines['Allowed_Shift']

            sigma_lows = pdata_lines['Sigma_low']
            sigma_highs = pdata_lines['Sigma_high']

            # SkewNorm Fitting
            init_parinfo = []
            self.line_header = ['Skew', 'Sigma', 'Norm', 'Wavelength']

            for index, (lname, wave, norm, shift, slow, shigh) in enumerate(zip(line_names,line_wave,line_norms,line_shifts,sigma_lows,sigma_highs)):
                # Line Parameters: Skew, Scale, Norm, Central Wavelength
                line_parinfo = [{'name':'Skew_'+str(index), 'limits': (-10., 10.), 'init_value': 0., 'fixed': True}, 
                                {'name':'Sigma_'+str(index), 'limits': (slow, shigh), 'init_value': (slow+shigh)/2.+index*1E-6, 'fixed':False}, 
                                {'name':'Norm_'+str(index), 'limits': (0, 1E2), 'init_value': norm+index*1E-6, 'fixed':False}, 
                                {'name':'Wavelength_'+str(index), 'limits': (wave+self.globalLineShift-shift, wave+self.globalLineShift+shift), 'init_value':wave+index*1E-6, 'fixed':False}]
                init_parinfo += line_parinfo

            params = Parameters()
            for pars in init_parinfo:
                params.add(pars['name'], value=pars['init_value'], min=pars['limits'][0], max=pars['limits'][1], vary=not pars['fixed'])
                
            #params.pretty_print()
            xx, yy, e_yy = lams, residual_flux, eflux

            if self.clipSigma > 0:
                mask = self.create_mask_window(xx, self.lineWindow)
                xx, yy, e_yy = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
                xx, yy, e_yy = self.sigma_mask_buffer(self.clipBoxWidth, self.clipSigma, xx, yy, e_yy, self.clipBufferWidth)

            result = minimize(self.residual_line, params, args=[(xx, yy, e_yy, self.lineWindow)], calc_covar=False, xtol=1E-8, ftol=1E-8, gtol=1E-8)
            fitted_params = result.params
            fitted_values = fitted_params.valuesdict()
            fitted_array = np.array(list(fitted_values.values()))
            #report_fit(result)
            
            print("Best-fit parameters:  ", fitted_array)
            print("Function Evaluations: ", result.nfev)
            print("Reduced Chi^2:        ", result.redchi)
            print()

            line_bestfit = fitted_array

            self.out_line_res(lams, line_bestfit, line_names)

    def evalFile(self, file):
        pdata = pd.read_csv(file)
        for ind, line_path in enumerate(pdata['LineFile'].to_numpy()):
            runName = pdata['runName'].to_numpy()[ind]
            paramFile = 'Line_Params/'+runName+'.csv'
            z = float(pdata['redshift'].to_numpy()[ind])
            lineCompInd = int(pdata['lineComplexInd'].to_numpy()[ind])
            
            self.evalLineProperties(line_path, paramFile, z, lineCompInd=lineCompInd)


    def evalLineProperties(self, lineFile, fitFile, redshift, monoLumAngstrom=3000.,
                           lamWindow=[1200, 8000], lineCompInd=0,
                           Fe_uv_ind=0, Fe_opt_ind=0, Fe_cycle=False, sigClipProp=False,
                           outDir='Line_Properties/'):
                           
        """
        Run main line property evaluation routine.

        Parameters:
        -----------
        lineFile: string
            Points to location of the file which determines the lines to be fit and their parameters.
        
        fitFile: string
            Points to location of file containing the line model data
            
        redshift: float
            Redshift of target
            
        monoLumAngstrom: string
            Rest-frame location in Angstrom for where to calculate the monochromatic luminosity.
            
        lamWindow: 1D array, size 2
            Window in Angstrom that should encompass all of the lines of interest
            
        lineCompInd: integer
            Points to the line complex whose properties are to be measured. Follows lineFile

        Fe_uv_ind, Fe_opt_ind: integer
            Choice for the UV and optical FeII model, defined up to 3
            
        Fe_cycle: bool
            Cycles through the available FeII models, depends on datafile following FeII template order
            
        sigClipProp: bool
            Switch to sigma clip line properties, useful if there are significant outliers in line properties
            
        outDir: string
            Determines where the output data file will be deposited.
            
        """

        # Assumes flux is in units of erg/s/cm2/Angstrom #
        self.useBalmer = True
        self.useFe = True
        self.Fe_uv_ind = Fe_uv_ind
        self.Fe_opt_ind = Fe_opt_ind
        outName = fitFile.split('/')[-1][:-4]
        res_props_header = ['FWHM', 'Sigma', 'Blueshift', 'EW', 'pWave', 'iLum', 'Mono_Lum']

        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        dl = self.cosmo.luminosity_distance(redshift).to(u.cm)
        
        N_gauss, vac_wav, line_index = self.readLineFile(lineFile, lineCompInd)

        temp_res_props = [[] for i in res_props_header]
        temp_res_props_err = [[] for i in res_props_header]
        lams = np.linspace(lamWindow[0], lamWindow[1], int(np.ptp(lamWindow)*5.))
    
        pdata = pd.read_csv(fitFile)
        
        rescale = pdata['Norm_Factor'].to_numpy().reshape(int(len(pdata['Norm_Factor'].to_numpy())/N_gauss), N_gauss)
        conti_header = ['PL_Norm', 'PL_Slope', 'Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 
                        'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del', 'Balmer_norm', 'Balmer_Te', 'Balmer_tau']
        contips = np.transpose([pdata[i].to_numpy().reshape(int(len(pdata[i].to_numpy())/N_gauss), N_gauss) for i in conti_header])[0]

        line_norms = pdata['Norm'].to_numpy().reshape(int(len(pdata['Norm'].to_numpy())/N_gauss), N_gauss)
        line_wavs = pdata['Wavelength'].to_numpy().reshape(int(len(pdata['Wavelength'].to_numpy())/N_gauss), N_gauss)
        line_FWHM = pdata['Sigma'].to_numpy().reshape(int(len(pdata['Sigma'].to_numpy())/N_gauss), N_gauss)
        
        for index, (norms, wavs, fwhms, contip, rescale_facs) in enumerate(zip(line_norms, line_wavs, line_FWHM, contips, rescale)):
            if Fe_cycle:
                self.Fe_uv_ind = index%4
                self.Fe_opt_ind = index%4
                
            line_profiles = []
            conti_profile = self.eval_conti_all(contip, lams)*rescale_facs[0]
            PL_profile = self.eval_PL(contip, lams)*rescale_facs[0]
            
            # Assumes flux is in units of erg/s/cm2/Angstrom #
            lum = self.eval_PL(contip, [monoLumAngstrom])*rescale_facs[0]*4*np.pi * dl.value**2 * monoLumAngstrom * (1+redshift)
        
            norms = norms[line_index]
            wavs = wavs[line_index]
            fwhms = fwhms[line_index]
            for N_gauss, (norm, wav, fwhm, rescale_fac) in enumerate(zip(norms, wavs, fwhms, rescale_facs)):
                line_profiles.append(self.eval_line_full([0, fwhm, norm, wav], lams)*rescale_fac)
            tot_line = np.array([0 for i in lams])
            for profile in line_profiles:
                tot_line = self.sum_nan_arrays(tot_line, profile) 

            pWave = lams[tot_line == np.nanmax(tot_line)][0]
            temp_res_props[0].append(self.calc_fwhm(lams, tot_line, pWave)) # FWHM
            temp_res_props[1].append(np.sqrt(self.second_moment(lams, tot_line))/pWave*con.c/1000.) # Sigma
            temp_res_props[2].append((vac_wav-self.general_med_wav(lams, tot_line))/vac_wav*con.c/1000.) # Blueshift
            temp_res_props[3].append(self.calc_ew(lams, tot_line, PL_profile)) # EW
            temp_res_props[4].append(pWave) # Peak Wavelength
            #temp_res_props[4].append(np.log10((conti_profile+tot_line)[tot_line == np.nanmax(tot_line)][0]*4*np.pi*dl.value**2*(1+redshift))) # Peak Luminosity
            temp_res_props[5].append(np.log10(self.calc_line_flux(lams, tot_line)*4*np.pi*dl.value**2*(1+redshift))) # Integrated Luminosity
            temp_res_props[6].append(np.log10(lum))
    
        
        if sigClipProp:
        # sigma_clipping is recommended if there are weird anomalous measurements, replace the above with below
            res_props = [np.nanmean(sigma_clip(i, sigma=3, maxiters=5)) for i in temp_res_props]
            res_props_err = [np.std(sigma_clip(i, sigma=3, maxiters=5)) for i in temp_res_props]
        else:
            res_props = [np.nanmean(i) for i in temp_res_props]
            res_props_err = [np.std(i) for i in temp_res_props]

        pdata = pd.DataFrame()
        for ind, val in enumerate(res_props_header):
            pdata[val] = [res_props[ind]]
            pdata['e'+val] = [res_props_err[ind]]
        pdata.to_csv(outDir+outName+'.csv', index=False)
    
        return [res_props_header, res_props, res_props_err]
        
    def plotFile(self, file):
        pdata = pd.read_csv(file)
        for ind, line_path in enumerate(pdata['LineFile'].to_numpy()):
            runName = pdata['runName'].to_numpy()[ind]
            dataFile = pdata['DataFile'].to_numpy()[ind]
            paramFile = 'Line_Params/'+runName+'.csv'
            z = float(pdata['redshift'].to_numpy()[ind])
            plotWindow = self.strToArray(pdata['plotWindow'].to_numpy()[ind])[0]

            # Create plots #
            fig, axs = plt.subplots(2,1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3,1], width_ratios=[1]), sharex=True)
            plt.subplots_adjust(wspace= 0.30, hspace= 0.00)
            self.plotLineFits(axs[0], axs[1], line_path, dataFile, paramFile, z, plotWindow=plotWindow)
            plt.savefig('Fit_Figs/'+runName, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
            plt.clf()
            plt.close()

    
    def plotLineFits(self, data_ax, resid_ax, lineFile, dataFile, fitFile, redshift,
                     plotWindow=[1200, 8000], dataInd=0, lineCompInd=-1,
                     Fe_uv_ind=0, Fe_opt_ind=0, vspanRanges=[]):
                     
        """
        Run main plotting routine.

        Parameters:
        -----------
        data_ax: matplotlib axes object
            Provides axes to plot the data
            
        resid_ax: matplotlib axes object
            Provides axes to plot the residual
            
        lineFile: string
            Points to location of the file which determines the lines to be fit and their parameters.
        
        dataFile: string
            Points to location of file containing the spectral data
        
        fitFile: string
            Points to location of file containing the line model data    
        
        redshift: float
            Redshift of target
            
        plotWindow: 1D array, size 2
            Window in Angstrom that determines wavelength range of plot
            
        dataInd: integer
            Used to determine which run in fitFile to plot
            
        lineCompInd: integer
            Points to the line complex to be plotted in red, where all others are green. Follows lineFile

        Fe_uv_ind, Fe_opt_ind: integer
            Choice for the UV and optical FeII model, defined up to 3
            
        outDir: string
            Determines where the output data file will be deposited.
            
        """
        
        self.useBalmer = True
        self.useFe = True
        self.Fe_uv_ind = Fe_uv_ind
        self.Fe_opt_ind = Fe_opt_ind
        
        print("Plotting...")
        
        #plt.rcParams.update({
        #    "font.family": "sans-serif",
        #    "font.sans-serif": ["Helvetica"]})
        #plt.rcParams['font.size'] = 16
        
        
        atm_file = 'atm_file/14k_R2k_1_5_micron.txt'
        atm_data = np.genfromtxt(atm_file)
        atm_lams, atm_trans = atm_data[:,0]*10**4/(1+redshift), atm_data[:,1]
        
        norm_fac = 1E17
        pdata = pd.read_csv(dataFile)
        lams = pdata['Wavelength'].to_numpy()
        data_flux = pdata['Flux'].to_numpy()*norm_fac
        data_eflux = pdata['eFlux'].to_numpy()*norm_fac
    
        pdata = pd.read_csv(fitFile)
        atm_lams_plot = atm_lams[np.logical_and(atm_lams>plotWindow[0], atm_lams<plotWindow[1])]
        atm_trans_plot = atm_trans[np.logical_and(atm_lams>plotWindow[0], atm_lams<plotWindow[1])]
        atm_lams_plot = atm_lams_plot[np.where(atm_trans_plot < 0.38)]
        atm_trans_plot = atm_trans_plot[np.where(atm_trans_plot < 0.38)]
    
        line_list_pdata = pd.read_csv(lineFile)
        N_gauss, vac_wav, line_indices = self.readLineFile(lineFile, lineCompInd)
    
        rescale = pdata['Norm_Factor'].to_numpy().reshape(int(len(pdata['Norm_Factor'].to_numpy())/N_gauss), N_gauss)[dataInd]*norm_fac
        conti_header = ['PL_Norm', 'PL_Slope', 'Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 
                        'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del', 'Balmer_norm', 'Balmer_Te', 'Balmer_tau']
        contips = np.transpose([pdata[i].to_numpy().reshape(int(len(pdata[i].to_numpy())/N_gauss), N_gauss) for i in conti_header])[0][dataInd]
        line_norms = pdata['Norm'].to_numpy().reshape(int(len(pdata['Norm'].to_numpy())/N_gauss), N_gauss)[dataInd]
        line_wavs = pdata['Wavelength'].to_numpy().reshape(int(len(pdata['Wavelength'].to_numpy())/N_gauss), N_gauss)[dataInd]
        line_FWHM = pdata['Sigma'].to_numpy().reshape(int(len(pdata['Sigma'].to_numpy())/N_gauss), N_gauss)[dataInd]
    
        line_profiles = []
        conti_profile = self.eval_conti_all(contips, lams)*rescale[0]
        PL_profile = self.eval_PL(contips, lams)*rescale[0]
    
        for line_index, (norm, wav, fwhm, rescale_fac) in enumerate(zip(line_norms, line_wavs, line_FWHM, rescale)):
            if norm == 0:
                continue
            temp_line = self.eval_line_full([0, fwhm, norm, wav], lams)*rescale_fac
            current_inset_window = np.clip(lams[np.where(temp_line > 0.01*np.nanmax(temp_line))], plotWindow[0], plotWindow[1])
            if len(current_inset_window) > 0:
                current_inset_window = np.array([np.nanmin(current_inset_window), np.nanmax(current_inset_window)])
            else:
                current_inset_window = np.copy(plotWindow)
            if line_index in line_indices or lineCompInd < 0:
                dashed_color = 'r'
            else:
                dashed_color = 'g'
            data_ax.plot(lams[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])], 
                                     (conti_profile+temp_line)[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])],
                                     '--', c = dashed_color)

            line_profiles.append(temp_line)

        tot_line = np.array([0 for i in lams])
        for profile in line_profiles:
            tot_line = self.sum_nan_arrays(tot_line, profile)

        current_inset_window = np.clip(lams[np.where(tot_line > 0.01*np.nanmax(tot_line))], plotWindow[0], plotWindow[1])
        if len(current_inset_window) > 0:
            current_inset_window = np.array([np.nanmin(current_inset_window), np.nanmax(current_inset_window)])
        else:
            current_inset_window = np.copy(plotWindow)
        data_ax.tick_params(which='both',axis='both', direction='in', bottom=True, top=True, right=True, labelleft=True, labelbottom=False)
        data_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                             data_flux[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                             c = 'k', label='Data', zorder=0)
        data_ax.set_ylim(np.clip(data_ax.get_ylim(), 0.25*np.min(((conti_profile+tot_line)[tot_line > 0.01])), 1.5*np.max((conti_profile+tot_line)[tot_line > 0.01])))

        data_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                             PL_profile[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                             c = 'orange', label='Power-law')
        data_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                             conti_profile[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                             c = 'b', label='Pseudo-Continuum')

        data_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                             (conti_profile+tot_line)[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                             c = 'b')
        data_ax.plot(lams[tot_line > 0.01][np.logical_and(lams[tot_line > 0.01]>plotWindow[0], lams[tot_line > 0.01]<plotWindow[1])],
                             (conti_profile+tot_line)[tot_line > 0.01][np.logical_and(lams[tot_line > 0.01]>plotWindow[0], lams[tot_line > 0.01]<plotWindow[1])],
                             c = 'r')
        data_ax.plot(lams[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])], 
                             (conti_profile+tot_line)[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])],
                             c = 'r', label='Line')
        e_vscale = data_ax.get_ylim()[0]
        data_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                             e_vscale+data_eflux[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                             c = 'gainsboro', zorder=-1)
                         
        data_ax.yaxis.set_minor_locator(AutoMinorLocator())
        data_ax.xaxis.set_minor_locator(AutoMinorLocator())
        data_ax.tick_params(labelleft=True)
        data_ax.legend(facecolor='white', framealpha=1.0, loc=2, fontsize=12)
        data_ax.set_ylabel(r'$\rm f_{\lambda, \rm{obs}}$ ($10^{-17} \rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)')

        for i in vspanRanges:
            data_ax.axvspan(i[0], i[1], fc='b', alpha=0.5)
    
        if resid_ax != None:
            resid_ax.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, right=True, labelleft=True)
            resid_ax.plot(lams[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])], 
                                     ((data_flux-conti_profile-tot_line)/data_eflux)[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])],
                                      c = 'r', zorder=3)
            resid_scale = 1.2*np.max(np.abs(resid_ax.get_ylim()))
            resid_ax.set_ylim([-resid_scale, resid_scale])
            resid_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])], 
                                 ((data_flux-conti_profile-tot_line)/data_eflux)[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])])
    
            resid_ax.plot(lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])],
                                  [0 for i in lams[np.logical_and(lams>plotWindow[0], lams<plotWindow[1])]], 
                                  c = 'k', zorder=4)
            resid_ax.set_ylim(np.clip(resid_ax.get_ylim(), -15, 15))
            resid_ax.plot(atm_lams_plot, [0.93*(resid_ax.get_ylim()[1]-resid_ax.get_ylim()[0])+resid_ax.get_ylim()[0] for i in atm_lams_plot], 
                                 'o', c='grey', alpha=0.7, zorder=2)
    

            resid_ax.plot(plotWindow, [-3.0, -3.0], '--', c='k', zorder=4)
            resid_ax.plot(plotWindow, [3.0, 3.0], '--', c='k', zorder=4)
            if np.max(np.abs(resid_ax.get_ylim())) < 5:
                resid_ax.set_ylim(-5,5)
            resid_ax.xaxis.set_minor_locator(AutoMinorLocator())
            resid_ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            resid_ax.tick_params(labelleft=True)
            resid_ax.set_ylabel('Residual ($\\sigma$)')
            resid_ax.set_xlabel('Rest Wavelength $(\\rm{\\AA})$')
    
            for i in vspanRanges:
                resid_ax.axvspan(i[0], i[1], fc='b', alpha=0.5)



    ###########
    # Methods #
    ###########
    
    def strToArray(self, inputStr):
        outArray = np.array([i.split('-') for i in inputStr.split('|')]).astype(np.float64)
        return outArray

    def create_mask_window(self, lams, windows):
        mask = [False for x in lams]
        for window in windows:
            wind_lo, wind_hi = window
            for index, lam in enumerate(lams):
                if lam > wind_lo and lam < wind_hi:
                    mask[index] = True
        return mask

    def sigma_mask_buffer(self, box_width, sigma, lams, flux, err, mask_buffer):
        master_mask = [True for x in lams]
        for lam_index, lam in enumerate(lams):
            if lam_index < box_width/2.:
                boxed_flux = flux[:int(lam_index+box_width/2.)]
                master_low_index = 0
                master_high_index = int(lam_index+box_width/2.)
            elif lam_index > len(lams) - box_width/2.:
                boxed_flux = flux[int(lam_index-box_width/2.):]
                master_low_index = int(lam_index-box_width/2.)
                master_high_index = len(lams)
            else:
                boxed_flux = flux[int(lam_index-box_width/2.):int(lam_index+box_width/2.)]
                master_low_index = int(lam_index-box_width/2.)
                master_high_index = int(lam_index+box_width/2.)
            filtered_data = sigma_clip(boxed_flux, sigma=sigma, maxiters=5, masked=True)
            invert_mask = np.array([not filt for filt in filtered_data.mask])
            master_mask[master_low_index:master_high_index] *= invert_mask
    
        new_master_mask = np.copy(master_mask)
        for i, val in enumerate(master_mask):
            if not val and i >= mask_buffer and len(master_mask) - i >= mask_buffer:
                new_master_mask[i-mask_buffer:i] = [False]*mask_buffer
                new_master_mask[i:i+mask_buffer] = [False]*mask_buffer

        lams = lams[new_master_mask]
        flux = flux[new_master_mask]
        err = err[new_master_mask]

        return [lams, flux, err]

    def write_to_file(self, filename, dataheader, datarow):
        if os.path.exists(filename):
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(datarow)
        else:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(dataheader)
                writer.writerow(datarow)
        return


    def create_synthetic(self, lams, flux, eflux):
        new_flux = rand.normal(flux, eflux)
        return [lams, new_flux, eflux]    
        
    def sum_nan_arrays(self, a,b):
        ma = np.isnan(a)
        mb = np.isnan(b)
        return np.where(ma&mb, np.nan, np.where(ma,0,a) + np.where(mb,0,b))

    def calc_fwhm(self, xx, line_yy, line_center):
        res = specutils_fwhm(Spectrum1D(spectral_axis=xx*u.angstrom, flux=line_yy*u.Jy)).value/line_center * con.c/1000.
        return res

    def first_moment(self, xx, yy):
        M0 = simps(yy,xx) 
        M1 = simps(yy*xx, xx)/M0
        return M1

    def second_moment(self, xx, yy):
        M0 = simps(yy, xx)
        M2 = simps(yy*(xx**2), xx)/M0 - (self.first_moment(xx,yy))**2
        return M2

    def calc_ew(self, xx, yy, contiyy): # region is [min, max]
        spec_continorm = (contiyy+yy)/contiyy
        spec_continorm = Spectrum1D(spectral_axis = xx*u.angstrom, flux = spec_continorm*u.Jy)
        ew = np.abs(equivalent_width(spec_continorm))
        return ew.value

    def calc_line_flux(self, xx, yy):
        spec_flux_line = Spectrum1D(spectral_axis=xx*u.angstrom, flux=yy*u.Jy)
        integrated_flux = line_flux(spec_flux_line)
        integrated_flux = integrated_flux.value
        return integrated_flux
        
    def general_med_wav(self, xx, line_yy):
        line_flux = sum(line_yy)
        summed_flux = 0
        for index, y in enumerate(line_yy):
            summed_flux += y
            if summed_flux > line_flux/2.:
                med_wav = xx[index]
                return med_wav
        return 0

        #############
        # Continuum #
        #############
    def evalQSOGen(self, pp, xx):
        model = Quasar_sed(z=pp[0], LogL3000=pp[1], wavlen=xx, ebv=pp[2], M_i=pp[3], tbb=pp[4], 
                           bbnorm=pp[5], scal_emline=pp[6], beslope=pp[7], fragal=pp[8], gplind=pp[9])
        return model

    def QSOGen_resid(self, p, data):
        xx, yy, e_yy = data
        pp = np.array(list(p.valuesdict().values()))
        model = self.evalQSOGen(pp, xx)
        return (yy-model.flux)/e_yy
        
    def conti_residuals(self, p, data):
        xx, yy, e_yy, windows = np.array(data, dtype="object")
        mask = self.create_mask_window(xx, windows)
        conti_xdata, conti_ydata, econti_ydata = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
        pp = np.array(list(p.valuesdict().values()))
        conti = self.eval_conti_all(pp, xx)[mask]
        return (conti_ydata-conti)/econti_ydata

    def eval_PL(self, p, xx):
        f_pl = p[0] * (np.array(xx) / 3000.0) ** p[1]
        return f_pl


    def Fe_flux(self, pp, xval, Fe_model, init_fwhm):
        # Fit the Fe component (both UV and optical) #
        # VW01 1200, 3500 #
        # T06 2200, 3500 #
        # M16 2200, 3500 #
        # BV08 2200, 7000 #
        yval = np.zeros_like(xval)

        fe_wave = Fe_model[:, 0] 
        fe_flux = Fe_model[:, 1]

        xval_lo, xval_hi = np.nanmin(fe_wave), np.nanmax(fe_wave)

        ind = np.where((fe_wave > xval_lo) & (fe_wave < xval_hi), True, False)

        fe_wave = fe_wave[ind]
        fe_flux = fe_flux[ind]

        pix_avg = (fe_wave+np.roll(fe_wave,1))[1:]/2.
        pix_dispersion = np.median((fe_wave-np.roll(fe_wave,1))[1:]/pix_avg*con.c/1000.)

        Fe_FWHM = pp[1] # FWHM of FeII
        xval_new = xval * (1.0 + pp[2]) # percentage wavelength shift

        ind = np.where((xval_new > xval_lo) & (xval_new < xval_hi), True, False)
        if np.sum(ind) > 100:
            if Fe_FWHM <= init_fwhm:
                sig_conv = np.sqrt((init_fwhm+10.) ** 2 - init_fwhm ** 2) / 2. / np.sqrt(2. * np.log(2.))
            else:
                sig_conv = np.sqrt(Fe_FWHM ** 2 - init_fwhm ** 2) / 2. / np.sqrt(2. * np.log(2.))  # in km/s
            # Get sigma in pixel space
            sig_pix = sig_conv / pix_dispersion # km/s dispersion
            khalfsz = np.round(4 * sig_pix + 1, 0)
            xx = np.arange(0, khalfsz * 2, 1) - khalfsz
            kernel = np.exp(-xx ** 2 / (2 * sig_pix ** 2))
            kernel = kernel / np.sum(kernel)

            flux_Fe_conv = np.convolve(fe_flux, kernel, 'same')
            tck = interpolate.splrep(fe_wave, flux_Fe_conv)
            yval[ind] = pp[0] * interpolate.splev(xval_new[ind], tck)
        return yval


    def Balmer_conti(self, pp, xval):
        """Fit the Balmer continuum from the model of Dietrich+02"""
        # xval = input wavelength, in units of A
        # pp=[norm, Te, tau_BE] -- in units of [--, K, --]

        if pp[0] > 0:
            lambda_BE = 3646.  # A
            bb = BlackBody(pp[1]*u.K)  
            bbflux = 1E-8*bb(xval*u.AA)*(np.pi*u.sr)*(con.c*1E10*u.AA*u.Hz)/(xval*u.AA)**2 # in units of ergs/cm2/s/A
            tau = pp[2]*(xval/lambda_BE)**3
            result = pp[0]*bbflux.value*(1.-np.exp(-tau))
            ind = np.where(xval > lambda_BE, True, False)
            if ind.any() == True:
                result[ind] = 0.
        else:
            result = np.array([0 for i in xval])
        
        return result


    def eval_conti_all(self, p, xx):
        """
        Continuum components described by 14 parameters
         pp[0]: norm_factor for continuum f_lambda = (lambda/3000.0)^{-alpha}
         pp[1]: slope for the power-law continuum
         pp[2]: norm_factor for the MgII Fe_template
         pp[3]: FWHM for the MgII Fe_template
         pp[4]: small shift of wavelength for the MgII Fe template
         pp[5:8]: same as pp[2:4] but for the Hbeta/Halpha Fe template
         pp[8:11]: norm, Te and Tau_e for the Balmer continuum at <3646 A
        """
        f_pl = self.eval_PL(p, xx)
        f_Fe_MgII = self.Fe_flux(p[2:5], xx, self.fe_uvs[self.Fe_uv_ind], self.init_fe_uv_fwhms[self.Fe_uv_ind])  # iron flux for MgII line region
        f_Fe_Balmer = self.Fe_flux(p[5:8], xx, self.fe_ops[self.Fe_opt_ind], self.init_fe_op_fwhms[self.Fe_opt_ind])  # iron flux for balmer line region
        f_Balmer = self.Balmer_conti(p[8:11], xx)

        yval = f_pl + f_Fe_MgII + f_Fe_Balmer + f_Balmer
        return yval

        ########################
        # General Line Fitting #
        ########################    
    def eval_line_full(self, p, xx):
        # p = [Skew, Scale FWHM (km/s), Norm, Central Wavelength]
        scale_AA = self.fwhm_to_angstrom(p[1], p[3])
        return p[2]* skewnorm.pdf(xx, p[0], p[3], scale_AA)
    
    def residual_line(self, p, data):
        # Data should be residual spectra
        xx, yy, e_yy, windows = np.array(data, dtype="object")
        mask = self.create_mask_window(xx, windows)
        xx, yy, e_yy = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
        pp = np.array(list(p.valuesdict().values()))
        line_result = self.eval_all_lines(pp, xx)

        return (yy - line_result)/e_yy


    def fwhm_to_angstrom(self, fwhm, line_center):
        angstrom = fwhm/con.c * 1000. * line_center
        return angstrom


    def eval_all_lines(self, p, xx):
        # Mainly depends on eval_line_full
        num_line_params = len(self.line_header)
        line_result = np.array([0. for i in xx])
        num_lines = int(len(p)/num_line_params)
        for i in range(num_lines):
            current_line_params = p[int(num_line_params*i):int(num_line_params*(i+1))]
            eval_params = current_line_params
            line_res = self.eval_line_full(eval_params, xx)
            line_result += np.array(line_res)
        return line_result

    def out_line_res(self, lams, p, line_names):
        num_line_params = len(self.line_header)
        num_lines = int(len(p)/num_line_params)
        print('Formatted Line Parameters:')
        print(self.line_header)
        for i in range(num_lines):
            current_line_params = np.array(p[int(num_line_params*i):int(num_line_params*(i+1))])
            eval_params = current_line_params
    
            current_line_yy = self.eval_line_full(eval_params, lams)
    
            print(line_names[i], '\t', current_line_params)

            if self.dataOut:
                dataout_filename = self.dataOutPath + self.runName + '.csv'
                data_row = [self.runName, line_names[i]]+ [self.norm_median] + list(current_line_params) + list(self.conti_bestfit)
                out_header = ['Filename', 'Name', 'Norm_Factor'] + list(self.line_header) + list(self.conti_header)
                self.write_to_file(dataout_filename, out_header, data_row)

    def readLineFile(self, lineFile, lineCompInd):
        line_list_pdata = pd.read_csv(lineFile)
        N_gauss = len(line_list_pdata['Name'])

        line_names, line_wavs = [], []
        line_indices = []
        for ind, (name, c_wav) in enumerate(zip(line_list_pdata['Name'].to_numpy(),line_list_pdata['Central Wavelength'])):
            if name != '' and name != ' ' and not pd.isnull(name):
                if len(line_names) > 0:
                    line_indices.append(indices)
                indices = [ind]
                line_names.append(name)
                line_wavs.append(c_wav)
            else:
                indices.append(ind)
            if ind == len(line_list_pdata['Name'].to_numpy())-1:
                line_indices.append(indices)

        #print(line_names, line_wavs, line_indices)
        vac_wav = line_wavs[lineCompInd]
        line_index = line_indices[lineCompInd]
        return [N_gauss, vac_wav, line_index]
        
        
    def mergeUncertainty(self, fileList, outfile='test.csv', mergeAll=False):
        pdata1 = pd.read_csv(fileList[0])
        headers = pdata1.columns
        headers = np.array([i for i in headers if i[0]!='e'])
        
        pdataList = [pd.read_csv(i) for i in fileList]

        for header in headers:
            valsList = np.array([p[header].to_numpy() for p in pdataList])
            evalsList = np.array([p['e'+header].to_numpy() for p in pdataList])
            eres = [0 for i in evalsList[0]]
            for val, eval in zip(valsList, evalsList):
                eres += eval**2
            eres = np.array([np.sqrt(i) for i in eres])
            if mergeAll:
                res, eres = self.flattenWeightedAvg(valsList, evalsList)
                pdata1[header] = res
                pdata1['e'+header] = eres
            else:
                eres[eres==0] = np.nan
                pdata1['e'+header] = eres

        pdata1.to_csv(outfile, index=False)
        
    def flattenWeightedAvg(self, resultMatrix, errMatrix):
        result1D, err1D = [], []
        resultMatrix, errMatrix = np.transpose(resultMatrix), np.transpose(errMatrix)
        for line, errline in zip(resultMatrix, errMatrix):
            result1D.append(self.weighted_avg(line, errline).n)
            err1D.append(self.weighted_avg(line, errline).s)
        return result1D, err1D
            
    def weighted_avg(self, data, weights): # weights in std
        new_test = [ufloat(a,b) for a, b in zip(data, weights)]
        data, weights = np.array(data), 1/np.array(weights)**2
        result = np.nansum(new_test*weights)/np.nansum(data*weights/data)
        if result >= 0 or result <= 0:
            return result
        else:
            return ufloat(np.nan, np.nan)

    def virial_BH_mass(self, mono_lum, line_FWHM, CIV_bshift=0):

        # Shen 2011
        mbh = 10**(6.74)*(mono_lum/10**(44))**(0.62) * (line_FWHM/10**3)**2 # MgII
        # Vestergaard & Osmer 2009
        #mbh = 10**(6.86)*(mono_lum/10**(44))**(0.5) * (line_FWHM/10**3)**2 # MgII
        # Le, Woo, & Xue 2020
        #mbh = 10**(7.04)*(mono_lum/10**(44))**(0.5) * (line_FWHM/10**3)**2 # MgII
        # Woo et al. 2018 (weird)
        #mbh = 10**(6.75)*(mono_lum/10**(44))**(0.74) * (line_FWHM/10**3)**(2.35) # MgII
        
        # Vestergaard & Peterson 2006
        #mbh = 10**(6.66)*(mono_lum/10**(44))**(0.53) * (line_FWHM/10**3)**2 # CIV
        # Shen 2011 (weird)
        #mbh = 10**(7.295)*(mono_lum/10**(44))**(0.471) * (line_FWHM/10**3)**(0.242) # CIV
        # Coatman et al. 2017
        #line_FWHM = line_FWHM/(ufloat(0.36, 0.03)*(CIV_bshift/1000)+ufloat(0.61, 0.04))
        #mbh = 10**(6.71)*(mono_lum/10**(44))**(0.53) * (line_FWHM/10**3)**2 # CIV

        # Le 2020
        #mbh = 10**(6.87)*(mono_lum/10**(44))**(0.53) * (line_FWHM/10**3)**2 # Hbeta
        # Vestergaard & Peterson 2006
        #mbh = 10**(6.91)*(mono_lum/10**(44))**(0.50) * (line_FWHM/10**3)**2 # Hbeta
        return mbh # in solar masses







