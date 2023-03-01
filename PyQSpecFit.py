# PyQSpecFit: Code for modelling emission lines in QSO spectra
# Auther: Samuel Lai, ANU
# Email: samuel.lai (AT) anu (DOT) edu (DOT) au

# Version 1.0
# 2023-02-28

###################
# Version History #
###################
# V1.0 - Forked from PyQSpecFit_Functional_v2
# 	
#
#

#########
# To Do #
#########
# Plotting 
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
#from colossus.cosmology import cosmology

# Numpy and Pandas and I/O
import numpy as np
import pandas as pd
import numpy.random as rand
import csv

# Astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from kapteyn import kmpfit
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

# Uncertainty
from uncertainties import ufloat
from uncertainties.umath import *

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator



class PyQSpecFit():
	def __init__(self, dataDir='data/'):
		"""
		Get input data

		Parameters:
		-----------
		dataDir: string
			points to location of csv datafiles setup with columns ['Wavelength', 'Flux', 'eFlux']
			line models will follow units of the Flux column
		"""
		
		self.dataDir = dataDir
		
		template_path = 'Fe_Templates/'
		VW01 = np.genfromtxt(template_path+'fe_uv_VW01.txt')
		VW01[:,0], VW01[:,1] = 10**VW01[:,0], VW01[:,1]*10**15
		T06 = np.genfromtxt(template_path+'Tsuzuki+2006.uvtemp.tab.txt')
		T06[:,0], T06[:,1] = T06[:,0], T06[:,1]*10**15
		M16 = np.genfromtxt(template_path+'feII_UV_Mejia-Restrepo_et_al_2016_2200-3646AA.data.txt')
		M16[:,0], M16[:,1] = M16[:,0], M16[:,1]*10
		BV08 = np.genfromtxt(template_path+'BruhweilerVerner2008/d11-m20-21-735.txt')
		BV08[:,0], BV08[:,1] = BV08[:,0], BV08[:,1]*10**(-6)
		self.fe_uvs = [VW01, T06, M16, BV08]
		self.init_fe_uv_fwhms = [900., 900., 900., 800.]

		BG92 = np.genfromtxt(template_path + 'fe_optical_BG92.txt')
		BG92[:,0], BG92[:,1] = 10**BG92[:,0], BG92[:,1]*10**15
		T06_opt = np.genfromtxt(template_path + 'Tsuzuki+2006.opttemp.tab.txt')
		T06_opt[:,0], T06_opt[:,1] = T06_opt[:,0], T06_opt[:,1]*10**15
		P22 = np.genfromtxt(template_path + 'Park+2022.Mrk493_OpticalFeTemplate.txt')
		P22[:,0], P22[:,1] = P22[:,0], P22[:,1]*10.
		self.fe_ops = [BG92, T06_opt, P22, BV08]
		self.init_fe_op_fwhms = [1200., 1200., 800., 600.]
	
	def runFit(self, lineFile, contiWindow, lineWindow, N_fits=1,
			   normSwitch=True, dataOut=True, syntheticFits=False,
			   sig_clip=False, clipSigma=3, clipBoxWidth=50, clipBufferWidth=3,
			   lineShift=0, allowedShift=30,
			   useBalmer=False, useFe=False, Fe_uv_ind=0, Fe_opt_ind=0,
			   dataOutPath='Line_Params/'):
			
			self.Fe_uv_ind = Fe_uv_ind
			self.Fe_opt_ind = Fe_opt_ind
			self.useBalmer = useBalmer
			self.useFe = useFe
			self.dataOut = dataOut
			datafiles = glob.glob(self.dataDir+'*.csv')

			for i in range(N_fits):
				for datafile in datafiles:
					self.datafile = datafile[len(self.dataDir):]
					datafile = datafile[len(self.dataDir):]
					obj_name = (datafile[:-4])
					print(datafile[:-4])

					pdata = pd.read_csv(self.dataDir+datafile)
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

					if syntheticFits:
						lams, flux, eflux = self.create_synthetic(lams, flux, eflux)

					if normSwitch:
						self.norm_median = np.nanmedian(flux)
						flux = flux/self.norm_median
						eflux = eflux/self.norm_median


					#################
					# Fit Continuum #
					#################

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
					self.conti_header = ['PL_Norm', 'PL_Slope', 
									'Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 
									'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del', 
									'Balmer_norm', 'Balmer_Te', 'Balmer_tau']
				
					tmp_parinfo = [{'limits': (0, 1E3)}, {'limits': (-5, 3)}, 
								  {'limits': (0., 1E3)}, {'limits': (1E3, 1E4)}, {'limits': (-0.02, 0.02)},
									{'limits': (0., 1E3)}, {'limits': (1E3, 1E4)}, {'limits': (-0.02, 0.02)},
									{'limits': (0., 1E3)}, {'limits': (1E3, 5E4)}, {'limits': (0.1, 4.)}]
					init_params = [1.0, 0., 0., 3000., 0., 0., 3000., 0., 0., 10000, 0.1]

					print('Fitting Continuum...')
					print()
					fitobj = kmpfit.Fitter(residuals=self.conti_residuals, data=(lams, flux, eflux, contiWindow))
					fitobj.parinfo = tmp_parinfo
					fitobj.fit(params0=init_params)
					print("Best-fit parameters: ", fitobj.params)
					print("Iterations:               ", fitobj.niter)
					print("Reduced Chi^2:              ", fitobj.rchi2_min)
					#print("Covariance Errors:           ", fitobj.xerror)
					print("Std Errors:          ", fitobj.stderr)
					print()

					self.conti_bestfit = fitobj.params
					conti_stderrs = fitobj.stderr


					################
					# Line Fitting #
					################
					residual_flux = flux - self.eval_conti_all(self.conti_bestfit, lams)
					print('Fitting Lines...')
					print()

					pdata_lines = pd.read_csv(lineFile)
					line_names = pdata_lines['Name']
					line_wave = pdata_lines['Central Wavelength']
					line_norms = pdata_lines['Init_Norm']
		
					fwhm_lows = pdata_lines['FWHM_low']
					fwhm_highs = pdata_lines['FWHM_high']

					# SkewNorm Fitting
					init_params = []
					init_parinfo = []
					self.line_header = ['Skew', 'FWHM', 'Norm', 'Wavelength']

					for index, line in enumerate(line_names):
						# Line Parameters: Skew, Scale, Norm, Central Wavelength
						line_init_params = [0., fwhm_lows[index], line_norms[index], line_wave[index]+lineShift]
						line_parinfo = [{'fixed': True}, {'limits': (fwhm_lows[index], fwhm_highs[index])}, {'limits': (0., 1E2)}, {'limits': (line_wave[index]+lineShift-allowedShift, line_wave[index]+lineShift+allowedShift)}]
						init_params += line_init_params
						init_parinfo += line_parinfo

					xx, yy, e_yy = lams, residual_flux, eflux

					if sig_clip:
						mask = self.create_mask_window(xx, lineWindow)
						xx, yy, e_yy = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
						xx, yy, e_yy = self.sigma_mask_buffer(50, 3.0, xx, yy, e_yy, 3)

					fitobj = kmpfit.Fitter(residuals=self.residual_line, data=(xx, yy, e_yy, lineWindow))
					fitobj.parinfo = init_parinfo
					fitobj.fit(params0=init_params)
					print("Best-fit parameters: ", fitobj.params)
					print("Iterations:               ", fitobj.niter)
					print("Reduced Chi^2:              ", fitobj.rchi2_min)
					#print("Covariance Errors:           ", fitobj.xerror)
					print("Std Errors:          ", fitobj.stderr)
					print()

					line_bestfit = fitobj.params
					line_stderrs = fitobj.stderr

					self.out_line_res(lams, line_bestfit, line_stderrs, line_names, dataOutPath)



	def evalLineProperties(self, lineFile, fitFile, redshift, monoLumAngstrom = 3000.,
						   lamWindow=[1200, 8000], lineCompInd = 0,
						   useBalmer=False, useFe=False, Fe_uv_ind=0, Fe_opt_ind=0,
						   outDir='Line_Properties/'):
		# Assumes flux is in units of erg/s/cm2/Angstrom #
		self.useBalmer = useBalmer
		self.useFe = useFe
		self.Fe_uv_ind = Fe_uv_ind
		self.Fe_opt_ind = Fe_opt_ind
		res_props_header = ['FWHM', 'Sigma', 'Blueshift', 'EW', 'pWave', 'iLum', 'Mono_Lum']

		self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		dl = self.cosmo.luminosity_distance(redshift).to(u.cm)
		
		line_list_pdata = pd.read_csv(lineFile)
		total_line_shape = len(line_list_pdata['Name'])

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

		#print(line_names, line_wavs)
		vac_wav = line_wavs[lineCompInd]
		line_index = line_indices[lineCompInd]
	
		temp_res_props = [[] for i in res_props_header]
		temp_res_props_err = [[] for i in res_props_header]
		lams = np.linspace(lamWindow[0], lamWindow[1], int(np.ptp(lamWindow)*5.))
	
		pdata = pd.read_csv(fitFile)
		
		N_gauss = total_line_shape
	
		rescale = pdata['Norm_Factor'].to_numpy().reshape(int(len(pdata['Norm_Factor'].to_numpy())/N_gauss), N_gauss)
		conti_header = ['PL_Norm', 'PL_Slope', 'Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 
						'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del', 'Balmer_norm', 'Balmer_Te', 'Balmer_tau']
		contips = np.transpose([pdata[i].to_numpy().reshape(int(len(pdata[i].to_numpy())/N_gauss), N_gauss) for i in conti_header])[0]

		line_norms = pdata['Norm'].to_numpy().reshape(int(len(pdata['Norm'].to_numpy())/N_gauss), N_gauss)
		line_wavs = pdata['Wavelength'].to_numpy().reshape(int(len(pdata['Wavelength'].to_numpy())/N_gauss), N_gauss)
		line_FWHM = pdata['FWHM'].to_numpy().reshape(int(len(pdata['FWHM'].to_numpy())/N_gauss), N_gauss)
		
		for index, (norms, wavs, fwhms, contip, rescale_facs) in enumerate(zip(line_norms, line_wavs, line_FWHM, contips, rescale)):

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
	
		res_props = [np.nanmean(i) for i in temp_res_props]
		res_props_err = [np.std(i) for i in temp_res_props]
		#res_props = [np.nanmean(sigma_clip(i, sigma=3, maxiters=5)) for i in temp_res_props]
		#res_props_err = [np.std(sigma_clip(i, sigma=3, maxiters=5)) for i in temp_res_props]
	
		#print(res_props, res_props_err)

		pdata = pd.DataFrame()
		for ind, val in enumerate(res_props_header):
			pdata[val] = [res_props[ind]]
			pdata['e'+val] = [res_props_err[ind]]
		pdata.to_csv(outDir+'example.csv', index=False)
	
	
	def plotLineFits(self, data_ax, resid_ax, lineFile, dataFile, fitFile, redshift,
					 plotWindow=[1200, 8000], dataInd = 0,
					 useBalmer=False, useFe=False, Fe_uv_ind=0, Fe_opt_ind=0,
					 outDir='Fit_Figs/'):
		self.useBalmer = useBalmer
		self.useFe = useFe
		self.Fe_uv_ind = Fe_uv_ind
		self.Fe_opt_ind = Fe_opt_ind
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
		N_gauss = len(line_list_pdata['Name'])
	
		rescale = pdata['Norm_Factor'].to_numpy().reshape(int(len(pdata['Norm_Factor'].to_numpy())/N_gauss), N_gauss)[dataInd]*norm_fac
		conti_header = ['PL_Norm', 'PL_Slope', 'Fe_UV_Norm', 'Fe_UV_FWHM', 'Fe_UV_del', 
						'Fe_Opt_Norm', 'Fe_Opt_FWHM', 'Fe_Opt_del', 'Balmer_norm', 'Balmer_Te', 'Balmer_tau']
		contips = np.transpose([pdata[i].to_numpy().reshape(int(len(pdata[i].to_numpy())/N_gauss), N_gauss) for i in conti_header])[0][dataInd]
		line_norms = pdata['Norm'].to_numpy().reshape(int(len(pdata['Norm'].to_numpy())/N_gauss), N_gauss)[dataInd]
		line_wavs = pdata['Wavelength'].to_numpy().reshape(int(len(pdata['Wavelength'].to_numpy())/N_gauss), N_gauss)[dataInd]
		line_FWHM = pdata['FWHM'].to_numpy().reshape(int(len(pdata['FWHM'].to_numpy())/N_gauss), N_gauss)[dataInd]
	
		line_profiles = []
		conti_profile = self.eval_conti_all(contips, lams)*rescale[0]
		PL_profile = self.eval_PL(contips, lams)*rescale[0]
	
		for line_index, (norm, wav, fwhm, rescale_fac) in enumerate(zip(line_norms, line_wavs, line_FWHM, rescale)):
			temp_line = self.eval_line_full([0, fwhm, norm, wav], lams)*rescale_fac
		
			if norm == 0:
				continue
			current_inset_window = np.clip(lams[np.where(temp_line > 0.01*np.nanmax(temp_line))], plotWindow[0], plotWindow[1])
			current_inset_window = np.array([np.nanmin(current_inset_window), np.nanmax(current_inset_window)])
			data_ax.plot(lams[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])], 
									 (conti_profile+temp_line)[np.logical_and(lams>current_inset_window[0], lams<current_inset_window[1])],
									 '--', c = 'r')

			line_profiles.append(temp_line)

		tot_line = np.array([0 for i in lams])
		for profile in line_profiles:
			tot_line = self.sum_nan_arrays(tot_line, profile)

		current_inset_window = np.clip(lams[np.where(tot_line > 0.01*np.nanmax(tot_line))], plotWindow[0], plotWindow[1])
		current_inset_window = np.array([np.nanmin(current_inset_window), np.nanmax(current_inset_window)])

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
		resid_ax.xaxis.set_minor_locator(AutoMinorLocator())
		resid_ax.yaxis.set_minor_locator(AutoMinorLocator())
	
		data_ax.tick_params(labelleft=True)
		data_ax.legend(facecolor='white', framealpha=1.0, loc=2, fontsize=12)
		data_ax.set_ylabel(r'$\rm f_{\lambda, \rm{obs}}$ ($10^{-17} \rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)')

		resid_ax.tick_params(labelleft=True)
		resid_ax.set_ylabel('Residual ($\\sigma$)')
		resid_ax.set_xlabel('Rest Wavelength $(\\rm{\\AA})$')


	###########
	# Methods #
	###########

	def create_mask_window(self, lams, windows):
		mask = [False for x in lams]
		for window in windows:
			wind_lo, wind_hi = window
			for index, lam in enumerate(lams):
				if lam > wind_lo and lam < wind_hi:
					mask[index] = True
		return mask

	def sigma_mask_buffer(self, box_width, sigma, lams, flux, err, mask_buffer):
		spectrum = Spectrum1D(spectral_axis=lams * u.angstrom, flux=flux*u.Jy)
		box_smoothed = box_smooth(spectrum, width = box_width)
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
	def conti_residuals(self, p, data):
		xx, yy, e_yy, windows = np.array(data, dtype="object")
		mask = self.create_mask_window(xx, windows)
		conti_xdata, conti_ydata, econti_ydata = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
		conti = self.eval_conti_all(p, xx)[mask]
		return (conti_ydata - conti)/econti_ydata

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

		lambda_BE = 3646.  # A
		bbflux = BlackBody(pp[1]*u.K, 1*u.erg/(u.cm**2*u.s*u.AA*u.sr))  
		bbflux = bbflux(xval*u.AA).value*np.pi # in units of ergs/cm2/s/A
		tau = pp[2]*(xval/lambda_BE)**3
		result = pp[0]*bbflux*(1.-np.exp(-tau))
		ind = np.where(xval > lambda_BE, True, False)
		if ind.any() == True:
			result[ind] = 0.
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

		yval = f_pl
		if self.useFe:
			yval += f_Fe_MgII + f_Fe_Balmer
		if self.useBalmer:
			yval += f_Balmer
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
		line_result = self.eval_all_lines(p, xx)

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

	def out_line_res(self, lams, p, ep, line_names, outDir):
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
				dataout_filename = outDir + self.datafile[:-4] + '.csv'
				data_row = [self.datafile, line_names[i]]+ [self.norm_median] + list(current_line_params) + list(self.conti_bestfit)
				out_header = ['Filename', 'Name', 'Norm_Factor'] + list(self.line_header) + list(self.conti_header)
				self.write_to_file(dataout_filename, out_header, data_row)

	









