# PyQSpecFit: Code for modelling emission lines in QSO spectra
# Auther: Samuel Lai, ANU
# Email: samuel.lai (AT) anu (DOT) edu (DOT) au

# Version 1.0
# 2023-02-28

###################
# Version History #
###################
# V1.0 - Forked from PyQSpecFit_Functional_v2, OOP
# 	
#
#

#########
# To Do #
#########
# Everything, OOP
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
import glob, os,sys,timeit
sys.path.append('../')
import warnings
import os
from os import path
#from colossus.cosmology import cosmology

# Numpy and Pandas and I/O
import numpy as np
import pandas as pd
from numpy.random import Generator, MT19937
import numpy.random as rand
import random
import csv

# Astropy
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
import astropy.units as u
from kapteyn import kmpfit
import math
from astropy.modeling.physical_models import BlackBody
from astropy.stats import sigma_clip

# Plotting 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm
from scipy import interpolate
import scipy.constants as con
from scipy.integrate import simps


# Specutils
from specutils import Spectrum1D
from specutils.analysis import line_flux, equivalent_width
from specutils.fitting import estimate_line_parameters, fit_lines
from specutils.manipulation import (extract_region, box_smooth, gaussian_smooth, trapezoid_smooth, median_smooth)
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from specutils.spectra import SpectralRegion
from specutils.analysis import fwhm as specutils_fwhm

# Uncertainty
from uncertainties import ufloat
from uncertainties.umath import *

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams['font.size'] = 12





class PyQSpecFit():
	def __init__(self, lams, flux, eflux):
		"""
		Get input data

		Parameters:
		-----------
		lams: 1-D array
			wavelength in unit of Angstrom in rest-frame
 
		flux: 1-D array
			flux density in unit of erg/s/cm^2/Angstrom

		eflux: 1-D array
			 1 sigma err with the same unit of flux

		"""

		self.inputLam = lams
		self.inputFlux = flux
		self.inputFluxError = eflux
		self.data = np.array([lams, flux, eflux])
	
	def runFit(self, dataDir, lineFile, N_fits=1, contiWindow, lineWindow,
			   normSwitch=True, dataOut=True,
			   sig_clip=False, clipSigma=3, clipBoxWidth=50, clipBufferWidth=3,
			   lineShift=0, allowedShift=30,
			   useBalmer=False, useFe=False, Fe_uv_ind=0, Fe_opt_ind=0):
		
		pass








####################
# Useful Functions #
####################

def create_mask_window(lams, windows):
    mask = [False for x in lams]
    for window in windows:
        wind_lo, wind_hi = window
        for index, lam in enumerate(lams):
            if lam > wind_lo and lam < wind_hi:
                mask[index] = True
    return mask
    
def sigma_mask_buffer(box_width, sigma, lams, flux, err, mask_buffer):
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
	
def write_to_file(filename, dataheader, datarow):
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
	
	
def create_synthetic(lams, flux, eflux):
    new_flux = rand.normal(flux, eflux)
    return [lams, new_flux, eflux]	

	#############
	# Continuum #
	#############
def conti_residuals(p, data):
    xx, yy, e_yy, windows = np.array(data, dtype="object")
    mask = create_mask_window(xx, windows)
    conti_xdata, conti_ydata, econti_ydata = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
    conti = eval_conti_all(p, xx)[mask]
    return (conti_ydata - conti)/econti_ydata

def eval_PL(p, xx):
    f_pl = p[0] * (np.array(xx) / 3000.0) ** p[1]
    return f_pl


def Fe_flux(pp, xval, Fe_model, init_fwhm):
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

    
def Balmer_conti(pp, xval):
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


def eval_conti_all(p, xx):
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
    f_pl = eval_PL(p, xx)
    f_Fe_MgII = Fe_flux(p[2:5], xx, fe_uvs[fe_uv_choice], init_fe_uv_fwhms[fe_uv_choice])  # iron flux for MgII line region
    f_Fe_Balmer = Fe_flux(p[5:8], xx, fe_ops[fe_op_choice], init_fe_op_fwhms[fe_op_choice])  # iron flux for balmer line region
    f_Balmer = Balmer_conti(p[8:11], xx)
    
    yval = f_pl
    if use_iron_template:
        yval += f_Fe_MgII + f_Fe_Balmer
    if use_Balmer:
        yval += f_Balmer
    return yval
    
	########################
	# General Line Fitting #
	########################	
def eval_line_full(p, xx):
	# p = [Skew, Scale FWHM (km/s), Norm, Central Wavelength]
	scale_AA = fwhm_to_angstrom(p[1], p[3])
	return p[2]* skewnorm.pdf(xx, p[0], p[3], scale_AA)
		
def residual_line(p, data):
    # Data should be residual spectra
    xx, yy, e_yy, windows = np.array(data, dtype="object")
    mask = create_mask_window(xx, windows)
    xx, yy, e_yy = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
    line_result = eval_all_lines(p, xx, None, None)

    return (yy - line_result)/e_yy

	
def fwhm_to_angstrom(fwhm, line_center):
	angstrom = fwhm/con.c * 1000. * line_center
	return angstrom

	
def eval_all_lines(p, xx, plot, inset):
	# Mainly depends on eval_line_full
    num_line_params = len(line_header)
    line_result = np.array([0. for i in xx])
    num_lines = int(len(p)/num_line_params)
    for i in range(num_lines):
        current_line_params = p[int(num_line_params*i):int(num_line_params*(i+1))]
        eval_params = current_line_params
        line_res = eval_line_full(eval_params, xx)
        if plot != None:
        	plot.plot(xx, line_res, '--', c='r', linewidth=1)
        if inset != None:
        	inset.plot(xx, line_res, '--', c='r', linewidth=1)
        line_result += np.array(line_res)
    return line_result
    
def out_line_res(p, ep, line_names):
    global datafile
    num_line_params = len(line_header)
    num_lines = int(len(p)/num_line_params)
    print('Formatted Line Parameters:')
    print(line_header)
    for i in range(num_lines):
        current_line_params = np.array(p[int(num_line_params*i):int(num_line_params*(i+1))])
        eval_params = current_line_params
        
        current_line_yy = eval_line_full(eval_params, lams)
        
        print(line_names[i], '\t', current_line_params)

        if line_data_out:
        	out_directory = "Line_Params/"
        	dataout_filename = out_directory + datafile[:-4] + '.csv'
        	data_row = [datafile, line_names[i]]+ [norm_median] + list(current_line_params) + list(conti_bestfit)
        	out_header = ['Filename', 'Name', 'Norm_Factor'] + list(line_header) + list(conti_header)
        	write_to_file(dataout_filename, out_header, data_row)



############
# Switches #
############

# Continuum Fitting Windows #
window_all = np.array([[1150., 1170.], [1275., 1290.],
                       [1690., 1705.], [1770., 1810.], [2060., 2340.], [2600., 2740.],
                       [2840., 3100.], [3775., 3832.], [4000., 4050.], [4200., 4230.],
                       [4435., 4640.], [5100., 5535.], [6005., 6035.], [6110., 6250.],
                       [6800., 7000.], [7160., 7180.], [7500., 7800.], [8050., 8150.]])

window_cut = np.array([[1275., 1290.],
                       [1690., 1705.], [1770., 1810.], [2060., 2340.], [2600., 2740.]])

FeiXu_windows = np.array([[1150, 1170.], [1275., 1290.], [1350., 1360.],
                         [1445., 1455.], [1700., 1705], [1770., 1800.], [2155., 2400.],
                         [2480., 2675.], [2925., 3400.]])
                         
FeiXu_windows_extend = np.array([[1445., 1455.], [1687., 1697.], [1973., 1983.], 
								[2155., 2400.], [2480., 2675.]])

FeiXu_windows_cut = np.array([[1445., 1455.], [1687., 1697.], [1973., 1983.]])

Nagao_windows = np.array([[1445., 1455.], [1973., 1983.]]) # Default
custom_conti_windows = np.array([[1445., 1455.], [1687., 1697.], [1973., 1983.]])
desperation_windows = np.array([[1432, 1440], [1687., 1697.]])
#desperation_windows = np.array([[1420, 1430], [1687., 1697.]])
extra_windows = np.array([[1320, 1325], [1370, 1390], [1445., 1455.], [1973., 1983.]])
PSS_J1723_2243_windows = np.array([[1445., 1455.], [1700., 1780.]])

J1144_windows_1 = np.array([[1973., 1983.], [2060., 2340.], [2600., 2740.], [2840., 3100.]])
J1144_windows_2 = np.array([[4200, 4230], [4435, 4700], [5100, 5535], [6000, 6250], [6800, 7000]])
J1144_windows_3 = np.array([[11500, 12300], [13100, 13400]])

conti_only_window = np.array([[1973., 1983.], [2060., 2340.], [2600., 2740.], [2840., 3100.],
							  [3600., 3800.], [4200, 4230], [4435, 4700]])

Hbeta_windows = np.array([[3500., 3800.], [4200, 4230], [4435, 4700]]) # Default

MgII_windows = np.array([[1770., 1810.], [2060., 2340.], [2600., 2740.], [2840., 3100.]])
MgII_XQ100_windows = np.array([[2200., 2740.], [2840., 3300.], [3500., 3650.]]) # Default
#MgII_XQ100_windows = np.array([[2200., 2750.], [2860., 3300.], [3500., 3650.]])


MgII_mod_windows = np.array([[1445., 1455.], [1687., 1697.], [1770., 1810.], [2060., 2180], [2305., 2340.], [2600., 2740.], [2850., 2950.]])

used_conti_windows = MgII_XQ100_windows

use_iron_template = True # Whether to use iron template
use_Balmer = False
template_path = 'Fe_Templates/'
VW01 = np.genfromtxt(template_path+'fe_uv_VW01.txt')
VW01[:,0], VW01[:,1] = 10**VW01[:,0], VW01[:,1]*10**15
T06 = np.genfromtxt(template_path+'Tsuzuki+2006.uvtemp.tab.txt')
T06[:,0], T06[:,1] = T06[:,0], T06[:,1]*10**15
M16 = np.genfromtxt(template_path+'feII_UV_Mejia-Restrepo_et_al_2016_2200-3646AA.data.txt')
M16[:,0], M16[:,1] = M16[:,0], M16[:,1]*10
BV08 = np.genfromtxt(template_path+'BruhweilerVerner2008/d11-m20-21-735.txt')
BV08[:,0], BV08[:,1] = BV08[:,0], BV08[:,1]*10**(-6)
fe_uv_choice = 0
fe_uvs = [VW01, T06, M16, BV08]
init_fe_uv_fwhms = [900., 900., 900., 800.]



BG92 = np.genfromtxt(template_path + 'fe_optical_BG92.txt')
BG92[:,0], BG92[:,1] = 10**BG92[:,0], BG92[:,1]*10**15
T06_opt = np.genfromtxt(template_path + 'Tsuzuki+2006.opttemp.tab.txt')
T06_opt[:,0], T06_opt[:,1] = T06_opt[:,0], T06_opt[:,1]*10**15
P22 = np.genfromtxt(template_path + 'Park+2022.Mrk493_OpticalFeTemplate.txt')
P22[:,0], P22[:,1] = P22[:,0], P22[:,1]*10.
fe_op_choice = 0
fe_ops = [BG92, T06_opt, P22, BV08]
init_fe_op_fwhms = [1200., 1200., 800., 600.]


# Normalisation and Rescaling #
normalization_switch = True


# Filepath with Lines #
lines_path = 'Lines/Lines_MgII.csv'

# Fit region and type #
N_fits = 1
sig_clip = False
fit_synthetic = False
line_shift = 0
allowed_shift = 30 # Allowed Central Wavelength Shift

line_fit_orig = [[1210, 1290], [1360, 1430], [1450, 1700], [1800, 1970]]
line_fit_small_lya = [[1216, 1290], [1360, 1430], [1450, 1700], [1800, 1970]]
line_fit_mod = [[1360, 1430], [1520, 1580], [1600, 1700], [1800, 1970]]
line_fit_temp = [[1220, 1290], [1360, 1430], [1450, 1700], [1800, 1970]]
#line_fit_no_NV = [[1360, 1430], [1450, 1700], [1800, 1970]]
line_fit_no_NV = [[1320, 1410], [1450, 1700], [1800, 1970]]

line_fit_civ_only = [[1450, 1570]]
line_fit_civ_R2020 = [[1500, 1570]] # modified
#line_fit_civ_R2020 = [[1500, 1537], [1549, 1600]]
#line_fit_civ_R2020 = [[1510, 1600]]
#line_fit_civ_R2020 = [[1500, 1600]] # Default
#line_fit_civ_R2020 = [[1530, 1545], [1550, 1570]]
#line_fit_civ_R2020 = [[1500, 1542], [1547, 1570]]
#line_fit_SiIV_CIV_HeII = [[1350, 1430], [1450, 1570], [1590, 1700]]
line_fit_SiIV_CIV_HeII = [[1370, 1430], [1460, 1510], [1510, 1570], [1630, 1680]]
#line_fit_SiIV_CIV_HeII = [[1350, 1430], [1450, 1537], [1547, 1570], [1630, 1700]]
#line_fit_SiIV_CIV_HeII = [[1360, 1389], [1400, 1410], [1419, 1430], [1480, 1536], [1544, 1590], [1630, 1700]]
#line_fit_SiIV_CIV_HeII = [[1350, 1430], [1450, 1550], [1555, 1580], [1630, 1700]]
#line_fit_SiIV_CIV_HeII = [[1380, 1425], [1450, 1700]]
#line_fit_SiIV_CIV_HeII = [[1350, 1430], [1450, 1700]]



line_fit_ciii = [[1800, 1970]]
line_fit_ciii_R2020 = [[1850, 1970]]

line_fit_J1144_1 = [[1850, 1970], [2750, 2850]]
line_fit_J1144_2 = [[4640, 5100], [6400, 6800]]
line_fit_J1144_3 = [[12500, 13200]]

#line_fit_Hbeta = [[4780, 4895]] # Orig
#line_fit_Hbeta = [[4780, 4905]] # wider version
#line_fit_Hbeta = [[4780, 4920]] # wide version (Default)
#line_fit_Hbeta = [[4780, 4940]] # widest version
line_fit_Hbeta = [[4780, 4880]] # narrower version
#line_fit_Hbeta = [[4720, 4860]] # narrow version
#line_fit_Hbeta = [[4780, 4870], [4877, 4920]] # HB89 0053-284


line_fit_MgII = [[2750., 2850.]]
#line_fit_MgII = [[2730., 2870.]] # Default
#line_fit_MgII = [[2710., 2910.]] # Wide
#line_fit_MgII = [[2710., 2930.]] # Wider
#line_fit_MgII = [[2750., 2870.]]
#line_fit_MgII = [[2750., 2815.]]
#line_fit_MgII = [[2770., 2830.]]
#line_fit_MgII = [[2785., 2820.]]
#line_fit_MgII = [[2800., 2880.]]
#line_fit_MgII_mod = [[2720, 2775], [2785, 2850]]
#line_fit_MgII = [[2730, 2807.], [2812., 2870.]]

line_fit_windows = line_fit_MgII
line_data_out = True # Write Data to file



########################
# Read Stacked Spectra #
########################

# Input data in rest-frame post-processed #
data_path = 'data/'
datafiles = glob.glob(data_path+'*.csv')


for i in range(N_fits):
	for datafile in datafiles:
		datafile = datafile[len(data_path):]
		obj_name = (datafile[:-4])
		print(datafile[:-4])

		pdata = pd.read_csv(data_path+datafile)
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

		if fit_synthetic:
			lams, flux, eflux = create_synthetic(lams, flux, eflux)

		if normalization_switch:
			norm_median = np.nanmedian(flux)
			flux = flux/norm_median
			eflux = eflux/norm_median


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
		conti_header = ['PL_Norm', 'PL_Slope', 
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
		fitobj = kmpfit.Fitter(residuals=conti_residuals, data=(lams, flux, eflux, used_conti_windows))
		fitobj.parinfo = tmp_parinfo
		fitobj.fit(params0=init_params)
		print("Best-fit parameters: ", fitobj.params)
		print("Iterations:               ", fitobj.niter)
		print("Reduced Chi^2:              ", fitobj.rchi2_min)
		#print("Covariance Errors:           ", fitobj.xerror)
		print("Std Errors:          ", fitobj.stderr)
		print()

		conti_bestfit = fitobj.params
		conti_stderrs = fitobj.stderr


		################
		# Line Fitting #
		################
		residual_flux = flux - eval_conti_all(conti_bestfit, lams)
		print('Fitting Lines...')
		print()

		pdata_lines = pd.read_csv(lines_path)
		line_names = pdata_lines['Name']
		line_wave = pdata_lines['Central Wavelength']
		line_norms = pdata_lines['Init_Norm']
		
		fwhm_lows = pdata_lines['FWHM_low']
		fwhm_highs = pdata_lines['FWHM_high']

		# SkewNorm Fitting
		init_params = []
		init_parinfo = []
		line_header = ['Skew', 'FWHM', 'Norm', 'Wavelength']

		for index, line in enumerate(line_names):
			# Line Parameters: Skew, Scale, Norm, Central Wavelength
			line_init_params = [0., fwhm_lows[index], line_norms[index], line_wave[index]+line_shift]
			line_parinfo = [{'fixed': True}, {'limits': (fwhm_lows[index], fwhm_highs[index])}, {'limits': (0., 1E2)}, {'limits': (line_wave[index]+line_shift-allowed_shift, line_wave[index]+line_shift+allowed_shift)}]
			init_params += line_init_params
			init_parinfo += line_parinfo

		xx, yy, e_yy = lams, residual_flux, eflux

		if sig_clip:
			mask = create_mask_window(xx, line_fit_windows)
			xx, yy, e_yy = np.array(xx)[mask], np.array(yy)[mask], np.array(e_yy)[mask]
			xx, yy, e_yy = sigma_mask_buffer(50, 3.0, xx, yy, e_yy, 3)

		fitobj = kmpfit.Fitter(residuals=residual_line, data=(xx, yy, e_yy, line_fit_windows))
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

		out_line_res(line_bestfit, line_stderrs, line_names)
















