# PyQSpecFit: Line modelling code for QSOs

Please use __[Tutorial.ipynb](https://github.com/samlaihei/PyQSpecFit/blob/main/Tutorial.ipynb)__ as a guide.

PyQSpecFit is a user-friendly and flexible code designed for modelling emission features in QSO spectra or continuum-subtracted spectra. 

The key features of this code are:
-  Fit the QSO continuum with a combined pseudo-continuum model composed of a power-law, Balmer continuum, and FeII flux. We include four main semi-empirical and empirical FeII templates.
-  Fit line complexes with a user-defined number of broad and narrow Gaussians profiles. 
-  Measure properties of the modelled lines (integrated flux, EW, FWHM, line dispersion, peak, wavelength shift, etc). Realistic uncertainties can be obtained by using the error spectrum to resample and re-fit the data.
-  Plot the data with the modelled continuum and emission-line profiles. 

## Cite this code

The preferred citations for this code are the following:

> @ARTICLE{2023MNRAS.526.3230L,\
>        author = {{Lai}, Samuel and {Onken}, Christopher A. and {Wolf}, Christian and {Bian}, Fuyan and {Cupani}, Guido and {Lopez}, Sebastian and {D'Odorico}, Valentina},\
>         title = "{Virial black hole mass estimates of quasars in the XQ-100 legacy survey}",\
>       journal = {\mnras},\
>      keywords = {galaxies: active, galaxies: high-redshift, quasars: emission lines, Astrophysics - Astrophysics of Galaxies},\
>          year = 2023,\
>         month = dec,\
>        volume = {526},\
>        number = {3},\
>         pages = {3230-3247},\
>           doi = {10.1093/mnras/stad2994},\
> archivePrefix = {arXiv},\
>        eprint = {2310.00271},\
>  primaryClass = {astro-ph.GA},\
>        adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3230L },\
>       adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }



> @software{samuel_lai_2023_7772752,\
>  author       = {Samuel Lai},\
>  title        = {samlaihei/PyQSpecFit: PyQSpecFit v1.0.0},\
>  month        = mar,\
>  year         = 2023,\
>  publisher    = {Zenodo},\
>  version      = {v1.0.0},\
>  doi          = {10.5281/zenodo.7772752},\
>  url          = {https://doi.org/10.5281/zenodo.7772752 }\
> }