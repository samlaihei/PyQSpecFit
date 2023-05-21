# PyQSpecFit: Line modelling code for QSOs

Please use tutorial.ipynb as a guide.

PyQSpecFit is a user-friendly and flexible code designed for modelling emission features in QSO spectra or continuum-subtracted spectra. 

The key features of this code are:
-  Fit the QSO continuum with a combined pseudo-continuum model composed of a power-law, Balmer continuum, and FeII flux. We include four main semi-empirical and empirical FeII templates.
-  Fit line complexes with a user-defined number of broad and narrow Gaussians profiles. 
-  Measure properties of the modelled lines (integrated flux, EW, FWHM, line dispersion, peak, wavelength shift, etc). Realistic uncertainties can be obtained by using the error spectrum to resample and re-fit the data.
-  Plot the data with the modelled continuum and emission-line profiles. 

## Cite this code

The preferred citations for this code are the following:

> @ARTICLE{2023MNRAS.521.3682L,\
>       author = {{Lai (赖民希)}, Samuel and {Wolf}, Christian and {Onken}, Christopher A. and {Bian (边福彦)}, Fuyan},\
>        title = "{Characterising SMSS J2157-3602, the most luminous known quasar, with accretion disc models}",\
>      journal = {\mnras},\
>     keywords = {galaxies: active, galaxies: high-redshift, quasars: emission lines, Astrophysics - Astrophysics of Galaxies},\
>         year = 2023,\
>        month = may,\
>       volume = {521},\
>       number = {3},\
>        pages = {3682-3698},\
>          doi = {10.1093/mnras/stad651},\
>      archivePrefix = {arXiv},\
>       eprint = {2302.10397},\
>      primaryClass = {astro-ph.GA},\
>       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.3682L},\
>      adsnote = {Provided by the SAO/NASA Astrophysics Data System}\
> }

> @software{samuel_lai_2023_7772752,\
>  author       = {Samuel Lai},\
>  title        = {samlaihei/PyQSpecFit: PyQSpecFit v1.0.0},\
>  month        = mar,\
>  year         = 2023,\
>  publisher    = {Zenodo},\
>  version      = {v1.0.0},\
>  doi          = {10.5281/zenodo.7772752},\
>  url          = {https://doi.org/10.5281/zenodo.7772752}\
> }