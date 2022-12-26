.. image:: logo/TGLC_Title.png
  :width: 800
  :alt: TESS-Gaia Light Curve
.. image:: https://zenodo.org/badge/420868490.svg
   :target: https://zenodo.org/badge/latestdoi/420868490
==================================
Introduction
==================================

TESS-Gaia Light Curve (TGLC) is a dataset of TESS full-frame image light curves publicly available via the MAST portal. It is fitted with effective PSF and decontaminated with Gaia DR3 and achieved percent-level photometric precision down to 16th TESS magnitude! It unlocks astrophysics to a vast number of dim stars below 12th TESS magnitude.

.. image:: logo/EB_comparison_git.png
  :width: 800
  :alt: EB light curve comparison to other pipeline

==================================
Usage
==================================

There are three data access methods:
•MAST Portal: Easiest for acquiring light curves for a few stars. However, new sectors are updated relatively slowly. 
•MAST bulk download: Best for downloading light curves for all stars (<16 TESS magnitude) in a sectors. 
•tglc package: Capable of producing similar quality light curves for any sector and any star with custom options. 

MAST availabilities are maintained `here <https://docs.google.com/spreadsheets/d/1FhHElWb1wmx9asWiZecAJ2umN0-P_aXn55OBVB34_rg/edit?usp=sharing>`_. 

Download from MAST
----------------
The easiest usage requires no package installation. Simply follow the `tutorial <tutorial/TGLC_tutorial.ipynb>`_ to download light curves from MAST. Light curves are being fitted sector by sector and will be available on MAST gradually. MAST hosts all Gaia DR3 stars down to 16th magnitude. Each .fits file includes PSF and aperture light curves and their calibrated versions. 

Available sectors: `sector worklist <https://docs.google.com/spreadsheets/d/1FhHElWb1wmx9asWiZecAJ2umN0-P_aXn55OBVB34_rg/edit?usp=sharing>`_


Fit from scratch
----------------
Users can also fit light curves using the package tglc. Using tglc, one can specify a region, sector(s), and customized aperture shape if needed. tglc is currently only available for linux. Run 

``pip install tglc``

for the latest tglc release. After installation, follow the `tutorial <tutorial/TGLC_tutorial.ipynb>`_ to fit light curves. If there is a problem, please leave a comment in the Issues section to help us improve. Thank you!
