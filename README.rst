.. image:: logo/TGLC_Title.png
  :width: 800
  :alt: TESS-Gaia Light Curve

==================================
Introduction
==================================

TESS-Gaia Light Curve (TGLC) is a dataset of TESS full-frame image light curves publicly available via the MAST portal. It is fitted with effective PSF and decontaminated with Gaia DR2 and achieved percent-level photometric precision down to 16th magnitude! It unlocks astrophysics to a vast number of dim stars below 12th magnitude.

.. image:: logo/EB_comparison_git.png
  :width: 800
  :alt: EB light curve comparison to other pipeline

==================================
Usage
==================================

Download from MAST
----------------
The easiest usage requires no package installation. Simply follow (coming soon) to download light curves from MAST. Light curves are being fitted sector by sector and will be available on MAST gradually. MAST hosts all Gaia DR2 stars down to 16th magnitude. Each .fits file includes PSF and aperture light curves and their calibrated versions. 

Available sectors: 

Fit from scratch
----------------
Users can also fit light curves using the package tglc. Using tglc, one can specify a region, sector(s), and customized aperture shape if needed. To install 
