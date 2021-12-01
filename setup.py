import sys
import setuptools
from TGLC.__init__ import __version__

sys.path.insert(0, "TGLC")
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="TGLC",
    version=__version__,
    author="Te Han",
    author_email="tehanhunter@gmail.com",
    description="TESS-Gaia Light Curve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TeHanHunter/TESS_Gaia_Light_Curve",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires= ['numpy', 'astropy', 'astroquery', 'matplotlib', 'pickle', 'tqdm', 'wotan'],
    packages=setuptools.find_packages(include=['TGLC', 'TGLC.*']),
    python_requires=">=3.6",
)
