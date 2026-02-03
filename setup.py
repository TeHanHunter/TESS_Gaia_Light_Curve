import sys
import setuptools

sys.path.insert(0, "tglc")
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="tglc",
    version='0.6.7',
    author="Te Han",
    author_email="tehanhunter@gmail.com",
    description="TESS-Gaia Light Curve",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/TeHanHunter/TESS_Gaia_Light_Curve",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'astropy>=6.1,<6.2',
        'astroquery==0.4.7',
        'matplotlib>=3.8,<4.0',
        'numpy>=1.23,<2.0',
        'oauthlib',
        'requests>=2.28,<3.0',
        'scipy>=1.10,<2.0',
        'threadpoolctl>=3.1,<4.0',
        'tqdm>=4.64',
        'wotan~=1.9',
        'seaborn',
        'pandas',
        'importlib_resources',
    ],
    packages=setuptools.find_packages(include=['tglc', 'tglc.*']),
    python_requires=">=3.8, <3.13",
    include_package_data=True
)
