import sys
import setuptools

sys.path.insert(0, "tglc")
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="tglc",
    version='0.2.2',
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
    install_requires=['astropy', 'astroquery', 'matplotlib', 'numpy', 'oauthlib', 'requests', 'scipy', 'tqdm', 'wheel', 'wotan'],
    packages=setuptools.find_packages(include=['tglc', 'tglc.*']),
    python_requires=">=3.6",
    include_package_data=True
)
