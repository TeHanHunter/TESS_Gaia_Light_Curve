import sys
import setuptools
from SEBIT.version import __version__

sys.path.insert(0, "SEBIT")
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="SEBIT",
    version=__version__,
    author="Te Han",
    author_email="tehanhunter@gmail.com",
    description="Searching Eclipsing Binaries in TESS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TeHanHunter/Searching-Eclipsing-Binaries-in-TESS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['SEBIT', 'SEBIT.*']),
    python_requires=">=3.6",
)
