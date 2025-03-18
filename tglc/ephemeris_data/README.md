# TESS Ephemeris Files

This directory contains files with TESS ephemeris data used for barycentric time correction by QLP. The files are CSV files with X,Y,Z coordinates in AU from the solar system barycenter and the reference frame is ICRF.

Each file has a set of sectors it is meant to be used for:

- Sectors 1-5: [20180720_tess_ephem.csv](20180720_tess_ephem.csv)
- Sectors 6-19: [20190101_tess_ephem.csv](20190101_tess_ephem.csv)
- Sectors 20-32: [20200101_tess_ephem.csv](20200101_tess_ephem.csv)
- Sectors 33-45: [20210101_tess_ephem.csv](20210101_tess_ephem.csv)
- Sectors 46-59: [20211215_tess_ephem.csv](20211215_tess_ephem.csv)
- Sectors 60-73: [20221201_tess_ephem.csv](20221201_tess_ephem.csv)
- Sectors 74-87: [20231201_tess_ephem.csv](20231201_tess_ephem.csv)
- Sectors 88-101: [20241201_tess_ephem.csv](20241201_tess_ephem.csv)

## Downloading the files

At the start of every year, we need to add a TESS predicted ephemeris file so we can correct spacecraft time to BJD TDB.These ephemerides are downloaded from the horizons website: <https://ssd.jpl.nasa.gov/horizons.cgi>

**Ephemeris Type:** Vector Table

**Target Body:** TESS

**Coordinate Origin:** Solar System Barycenter (SSB)

**Time Span:** The required time space for the year. We want the step size to be at least one hour

**Table Settings:**

- Quantities code = 4 (position, LT, range, and range-rate)
- Reference frame = ICRF
- Reference plane = x-y axes of reference frame
- Vector correction = geometric states
- Output units = AU and days
- CSV format = YES
- Object summary = up to you. The summary will need to be commented out anyways, so I prefer to not include it.

Make sure you comment out the header and footer information in the csv file. Also, when last generated (Jan 15 2023) I needed to add the following line before the values:

```
JDTDB,Calendar Date (TDB),X,Y,Z,LT,RG,RR,
```

(see the format of the other .csv files in the directory for guidance)

After adding the file to this directory, plot all the ephemerides by running [plot_ephemerides.py](plot_ephemerides.py). This ensures the files can all be read successfully. Examine the plot it produces and check that the transition between years is smooth, and that the new file continues the existing sine wave patterns.

If you have a function that specifies where the ephemeris file is for each sector (like in the [notebook in this repository](/TESS%20Time%20Correction.ipynb)), make sure to update that as well.
