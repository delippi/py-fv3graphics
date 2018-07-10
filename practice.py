#!/usr/bin/env python
#PBS -N fv3py
#PBS -l walltime=0:05:00
#PBS -l nodes=1:ppn=8
#PBS -q debug
#PBS -A fv3-cpu
#PBS -o fv3py.out
#PBS -j oe

from timeit import default_timer as timer
tic=timer()
import sys,os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap, cm, maskoceans
import multiprocessing
import numpy as np
import math
#import pynemsio
#import pygrib
from netCDF4 import Dataset
from netcdftime import utime
from datetime   import datetime,timedelta
import scipy, ncepy
#Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')
import pdb
import subprocess

infile='/gpfs/hps2/stmp/Donald.E.Lippi/RUNDIRS/fv3gfs_dl2rw_DAS_exp_001_2018050218/2018050300/gfs/anal/atmanl_mem001'
fh=Dataset(infile,mode='r')
lons=fh.variables['lon'][:]
lats=fh.variables['lat'][:]
tmax=fh.variables['Tmax'][:]
tmax_units=fh.variables['Tmax'].units
fh.close
# Get some parameters for the Stereographic Projection
lon_0 = lons.mean()
lat_0 = lats.mean()

m = Basemap(width=5000000,height=3500000,
            resolution='l',projection='stere',\
            lat_ts=40,lat_0=lat_0,lon_0=lon_0)

# Because our lon and lat variables are 1D,
# use meshgrid to create 2D arrays
# Not necessary if coordinates are already in 2D arrays.
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

# Plot Data
cs = m.pcolor(xi,yi,np.squeeze(tmax))

# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(tmax_units)

# Add Title
plt.title('DJF Maximum Temperature')

plt.show()








