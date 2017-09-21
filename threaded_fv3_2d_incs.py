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
from netCDF4 import Dataset
from netcdftime import utime
from datetime   import datetime,timedelta
import scipy
import ncepy
#Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')
import pdb

######################   USER DEFINED SETTINGS    ############################################
outputdir='/scratch4/NCEPDEV/meso/save/Donald.E.Lippi/FV3RDA/graphics_utils/'# output directory
gesdir='/scratch4/NCEPDEV/meso/save/Donald.E.Lippi/data'  # directory containing your fv3nc ges files
anldir='/scratch4/NCEPDEV/meso/save/Donald.E.Lippi/data'  # directory containing your fv3nc anl files
startplot=datetime.strptime('2017080700','%Y%m%d%H')      # valid time
nesteddata_ges = os.path.join(gesdir,'nggps2d.nest02.nc')    # name of guess file
nesteddata_anl = os.path.join(anldir,'nggps2d.nest02.nc')    # name of analysis file
nestedgrid = os.path.join(gesdir,'grid_spec.nest02.nc')      # name of grid spec file
dom="CONUS"                                               # domain (can be CONUS, SC, etc.)
proj="gnom"                                               # map projection
varnames=[                                                # uncomment the desired variables below
#          'ALBDOsfc',\
#          'CPRATsfc',\
          'PRATEsfc',\
#          'DLWRFsfc',\
#          'ULWRFsfc',\
#          'DSWRFsfc',\
#          'USWRFsfc',\
#          'DSWRFtoa',\
#          'USWRFtoa',\
#          'ULWRFtoa',\
#          'GFLUXsfc',\
#          'HGTsfc',\
#          'HPBLsfc',\
#          'ICECsfc',\
#          'SLMSKsfc',\
#          'LHTFLsfc',\
#          'SHTFLsfc',\
          'PRESsfc',\
          'PWATclm',\
#          'SOILM',\
#          'SOILW1',\
#          'SOILW2',\
#          'SOILW3',\
#          'SOILW4',\
#          'SPFH2m',\
#          'SOILT1',\
#          'SOILT2',\
#          'SOILT3',\
#          'SOILT4',\
#          'TMP2m',\
#          'TMPsfc',\
#          'UGWDsfc',\
#          'VGWDsfc',\
#          'UFLXsfc',\
#          'VFLXsfc',\
#          'UGRD10m',\
#          'VGRD10m',\
#          'WEASDsfc',\
#          'SNODsfc',\
#          'ZORLsfc',\
#          'VFRACsfc',\
#          'F10Msfc',\
#          'VTYPEsfc',\
#          'STYPEsfc',\
#          'TCDCclm',\
#          'TCDChcl',\
#          'TCDCmcl',\
#          'TCDClcl',\
         ]
######################   USER DEFINED SETTINGS    ############################################

# Create the basemap
# create figure and axes instances
fig = plt.figure(figsize=(11,11))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

# Setup map corners for plotting.  This will give us CONUS
llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=ncepy.corners_res(dom,proj=proj)
m = Basemap(llcrnrlon=llcrnrlon,   llcrnrlat=llcrnrlat,
               urcrnrlon=urcrnrlon,  urcrnrlat=urcrnrlat,
               projection=proj, lat_0=35.4,lon_0=-97.6,
               resolution=res)

# Map background stuff to make things look nice
parallels = np.arange(-80.,90,10.)
meridians = np.arange(0.,360.,10.)
m.drawcoastlines(linewidth=1.25)
m.drawstates(linewidth=1.25)
m.drawcountries(linewidth=1.25)
m.drawparallels(parallels,labels=[1,0,0,1])
m.drawmeridians(meridians,labels=[1,0,0,1])
#m.drawcounties(linewidth=0.2, color='k')

def mkplot(varname):
    print("mkplot - "+str(multiprocessing.current_process()))
    fndges = Dataset(nesteddata_ges,'r')
    fndanl = Dataset(nesteddata_anl,'r')
    fng   = Dataset(nestedgrid,'r')
    varnames2d=fndges.variables.keys()

    # Get the map navigation info from the the grid spec file
    # The variables 'grid_lon' and 'grid_lat' refer to the coordinates of the grid corners, which define the extents of the grid cell.
    # The cell centroids, or 'T-cells' are defined in 'grid_lont' and 'grid_latt'.
    #   Which set of coordinates you want to use depend on the sort of plot you want to make:
    #   a contour plot should use grid_lont and grid_latt, the cell centroids, but a color fill plot, like pcolor or grfill,
    #   should use the grid corners in grid_lon and grid_lat.

    # The natural definition for the grid, defined by the npx and npy variables in the input namelist,
    # is the number of grid corners per grid face. The number of cells in a grid face is then npx-1 and npy-1.
    # So a c768 grid will have npx = npy = 769.

    # The winds in the history files are re-gridded to cell centers and then rotated to be the zonal and meridional winds.
    #  You do not need to do any rotation to plot the grid vectors.

    #grid_lon_n  = fng.variables['grid_lon'][:,:]
    #grid_lat_n  = fng.variables['grid_lat'][:,:]

    grid_lont_n  = fng.variables['grid_lont'][:,:]
    grid_latt_n  = fng.variables['grid_latt'][:,:]
    global lons,lats
    lons=grid_lont_n; lats=grid_latt_n
    lons[lons>180]-=360 # grid_lont_n is in units of 0-360 deg. We need -180 to 180 for maskoceans.

    times = fndges.variables['time'][:]
    cdftime = utime(getattr(fndges.variables['time'],'units'))
    # Grab the cycledate
    cycledate=roundTime(cdftime.num2date(times[0]),roundTo=60.*60.)

    #  Map/figure has been set up here (bulk of the work), save axes instances for
    #     use again later
    keep_ax_lst = ax.get_children()[:]

    # Transform lats and lons to map proj coords
    #x_n,y_n = m(grid_lon_n[:,:],grid_lat_n[:,:])
    xt_n,yt_n = m(grid_lont_n[:,:],grid_latt_n[:,:])

    cycledate=roundTime(cdftime.num2date(times[0]),roundTo=60.*60.)
    dispatcher=plot_Dictionary()
    for t,time in enumerate(times):
        # Just to get a nicely formatted date for writing out
        datestr =roundTime(cdftime.num2date(time),roundTo=60.*60.)
        outdate=datestr.strftime('%Y%m%d%H')
        diff=(datestr-cycledate)
        fhr=int(diff.days*24.+diff.seconds/3600.)
        if datestr==startplot:
            # Clear off old plottables but keep all the map info
            ncepy.clear_plotables(ax,keep_ax_lst,fig)
            var_nges=fndges.variables[varname][t,:,:]
            var_nanl=fndanl.variables[varname][t,:,:]
            var_n=var_nanl-var_nges
            try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
               print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
               function=dispatcher[varname]
               var_n,clevs,cm,units,longname=function(var_n)
            except KeyError:
               raise ValueError("invalid varname:"+varname)
            cs = m.pcolormesh(xt_n,yt_n,var_n,cmap=cm,vmin=clevs[0],vmax=clevs[-1])
            cbar = m.colorbar(cs,location='bottom',pad="5%",extend="both",ticks=clevs)
            cbar.ax.tick_params(labelsize=8.5)
            cbar.set_label(varname+": "+longname+" increments  ["+str(units)+"]")
            plt.title(outdate+"    FHR "+str(fhr).zfill(3),loc='left',fontsize=10,)
            plt.savefig(outputdir+ varname + '_%03d_incs.png' % (fhr),dpi=125, bbox_inches='tight')

    plt.close('all')


############### useful functions ###########################################
def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time laps in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)

def gemplot(clist):
    gemlist=ncepy.gem_color_list()
    colors=[gemlist[i] for i in clist]
    cm = matplotlib.colors.ListedColormap(colors)
    return cm

############### plot_ functions ###########################################
def plot_ALBDOsfc(var_n):
    """surface albedo (%)"""
    longname="surface albedo"; units="%"
    clevs=np.arange(0.,100.5,5.)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_CPRATsfc(var_n):
    """surface convective precipitation rate [kg/m**2/s]"""
    longname="surface convective precipitation rate"; units="in."
    if(units=="in."): 
       var_n = var_n*3*3600/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    elif(units=="mm"):
       var_n = var_n*3*3600      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)


def plot_PRATEsfc(var_n):
    """surface precipitation rate [kg/m**2/s]"""
    longname="surface precipitation rate"; units="in."
    if(units=="in."):
       var_n = var_n*3*3600/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    elif(units=="mm"):
       var_n = var_n*3*3600      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_DLWRFsfc(var_n):
    """surface downward longwave flux [W/m**2]"""
    longname="surface downward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_ULWRFsfc(var_n):
    """surface upward longwave flux [W/m**2]"""
    longname="surface upward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_DSWRFsfc(var_n):
    """surface downward shortwave flux [W/m**2]"""
    longname="surface downward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_USWRFsfc(var_n):
    """surface upward shortwave flux [W/m**2]"""
    longname="surface upward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_DSWRFtoa(var_n):
    """top of atmos downward shortwave flux [W/m**2]"""
    longname="top of atmos downward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_USWRFtoa(var_n):
    """top of atmos upward shortwave flux [W/m**2]"""
    longname="top of atmos upward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_ULWRFtoa(var_n):
    """top of atmos upward longwave flux [W/m**2]"""
    longname="top of atmos upward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    #clevs=np.arange(0,15000.5,1000)
    return(var_n,clevs,cm,units,longname)

def plot_GFLUXsfc(var_n):
    """surface ground heat flux [W/m**2]"""
    longname="surface ground heat flux"; units="W/m**2"
    clevs= [-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,0.,10.0,25.,50.,75.,100.,200.,300.]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_HGTsfc(var_n):
    """surface geopotential height [gpm]"""
    longname="surface geopotential height"; units="gpm"
    clevs=[0,250.,500.,750.,1000.,1500.,2000.,3000.,4000.,5000.,7500.,10000.,15000.,20000.,25000.,30000.,30000.5]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_HPBLsfc(var_n):
    """surface planetary boundary layer height [m]"""
    longname="surface planetary boundary layer height"; units="m"
    clevs=[0,50.,100.,150.,200.,250.,500.,750.,1000.,1500.,2000.,3000.,4000.,5000.,7500.]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_ICECsfc(var_n):
    """surface ice concentration (ice=1; no ice=0) [fraction]"""
    longname="surface ice concentration"; units="(ice=1; no ice=0)"
    clevs=[0,0.5,1]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    clist=[23,27]
    return(var_n,clevs,cm,units,longname)

def plot_SLMSKsfc(var_n):
    """sea-land-ice mask (0-sea, 1-land, 2-ice)"""
    longname="sea-land-ice mask"; units="0-sea, 1-land, 2-ice"
    clevs=[0,1,2,2.01]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    clist=[24,18,27]
    return(var_n,clevs,cm,units,longname) 

def plot_LHTFLsfc(var_n):
    """surface latent heat flux [W/m**2]"""
    longname="surface latent heat flux"; units="W/m**2"
    clevs=[-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,-5.,5.,10.0,25.,50.,75.,100.,200.,300]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SHTFLsfc(var_n):
    """surface sensible heat flux [W/m**2]"""
    longname="surface sensible heat flux"; units="W/m**2"
    clevs=[-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,0.,10.0,25.,50.,75.,100.,200.,300.]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_PRESsfc(var_n): # done
    """surface pressure [Pa]"""
    longname="surface pressure"; units="Pa"
    var_n=var_n*0.01
    var_n=scipy.ndimage.gaussian_filter(var_n, 2) # first pass
    var_n=scipy.ndimage.gaussian_filter(var_n, 2) # second pass
    clevs=np.arange(950.,1050.,4.)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_PWATclm(var_n):
    """atmos column precipitable water [kg/m**2]"""
    longname="atmos column precipitable water"; units="mm"
    clevs= [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    clist = [0,16,18,30,28,27,25,4,23,3,21,8,5,19,17,31,12,2,7,14]
    return(var_n,clevs,cm,units,longname)

def plot_SOILM(var_n):
    """total column soil moisture content [kg/m**2]"""
    longname="total column soil moisture content"; units="in."
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="in."): var_n = var_n/25.4 #inches
    clevs= [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    clist = [0,16,18,30,28,27,25,4,23,3,21,8,5,19,17,31,12,2,7,14]
    return(var_n,clevs,cm,units,longname)

def plot_SOILW1(var_n):
    """volumetric soil moisture 0-10cm [fraction]"""
    longname="volumetric soil moisture 0-10cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILW2(var_n):
    """volumetric soil moisture 10-40cm [fraction]"""
    longname="volumetric soil moisture 10-40cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILW3(var_n):
    """volumetric soil moisture 40-100cm [fraction]"""
    longname="volumetric soil moisture 40-100cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILW4(var_n):
    """volumetric soil moisture 100-200cm [fraction]"""
    longname="volumetric soil moisture 100-200cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SPFH2m(var_n):
    """2m specific humidity [kg/kg]"""
    longname="2m specific humidity"; units="g/kg" #units="kg/kg *10-3"
    var_n=var_n*1000
    clevs=np.arange(0.,32.5,1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILT1(var_n): #done
    """soil temperature 0-10cm [K]"""
    longname="soil temperature 0-10cm"; units="C"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILT2(var_n): #done
    """soil temperature 10-40cm [K]"""
    longname="soil temperature 10-40cm"; units="C"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILT3(var_n): #done
    """soil temperature 40-100cm [K]"""
    longname="soil temperature 40-100cm"; units="C"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SOILT4(var_n): #done
    """soil temperature 100-200cm [K]"""
    longname="soil temperature 100-200cm"; units="C"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TMP2m(var_n):
    """2m temperature [K]"""
    longname="2m temperature"; units="C"
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TMPsfc(var_n):
    """surface temperature [K]"""
    longname="surface temperature"; units="C"
    clevs= np.arange(-36.,104.,4)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_UGWDsfc(var_n): 
    """surface zonal gravity wave stress [N/m**2]"""
    longname="surface zonal gravity wave stress"; units="N/m**2"
    clevs=[-5,-2.5,-1,-0.05,-0.01,0.01,0.05,1,2.5,5]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VGWDsfc(var_n): 
    """surface meridional gravity wave stress [N/m**2]"""
    longname="surface meridional gravity wave stress"; units="N/m**2"
    clevs=[-5,-2.5,-1,-0.05,-0.01,0.01,0.05,1,2.5,5]
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_UFLXsfc(var_n): 
    """surface zonal momentum flux [N/m**2]"""
    longname="surface zonal momentum flux"; units="N/m**2"
    clevs=np.arange(-1,1.05,0.1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VFLXsfc(var_n): 
    """surface meridional momentum flux [N/m**2]"""
    longname="surface meridional momentum flux"; units="N/m**2"
    clevs=np.arange(-1,1.05,0.1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_UGRD10m(var_n): 
    """10 meter u wind [m/s]"""
    longname="10 meter u wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VGRD10m(var_n): 
    """10 meter v wind [m/s]"""
    longname="10 meter v wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_WEASDsfc(var_n): 
    """surface snow water equivalent [kg/m**2]"""
    longname="surface snow water equivalent"; units="in."
    if(units=="in."):
       var_n = var_n/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    elif(units=="mm"):
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SNODsfc(var_n): 
    """surface snow depth [m]"""
    longname="surface snow depth"; units="in."
    if(units=="in."):
       var_n = var_n/0.0254  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    elif(units=="mm"):
       var_n = var_n/1000.      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
       clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_ZORLsfc(var_n):
    """surface roughness [m]"""
    longname="surface roughness"; units="m"
    clevs=np.arange(0,3.1,0.1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VFRACsfc(var_n):
    """vegetation fraction"""
    longname="vegetation fraction"; units="fraction"
    clevs=np.arange(0.,100.5,5.)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_F10Msfc(var_n): 
    """10-meter wind speed divided by lowest model wind speed"""
    longname="10-meter wind speed divided by lowest model wind speed"; units="none"
    clevs=np.arange(0,2.05,.1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VTYPEsfc(var_n): 
    """vegetation type in integer 1-13"""
    longname="vegetation type in integer 1-13"; units="1-13"
    clevs=np.arange(1,15,1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_STYPEsfc(var_n): 
    """soil type in integer 1-9"""
    longname="soil type"; units="1-9"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=np.arange(1,11,1)
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TCDCclm(var_n):
    """atmos column total cloud cover [%]"""
    longname="atmos column total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] # percent
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TCDChcl(var_n):
    """high cloud level total cloud cover [%]"""
    longname="high cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] #percent
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TCDCmcl(var_n): 
    """mid cloud level total cloud cover [%]"""
    longname="mid cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] #percent
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_TCDClcl(var_n): 
    """low cloud level total cloud cover [%]"""
    longname="low cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] # percent
    clevs=np.arange(-0.5*clevs[-1],0.5*clevs[-1]+.5,0.1*clevs[-1])
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname) 

############### Dictionary for plot_function calls ###################################
def plot_Dictionary():
    #As fields are added to fv3 output just put those in the following dictionary
    #   according to the syntax used. Then all you have to do is create a function
    #   that defines the clevs, cm, and var_n if it requires unit conversion 
    #   (e.g., plot_PRATEsfc(var_n) )
    """The purpose of this dictionary is so that for each variable name (e.g., "ALBDOsfc") 
       the corresponding function is called (e.g., plot_ALBDOsfc(var_n)) to provide the 
       appropriate variable specific name, units, clevs, clist, and colormap for plotting.
    """
    dispatcher={  
        'ALBDOsfc':plot_ALBDOsfc,
        'CPRATsfc':plot_CPRATsfc,
        'PRATEsfc':plot_PRATEsfc,
        'DLWRFsfc':plot_DLWRFsfc,
        'ULWRFsfc':plot_ULWRFsfc,
        'DSWRFsfc':plot_DSWRFsfc,
        'USWRFsfc':plot_USWRFsfc,
        'DSWRFtoa':plot_DSWRFtoa,
        'USWRFtoa':plot_USWRFtoa,
        'ULWRFtoa':plot_ULWRFtoa,
        'GFLUXsfc':plot_GFLUXsfc,
        'HGTsfc':plot_HGTsfc,
        'HPBLsfc':plot_HPBLsfc,
        'ICECsfc':plot_ICECsfc,
        'SLMSKsfc':plot_SLMSKsfc,
        'LHTFLsfc':plot_LHTFLsfc,
        'SHTFLsfc':plot_SHTFLsfc,
        'PRESsfc':plot_PRESsfc,
        'PWATclm':plot_PWATclm,
        'SOILM':plot_SOILM,
        'SOILW1':plot_SOILW1,
        'SOILW2':plot_SOILW2,
        'SOILW3':plot_SOILW3,
        'SOILW4':plot_SOILW4,
        'SPFH2m':plot_SPFH2m,
        'SOILT1':plot_SOILT1,
        'SOILT2':plot_SOILT2,
        'SOILT3':plot_SOILT3,
        'SOILT4':plot_SOILT4,
        'TMP2m':plot_TMP2m,
        'TMPsfc':plot_TMPsfc,
        'UGWDsfc':plot_UGWDsfc,
        'VGWDsfc':plot_VGWDsfc,
        'UFLXsfc':plot_UFLXsfc,
        'VFLXsfc':plot_VFLXsfc,
        'UGRD10m':plot_UGRD10m,
        'VGRD10m':plot_VGRD10m,
        'WEASDsfc':plot_WEASDsfc,
        'SNODsfc':plot_SNODsfc,
        'ZORLsfc':plot_ZORLsfc,
        'VFRACsfc':plot_VFRACsfc,
        'F10Msfc':plot_F10Msfc,
        'VTYPEsfc':plot_VTYPEsfc,
        'STYPEsfc':plot_STYPEsfc,
        'TCDCclm':plot_TCDCclm,
        'TCDChcl':plot_TCDChcl,
        'TCDCmcl':plot_TCDCmcl,
        'TCDClcl':plot_TCDClcl,
               }
    return dispatcher  

if __name__ == '__main__':
    pool=multiprocessing.Pool(len(varnames)) # one processor per variable
    #pool=multiprocessing.Pool(8) # 8 processors for all variables. Just a little slower.
    pool.map(mkplot,varnames) 
    toc=timer()
    time=toc-tic
    hrs=int(time/3600)
    mins=int(time%3600/60)
    secs=int(time%3600%60)
    print("Total elapsed time: "+str(toc-tic)+" seconds.")
    print("Total elapsed time: "+str(hrs).zfill(2)+":"+str(mins).zfill(2)+":"+str(secs).zfill(2))





