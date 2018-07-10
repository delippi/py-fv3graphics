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
import scipy
import ncepy
#Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')
import pdb
import subprocess
import os

def get_PDY_HH(date):
    ps=subprocess.Popen(('echo',date),stdout=subprocess.PIPE)
    PDY=subprocess.check_output(('cut','-c','1-8'),stdin=ps.stdout).strip()
    ps=subprocess.Popen(('echo',date),stdout=subprocess.PIPE)
    HH=subprocess.check_output(('cut','-c','9-10'),stdin=ps.stdout).strip()
    return PDY,HH

######################   USER DEFINED SETTINGS    ############################################
date='2018050300'; PDY,HH=get_PDY_HH(date)
tmpdir='stmp'
exp='DAS_exp_001_'+date
#exp='NATURE'

outputdir='/gpfs/hps3/emc/meso/noscrub/Donald.E.Lippi/py-fv3graphics/figs/'# output directory
#datadir='/gpfs/hps2/'+tmpdir+'/Donald.E.Lippi/fv3gfs_dl2rw_'+exp+'/gfs.'+PDY+'/'+HH
datadir='/gpfs/hps2/stmp/Donald.E.Lippi/RUNDIRS/fv3gfs_dl2rw_DAS_exp_001_2018050218/2018050300/gfs/anal'
#datadir='/gpfs/hps2/ptmp/Donald.E.Lippi/FV3ICS/'+date+'/C768/control/INPUT/'
startplot=datetime.strptime(date,'%Y%m%d%H')   # start time
nhrs=6                                                 # number of plots to generate plus one for the start time.
endplot=startplot+timedelta(hours=nhrs)                # end time. 
levs=[50]                                              # FV3 model level if data is 3D, otherwise ignored.
                                                       # 1=model top; 63=model bottom.

                                                       # levs=['all'] will plot all levels.
#nesteddata = os.path.join(datadir,'refl2D.nest02.nc') # name of file
#nesteddata = os.path.join(datadir,'gfs_data.tile1.nc') # name of file
#nestedgrid = os.path.join(datadir,'gfs_ctrl.nc')       # name of file
proj="gnom"                                            # map projection
dom="SC"                                               # domain (CONUS,NW,NWRFC,NC,NE,SC,SE,SW,MIDATL
                                                       # Great_Lakes,AK,NAK,SAK,SWAK,SEAK,PR,GUAM,HI,
                                                       # POWER,OK,LAKE_VICTORIA,AFRICA,MEDFORD))
varnames=[                                             # uncomment the desired variables below
          'REFC',\
#          'MSL',\
#          'SRHL0_1km_max',\
#          'SRHL0_3km_max',\
#          'VUSHR0-6000',\
#          'VVSHR0-6000',\
#          'CAPEsfc',\
#          'CAPE18000-0',\
#          'CAPE9000-0',\
#          'CAPE25500-0',\
#          'MXUPHL2_5km_max',\
#          'MNUPHL2_5km_min',\
#          'MAXREFC_max',\
#          'MAXREF_1km_max',\

         ]
######################   END OF USER DEFINED SETTINGS    ########################################
#nesteddata = os.path.join(datadir,'gfs.t00z.master.grb2f'+str(nhrs).zfill(3))
#nesteddata = os.path.join(datadir,'gfs.t00z.master.grb2f003')
nesteddata = os.path.join(datadir,'atmanl_mem001')
print(nesteddata)
#nesteddata = os.path.join(datadir,'gfs.t00z.atmf006.nemsio'); file_type='NEMSIO'

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

def NETCDF_read(nesteddata):
    fnd = Dataset(nesteddata,'r')
    fng   = Dataset(nestedgrid,'r')
    #varnames2d=fnd.variables.keys()
    grid_lont_n  = fng.variables['grid_lont'][:,:]
    grid_latt_n  = fng.variables['grid_latt'][:,:]
    global lons,lats
    lons=grid_lont_n; lats=grid_latt_n
    lons[lons>180]-=360 # grid_lont_n is in units of 0-360 deg. We need -180 to 180 for maskoceans.
    try:
       times = fnd.variables['time'][:]
       cdftime = utime(getattr(fnd.variables['time'],'units'))
       cycledate=roundTime(cdftime.num2date(times[0]),roundTo=60.*60.)
    except KeyError:
       times = fnd.variables['Time'][:]
       #cycledate = startplot
       print("How to get time from fv_tracer.res.nest02.*nc and fv_core.res.nest02.*nc???")
       exit("Not able to plot from these files yet.")
    keep_ax_lst = ax.get_children()[:]
    # Transform lats and lons to map proj coords
    #x_n,y_n = m(grid_lon_n[:,:],grid_lat_n[:,:])
    xt_n,yt_n = m(grid_lont_n[:,:],grid_latt_n[:,:])

    


def mkplot(varname):
    print("mkplot - "+str(multiprocessing.current_process()))
    #fnd = pynemsio.nemsfile(nesteddata)
    fnd = NETCDF_read(nesteddata)

    pdb.set_trace()

    dispatcher=plot_Dictionary()
    if(levs[0]==63 or levs[0]=='all'): levels=np.arange(63,0,-1)
    else: levels=levs
    break_levlevels=False
    for lev in levels:
        for t,time in enumerate(times):
            # Just to get a nicely formatted date for writing out
            try: datestr =roundTime(cdftime.num2date(time),roundTo=60.*60.)
            except UnboundLocalError: datestr = cycledate
            outdate=datestr.strftime('%Y%m%d%H')
            diff=(datestr-cycledate)
            fhr=int(diff.days*24.+diff.seconds/3600.)
            if datestr>=startplot and datestr<=endplot:
                # Clear off old plottables but keep all the map info
                ncepy.clear_plotables(ax,keep_ax_lst,fig)

                try: var_n=fnd.variables[varname][t] # Read 2d and 3d fields
                except KeyError: exit("Double check what file you are reading in and the variable(s) you are plotting...")
                #print(varname+': '+str(np.shape(var_n)))
                NDim=len(np.shape(var_n))       # Figure out if 2d or 3d
                if(NDim==3):
                    var_n=var_n[lev-1]  #in python, indicies start at 0.
                    figlev='_lev'+str(lev).zfill(2)  # If 3d, use lev to get 2d
                    figlevtitle="    LEV "+str(lev).zfill(2)
                else: figlev=''; figlevtitle=""; break_levlevels=True

                try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
                   if(NDim==3): print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z at lev={:s}".format(str(fhr).zfill(3),outdate, str(lev).zfill(2)))
                   else:        print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
                   function=dispatcher[varname]
                   var_n,clevs,cm,units,longname=function(var_n)
                except KeyError:
                   raise ValueError("invalid varname:"+varname)
    
                norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
                cs = m.pcolormesh(xt_n,yt_n,var_n,cmap=cm,norm=norm,vmin=clevs[0],vmax=clevs[-1])
                cbar = m.colorbar(cs,location='bottom',pad="5%",extend="both",ticks=clevs)
                cbar.ax.tick_params(labelsize=8.5)
                cbar.set_label(varname+": "+longname+"  ["+str(units)+"]")
                plt.title(outdate+"    FHR "+str(fhr).zfill(3)+figlevtitle,loc='left',fontsize=10,)
                plt.savefig(outputdir+varname+'_fhr'+'%03d'% (fhr) +'_v'+outdate+'Z'+figlev+'.png',\
                            dpi=125, bbox_inches='tight')
        if(break_levlevels): break #break out of lev levels loop.  


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

#-------------- tracer   -------------------------------------------------- 
def plot_sphum(var_n):
    """specific humidity [kg/kg]"""
    longname="specific humidity"; units="g/kg" #units="kg/kg *10-3"
    var_n=var_n*1000
    clevs=np.arange(0.,32.5,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_liq_wat(var_n):
    """liquid water content [g/kg]"""
    exit("Need to fix plot_liq_wat")
    longname="liq_wat"; units="g/kg" #units="kg/kg *10-3"
    clevs=np.arange(0.,32.5,1)
    clevs=[0.00,0.03,0.05,0.25,0.25,1.,3.,4.,5.]
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_o3mr(var_n):
    """ozone mixing ratio [kg/kg]"""
    exit("Need to fix plot_o3mr")
    longname="Ozone mixing ratio"; units="g/kg" #units="kg/kg *10-3"
    var_n=var_n*1000
    clevs=np.arange(0.,32.5,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

#-------------- core    -------------------------------------------------- 
def plot_u(var_n):
    """zonal wind [m/s]"""
    longname="zonal wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_v(var_n):
    """meridional wind [m/s]"""
    longname="meridional wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_W(var_n):
    """vertical velocity [m/s]"""
    longname="vertical velocity"; units="cm/s"
    if(units=="cm/s"): var_n=var_n*10
    clevs=np.arange(-20,20.5,2)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_DZ(var_n):
    """DZ [??]"""
    exit("Need to fix plot_DZ")
    longname="DZ"; units="??"
    clevs= np.arange(0,10,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_T(var_n):
    """temperature [K]"""
    longname="temperature"; units="F"
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_delp(var_n):
    """delp [??]"""
    exit("Need to fix plot_delp")
    longname="??"; units="??"
    clevs= np.arange(0,10,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_phis(var_n):
    """incremental pressure on each level [Pa]"""
    exit("Need to fix plot_phis")
    longname="incremental pressure on each level"; units="hPa"
    if(units=='hPa'): var_n=var_n*0.01
    clevs=np.arange(0,10,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)


#-------------- nggps2d -------------------------------------------------- 
def plot_ALBDOsfc(var_n):
    """surface albedo (%)"""
    longname="surface albedo"; units="%"
    clevs=np.arange(0.,100.5,5.)
    cm=ncepy.tcamt()
    return(var_n,clevs,cm,units,longname)

def plot_CPRATsfc(var_n):
    """surface convective precipitation rate [kg/m**2/s]"""
    longname="surface convective precipitation rate"; units="in."
    if(units=="in."): 
       var_n = var_n*3*3600/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
    elif(units=="mm"):
       var_n = var_n*3*3600      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
    clist=[0,23,22,21,20,19,10,17,16,15,14,29,28,24,25]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)


def plot_PRATEsfc(var_n):
    """surface precipitation rate [kg/m**2/s]"""
    longname="surface precipitation rate"; units="in."
    if(units=="in."):
       var_n = var_n*3*3600/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
    elif(units=="mm"):
       var_n = var_n*3*3600      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
    clist=[0,23,22,21,20,19,10,17,16,15,14,29,28,24,25]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_DLWRFsfc(var_n):
    """surface downward longwave flux [W/m**2]"""
    longname="surface downward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_ULWRFsfc(var_n):
    """surface upward longwave flux [W/m**2]"""
    longname="surface upward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_DSWRFsfc(var_n):
    """surface downward shortwave flux [W/m**2]"""
    longname="surface downward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_USWRFsfc(var_n):
    """surface upward shortwave flux [W/m**2]"""
    longname="surface upward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_DSWRFtoa(var_n):
    """top of atmos downward shortwave flux [W/m**2]"""
    longname="top of atmos downward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_USWRFtoa(var_n):
    """top of atmos upward shortwave flux [W/m**2]"""
    longname="top of atmos upward shortwave flux"; units="W/m**2"
    clevs=np.arange(0,1050,50)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_ULWRFtoa(var_n):
    """top of atmos upward longwave flux [W/m**2]"""
    longname="top of atmos upward longwave flux"; units="W/m**2"
    clevs=np.arange(0,525,25)
    #clevs=np.arange(0,15000.5,1000)
    cm=plt.get_cmap(name='Spectral_r')
    return(var_n,clevs,cm,units,longname)

def plot_GFLUXsfc(var_n):
    """surface ground heat flux [W/m**2]"""
    longname="surface ground heat flux"; units="W/m**2"
    clevs= [-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,0.,10.0,25.,50.,75.,100.,200.,300.]
    cm=ncepy.ncl_grnd_hflux()
    return(var_n,clevs,cm,units,longname)

def plot_HGTsfc(var_n):
    """surface geopotential height [gpm]"""
    longname="surface geopotential height"; units="gpm"
    clevs=[0,250.,500.,750.,1000.,1500.,2000.,3000.,4000.,5000.,7500.,10000.,15000.,20000.,25000.,30000.,30000.5]
    cm=plt.get_cmap(name='jet')
    return(var_n,clevs,cm,units,longname)

def plot_HPBLsfc(var_n):
    """surface planetary boundary layer height [m]"""
    longname="surface planetary boundary layer height"; units="m"
    clevs=[0,50.,100.,150.,200.,250.,500.,750.,1000.,1500.,2000.,3000.,4000.,5000.,7500.]
    cm=plt.get_cmap(name='jet')
    return(var_n,clevs,cm,units,longname)

def plot_ICECsfc(var_n):
    """surface ice concentration (ice=1; no ice=0) [fraction]"""
    longname="surface ice concentration"; units="(ice=1; no ice=0)"
    clevs=[0,0.5,1]
    clist=[23,27]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SLMSKsfc(var_n):
    """sea-land-ice mask (0-sea, 1-land, 2-ice)"""
    longname="sea-land-ice mask"; units="0-sea, 1-land, 2-ice"
    clevs=[0,1,2,2.01]
    clist=[24,18,27]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname) 

def plot_LHTFLsfc(var_n):
    """surface latent heat flux [W/m**2]"""
    longname="surface latent heat flux"; units="W/m**2"
    clevs=[-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,-5.,5.,10.0,25.,50.,75.,100.,200.,300]
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_SHTFLsfc(var_n):
    """surface sensible heat flux [W/m**2]"""
    longname="surface sensible heat flux"; units="W/m**2"
    clevs=[-300.,-200.,-100.,-75.,-50.,-25.0,-10.0,0.,10.0,25.,50.,75.,100.,200.,300.]
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_PRESsfc(var_n): # done
    """surface pressure [Pa]"""
    longname="surface pressure"; units="hPa"
    if(units=='hPa'): var_n=var_n*0.01
    var_n=scipy.ndimage.gaussian_filter(var_n, 2) # first pass
    var_n=scipy.ndimage.gaussian_filter(var_n, 2) # second pass
    clevs=np.arange(950.,1050.,4.)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_PWATclm(var_n):
    """atmos column precipitable water [kg/m**2]"""
    longname="atmos column precipitable water"; units="mm"
    clevs= [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76]
    clist = [0,16,18,30,28,27,25,4,23,3,21,8,5,19,17,31,12,2,7,14]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SOILM(var_n):
    """total column soil moisture content [kg/m**2]"""
    longname="total column soil moisture content"; units="in."
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="in."): var_n = var_n/25.4 #inches
    clevs= [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76]
    clist = [0,16,18,30,28,27,25,4,23,3,21,8,5,19,17,31,12,2,7,14]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SOILW1(var_n):
    """volumetric soil moisture 0-10cm [fraction]"""
    longname="volumetric soil moisture 0-10cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SOILW2(var_n):
    """volumetric soil moisture 10-40cm [fraction]"""
    longname="volumetric soil moisture 10-40cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SOILW3(var_n):
    """volumetric soil moisture 40-100cm [fraction]"""
    longname="volumetric soil moisture 40-100cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SOILW4(var_n):
    """volumetric soil moisture 100-200cm [fraction]"""
    longname="volumetric soil moisture 100-200cm"; units="fraction"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=[0,.10,.20,.30,.40,.50,.60,.70,.75,.80,.85,.90,.95,1]
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SPFH2m(var_n):
    """2m specific humidity [kg/kg]"""
    longname="2m specific humidity"; units="g/kg" #units="kg/kg *10-3"
    var_n=var_n*1000
    clevs=np.arange(0.,32.5,1)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_SOILT1(var_n): #done
    """soil temperature 0-10cm [K]"""
    longname="soil temperature 0-10cm"; units="F"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_SOILT2(var_n): #done
    """soil temperature 10-40cm [K]"""
    longname="soil temperature 10-40cm"; units="F"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_SOILT3(var_n): #done
    """soil temperature 40-100cm [K]"""
    longname="soil temperature 40-100cm"; units="F"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_SOILT4(var_n): #done
    """soil temperature 100-200cm [K]"""
    longname="soil temperature 100-200cm"; units="F"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_TMP2m(var_n):
    """2m temperature [K]"""
    longname="2m temperature"; units="F"
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_TMPsfc(var_n):
    """surface temperature [K]"""
    longname="surface temperature"; units="F"
    if(units=="F"): var_n=ncepy.Kelvin2F(var_n) # [F]
    clevs= np.arange(-36.,104.,4)
    cm=ncepy.ncl_t2m()
    return(var_n,clevs,cm,units,longname)

def plot_UGWDsfc(var_n): 
    """surface zonal gravity wave stress [N/m**2]"""
    longname="surface zonal gravity wave stress"; units="N/m**2"
    clevs=[-20,-10,-5,-2.5,-1,-0.05,-0.001,0.001,0.05,1,2.5,5,10,20]
    clevs=[-5,-2.5,-1,-0.05,-0.01,0.01,0.05,1,2.5,5]
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VGWDsfc(var_n): 
    """surface meridional gravity wave stress [N/m**2]"""
    longname="surface meridional gravity wave stress"; units="N/m**2"
    clevs=[-5,-2.5,-1,-0.05,-0.01,0.01,0.05,1,2.5,5]
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_UFLXsfc(var_n): 
    """surface zonal momentum flux [N/m**2]"""
    longname="surface zonal momentum flux"; units="N/m**2"
    clevs=np.arange(-1,1.05,0.1)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VFLXsfc(var_n): 
    """surface meridional momentum flux [N/m**2]"""
    longname="surface meridional momentum flux"; units="N/m**2"
    clevs=np.arange(-1,1.05,0.1)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_UGRD10m(var_n): 
    """10 meter u wind [m/s]"""
    longname="10 meter u wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_VGRD10m(var_n): 
    """10 meter v wind [m/s]"""
    longname="10 meter v wind"; units="m/s"
    clevs=np.arange(-20,20.5,2)
    cm=plt.get_cmap(name='RdBu_r')
    return(var_n,clevs,cm,units,longname)

def plot_WEASDsfc(var_n): 
    """surface snow water equivalent [kg/m**2]"""
    longname="surface snow water equivalent"; units="in."
    if(units=="in."):
       var_n = var_n/25.4  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
    elif(units=="mm"):
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
    clist=[0,23,22,21,20,19,10,17,16,15,14,29,28,24,25]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_SNODsfc(var_n): 
    """surface snow depth [m]"""
    longname="surface snow depth"; units="in."
    if(units=="in."):
       var_n = var_n/0.0254  # inches
       clevs=[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
    elif(units=="mm"):
       var_n = var_n/1000.      # mm
       clevs= [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
    clist=[0,23,22,21,20,19,10,17,16,15,14,29,28,24,25]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_ZORLsfc(var_n):
    """surface roughness [m]"""
    longname="surface roughness"; units="m"
    clevs=np.arange(0,3.1,0.1)
    cm=plt.get_cmap(name='jet') 
    return(var_n,clevs,cm,units,longname)

def plot_VFRACsfc(var_n):
    """vegetation fraction"""
    longname="vegetation fraction"; units="fraction"
    clevs=np.arange(0.,100.5,5.)
    cm=ncepy.tcamt()
    return(var_n,clevs,cm,units,longname)

def plot_F10Msfc(var_n): 
    """10-meter wind speed divided by lowest model wind speed"""
    longname="10-meter wind speed divided by lowest model wind speed"; units="none"
    clevs=np.arange(0,2.05,.1)
    cm=plt.get_cmap(name='jet') 
    return(var_n,clevs,cm,units,longname)

def plot_VTYPEsfc(var_n): 
    """vegetation type in integer 1-13"""
    longname="vegetation type in integer 1-13"; units="1-13"
    clevs=np.arange(1,15,1)
    cm=plt.get_cmap(name='jet')
    return(var_n,clevs,cm,units,longname)

def plot_STYPEsfc(var_n): 
    """soil type in integer 1-9"""
    longname="soil type"; units="1-9"
    var_n=maskoceans(lons, lats, var_n, inlands=True, resolution=res)
    clevs=np.arange(1,11,1)
    cm=plt.get_cmap(name='jet')
    return(var_n,clevs,cm,units,longname)

def plot_TCDCclm(var_n):
    """atmos column total cloud cover [%]"""
    longname="atmos column total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] # percent
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_TCDChcl(var_n):
    """high cloud level total cloud cover [%]"""
    longname="high cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] #percent
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_TCDCmcl(var_n): 
    """mid cloud level total cloud cover [%]"""
    longname="mid cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] #percent
    clist=[0,30,29,27,24,4,23,3,5,19,17,7,2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_TCDClcl(var_n): 
    """low cloud level total cloud cover [%]"""
    longname="low cloud level total cloud cover"; units="%"
    clevs=[0,10,20,30,40,50,60,70,75,80,85,90,95,100] # percent
    clist=[ 0,30,29,27,24, 4,23, 3, 5,19,17, 7, 2]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname) 

def plot_REFC(var_n):
    """Stoelinga simulated maximum (composite) reflectivity [dbz]"""
    longname="Stoelinga simulated maximum (composite) reflectivity"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    return(var_n,clevs,cm,units,longname) 

def plot_REFD1km(var_n):
    """Stoelinga simulated base (1 km AGL) reflectivity"""
    longname="Stoelinga simulated base (1 km AGL) reflectivity"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    return(var_n,clevs,cm,units,longname) 

def plot_REFD4km(var_n):
    """Stoelinga simulated base reflectivity"""
    longname="Stoelinga simulated base reflectivity"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    return(var_n,clevs,cm,units,longname) 

def plot_REFDm10C(var_n):
    """Reflectivity at -10C level"""
    longname="Reflectivity at -10C level"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    return(var_n,clevs,cm,units,longname) 

def plot_RETOP(var_n):
    """Echo top ( <= 18.5 dBz ) [m]"""
    longname="Echo top ( <= 18.5 dBz )"; units="kft"
    if(units=="kft"): var_n = var_n*304.8 #meters to kft 
    clevs=[0, 4, 7,10,13,16,19,22,25,28,31,34,37,40,43,46,49] # kft
    clist=[ 0,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname) 

def plot_MXUPHL2_5km(var_n):
    """2-5 km Updraft Helicity (max)"""
    longname="2-5 km Updraft Helicity (max)"; units="m/s**2"
    clevs=[0,25,50,75,100,150,200,250,300,400,500,600,700,800] # m/s**2 
    clist=[ 0, 4, 25,26,27, 23, 22, 21, 20, 18, 17, 15, 7 ]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname) 

def plot_MNUPHL2_5km(var_n):
    """2-5 km Updraft Helicity (min)"""
    longname="2-5 km Updraft Helicity (min)"; units="m/s**2"
    clevs=[0,25,50,75,100,150,200,250,300,400,500,600,700,800] # m/s**2 
    clist=[ 0, 4, 25,26,27, 23, 22, 21, 20, 18, 17, 15, 7 ]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname) 


def plot_MAXREFC(var_n):
    """Stoelinga simulated maximum (composite) reflectivity"""
    longname="Stoelinga simulated maximum (composite) reflectivity"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    return(var_n,clevs,cm,units,longname) 

def plot_MAXREF_1km(var_n):
    """Stoelinga simulated maximum (composite) reflectivity"""
    longname="Stoelinga simulated maximum (composite) reflectivity"; units="dbz"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
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
# tracer
        'sphum':plot_sphum,
        'liq_wat':plot_liq_wat,
        'o3mr':plot_o3mr,
# core
        'u':plot_u,
        'v':plot_v,
        'W':plot_W,
        'T':plot_T,
        'DZ':plot_DZ,
        'delp':plot_delp,
        'phis':plot_phis,
# nggps 2d
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
        'REFC':plot_REFC,
        'REFD1km':plot_REFD1km,
        'REFD4km':plot_REFD4km,
        'REFDm10C':plot_REFDm10C,
        'RETOP':plot_RETOP,
        'MXUPHL2_5km_max':plot_MXUPHL2_5km,
        'MNUPHL2_5km_min':plot_MNUPHL2_5km,
        'MAXREFC_max':plot_MAXREFC,
        'MAXREF_1km_max':plot_MAXREF_1km,
               }
    return dispatcher  

if __name__ == '__main__':
    #pool=multiprocessing.Pool(len(varnames)) # one processor per variable
    #pool=multiprocessing.Pool(8) # 8 processors for all variables. Just a little slower.
    #pool.map(mkplot,varnames) 
    mkplot(varnames[0])
    toc=timer()
    time=toc-tic
    hrs=int(time/3600)
    mins=int(time%3600/60)
    secs=int(time%3600%60)
    print("Total elapsed time: "+str(toc-tic)+" seconds.")
    print("Total elapsed time: "+str(hrs).zfill(2)+":"+str(mins).zfill(2)+":"+str(secs).zfill(2))





