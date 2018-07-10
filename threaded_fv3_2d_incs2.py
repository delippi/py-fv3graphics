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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import multiprocessing
import numpy as np
from netCDF4 import Dataset
import colormap
from netcdftime import utime
from datetime   import datetime,timedelta
import scipy
import ncepy
#Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')
import pdb

######################   USER DEFINED SETTINGS    ############################################
CDUMP='gdas' #gdas or gfs
datadir='/gpfs/hps2/stmp/Donald.E.Lippi/fv3gfs_dl2rw_DAS_exp_001_2018050218/'+CDUMP+'.20180503/00/'
outputdir=datadir 
global SOT
SOT='00-deg' # 90-deg or 00-deg
global date
date=2018050300
data_inc = os.path.join(datadir,'gdas.t00z.atminc.nc')   # name of analysis file
dom="CONUS"                                               # domain (can be CONUS, SC, etc.)
proj="gnom"                                               # map projection
proj="cyl"                                               # map projection
varnames=[                                                # uncomment the desired variables below
          'u_inc',#\
#          'v_inc',\
#          'T_inc',\
         ]
######################   USER DEFINED SETTINGS    ############################################

# Create the basemap
# create figure and axes instances
fig = plt.figure(figsize=(12,12))
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax = plt.subplot(111)

# Setup map corners for plotting.  This will give us CONUS
llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=ncepy.corners_res(dom,proj=proj)
#lat_0=35.4
#lon_0=-97.6
lon_0=-95.0
lat_0=25.0
offsetup=3.
offsetright=14.
m = Basemap(llcrnrlon=llcrnrlon+offsetright,   llcrnrlat=llcrnrlat+offsetup,
               urcrnrlon=urcrnrlon+offsetright,  urcrnrlat=urcrnrlat+offsetup,
               projection=proj, lat_0=lat_0,lon_0=lon_0,
               resolution=res,ax=ax)

# Map background stuff to make things look nice
parallels = np.arange(-80.,90,10.)
meridians = np.arange(0.,360.,10.)
m.drawcoastlines(linewidth=1.25)
m.drawstates(linewidth=1.25)
m.drawcountries(linewidth=1.25)
m.drawparallels(parallels,labels=[1,0,0,1])
m.drawmeridians(meridians,labels=[1,0,0,1])
m.drawcounties(linewidth=0.2, color='k')

def mkplot(varname):
    print("mkplot - "+str(multiprocessing.current_process()))
    fnd = Dataset(data_inc,mode='r')
    #varnames2d=fnd.variables.keys()
    global lons,lats
    lons  = fnd.variables['lon'][:]
    lats  = fnd.variables['lat'][:]
    keep_ax_lst = ax.get_children()[:]
    # Transform lats and lons to map proj coords
    lon,lat=np.meshgrid(lons,lats)
    xi,yi = m(lon,lat)
    dispatcher=plot_Dictionary()
    global model_level
    model_level=11
    var_n=fnd.variables[varname][64-model_level,:,:]
    try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
       #print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
       function=dispatcher[varname]
       var_n,clevs,cticks,cm,units,longname,title=function(var_n)
    except KeyError:
       raise ValueError("invalid varname:"+varname)
    cs = m.contourf(xi,yi,var_n,clevs,cmap=cm,latlon=True,extend='both')
    cbar = m.colorbar(cs,location='bottom',pad="8%",extend="both",ticks=cticks)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label(varname+": "+longname+" increments  ["+str(units)+"]")
    #plt.title(outdate+"    FHR "+str(fhr).zfill(3),loc='left',fontsize=10,)
    plt.title(title)

    # Make zoomed inset between title and save fig so that the title is placed correctly.
    oblat=30.72
    oblon=-97.38
    m.scatter(oblon,oblat,s=25,color='k',marker='.',latlon=True)
    ratio=16./9.
    num=3.0 #numerator, also the half width of the plot (num=1 means 2 degrees lon wide).
    x1,y1,x2,y2= oblon-num, oblat-(num/ratio), oblon+num, oblat+(num/ratio)
    ratioaxinszoom=16./num #8./num #5./num
    axins = zoomed_inset_axes(ax, ratioaxinszoom, loc=1)
    map2 = Basemap(llcrnrlon=x1,llcrnrlat=y1,urcrnrlon=x2,urcrnrlat=y2,\
              rsphere=(6378137.00,6356752.3142),\
              resolution=res,projection=proj,\
              lat_0=lat_0,lon_0=lon_0,ax=axins)
    map2.drawcoastlines(linewidth=1.25)
    map2.drawstates(linewidth=1.25)
    map2.drawcountries(linewidth=1.25)
    map2.drawcounties(linewidth=0.2)
    map2.contourf(xi,yi,var_n,clevs,cmap=cm,latlon=True,extend='both')
    CS = map2.contour(xi,yi,var_n,clevs,colors='k')
    plt.clabel(CS, inline=1, fontsize=10, colors='k')
    map2.scatter(oblon,oblat,s=25,color='k',marker='.',latlon=True)
    x1,y1=m(x1,y1)
    x2,y2=m(x2,y2)
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", lw=1.75, ec="green")
    for axis in ['top','bottom','left','right']:
        axins.spines[axis].set_linewidth(1.75)
        axins.spines[axis].set_color('g')
    ########### END make inset ##################

    plt.savefig(outputdir+ varname + '_%03d_incs.png' % (00),dpi=125, bbox_inches='tight')

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
def plot_u_inc(var_n): 
    """zonal wind [m/s]"""
    longname="zonal wind (u)"; units="m/s"
    print(SOT)
    if(SOT == '00-deg'):
       clevmin,clevmax,inc=-1.,1.,.1
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]
    elif(SOT == '90-deg'):
       clevmin,clevmax,inc=-1./100,1./100,.1/100
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]/100
    N=int((-1.*clevmin+clevmax)/inc+1)
    clevs = np.linspace(clevmin,clevmax,N)
    cm=colormap.diff_colormap(clevs)
    title="%s Single Observation Experiment \nIncrements (anl - bkgnd) of %s %d \nmodel level: %2d" \
           % (SOT,longname,date,model_level)
    return(var_n,clevs,cticks,cm,units,longname,title)

def plot_v_inc(var_n): 
    """meridional wind [m/s]"""
    longname="meridional wind (v)"; units="m/s"
    if(SOT == '00-deg'):
       clevmin,clevmax,inc=-1.,1.,.1
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]
    elif(SOT == '90-deg'):
       clevmin,clevmax,inc=-1./100,1./100,.1/100
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]/100
    N=int((-1.*clevmin+clevmax)/inc+1)
    clevs = np.linspace(clevmin,clevmax,N)
    cm=colormap.diff_colormap(clevs)
    title="%s Single Observation Experiment \nIncrements (anl - bkgnd) of %s %d \nmodel level: %2d" \
           % (SOT,longname,date,model_level)
    return(var_n,clevs,cticks,cm,units,longname,title)

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
        'u_inc':plot_u_inc,
        'v_inc':plot_v_inc,
               }
    return dispatcher  

#def Make_Zoomed_Inset_Plot():

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





