#!/usr/bin/env python
#PBS -N fv3py
#PBS -l walltime=0:05:00
#PBS -l nodes=1:ppn=4
#PBS -q debug
#PBS -A fv3-cpu
#PBS -o fv3py.out
#PBS -j oe

import sys,ncepy,os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap, cm
import multiprocessing
import numpy as np
from netCDF4 import Dataset
from netcdftime import utime
from datetime   import datetime,timedelta
#Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')
import pdb

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

def plot_PRATEsfc(var_n):
    """surface precipitation rate [kg/m**2/s]"""
    longname="surface precipitation rate"; units="in."
    if(units=="in."):
       var_n = var_n*3*3600/25.4  # inches
       clevs =[0,0.01,0.05,0.1,0.25,0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,7.] #inches
    elif(units=="mm"):
       var_n = var_n*3*3600      # mm
       clevs = [0,0.1,2,5,10,15,20,25,35,50,75,100,125,150,175]  #mm
    clist=[0,23,22,21,20,19,10,17,16,15,14,29,28,24,25]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)

def plot_DLWRFsfc(var_n):
    """surface downward longwave flux [W/m**2]"""
    longname="surface downward longwave flux"; units="W/m**2"
    clevs = [0,10,25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200]
    clist = [0,30,28,27,25,4,23,3,21,8,5,19,17,31,12,2,7,14]
    cm=gemplot(clist)
    return(var_n,clevs,cm,units,longname)


dispatcher={'PRATEsfc':plot_PRATEsfc,'DLWRFsfc':plot_DLWRFsfc}
varnames=['PRATEsfc','DLWRFsfc']
outputdir = './'                                         # output directory
dir = '/scratch4/NCEPDEV/meso/save/Donald.E.Lippi/data'  # directory containing your fv3nc files.
startplot=datetime.strptime('2017080700','%Y%m%d%H')     # start time
endplot=datetime.strptime('2017080700','%Y%m%d%H')       # end time (if start=end ==> 1 plot)
nesteddata2d = os.path.join(dir,'nggps2d.nest02.nc')     # name of file
nestedgrid = os.path.join(dir,'grid_spec.nest02.nc')     # name of file
dom="CONUS"                                              # domain (can be CONUS, SC, etc.)
proj="gnom"   
print('Reading {:s}'.format(nesteddata2d))
fnd = Dataset(nesteddata2d,'r')
print('Reading {:s}'.format(nestedgrid))
fng   = Dataset(nestedgrid,'r')
grid_lont_n  = fng.variables['grid_lont'][:,:]
grid_latt_n  = fng.variables['grid_latt'][:,:]

times = fnd.variables['time'][:]
cdftime = utime(getattr(fnd.variables['time'],'units'))
# Grab the cycledate
cycledate=roundTime(cdftime.num2date(times[0]),roundTo=60.*60.)

def main():
 pool=multiprocessing.Pool(10)
 pool.map(mkplot,varnames)

def mkplot(varname):
   # ceate the basemap
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
   
   #  Map/figure has been set up here (bulk of the work), save axes instances for
   #     use again later
   keep_ax_lst = ax.get_children()[:]

   # Transform lats and lons to map proj coords
   #x_n,y_n = m(grid_lon_n[:,:],grid_lat_n[:,:])
   xt_n,yt_n = m(grid_lont_n[:,:],grid_latt_n[:,:])

   cycledate=roundTime(cdftime.num2date(times[0]),roundTo=60.*60.)
   for t,time in enumerate(times):

      # Just to get a nicely formatted date for writing out
      datestr =roundTime(cdftime.num2date(time),roundTo=60.*60.)
      outdate=datestr.strftime('%Y%m%d%H')
      diff=(datestr-cycledate)
      fhr=int(diff.days*24.+diff.seconds/3600.)
   # for varname in varnames:
      if datestr>=startplot and datestr<=endplot:
        print("Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
        # Clear off old plottables but keep all the map info
        ncepy.clear_plotables(ax,keep_ax_lst,fig)
        var_n=fnd.variables[varname][:,:,:]
        try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
           print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
           function=dispatcher[varname]
           var_n,clevs,cm,units,longname=function(var_n)
        except KeyError:
           raise ValueError("invalid varname:"+varname)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
        cs = m.pcolormesh(xt_n,yt_n,var_n[t,:,:],cmap=cm,norm=norm,vmin=clevs[0],vmax=clevs[-1])
        cbar = m.colorbar(cs,location='bottom',pad="5%",extend="both",ticks=clevs)
        cbar.ax.tick_params(labelsize=8.5)
        cbar.set_label(varname+": "+longname+"  ["+str(units)+"]")
        plt.title(outdate+"    FHR "+str(fhr).zfill(3),loc='left',fontsize=10,)
        plt.savefig(outputdir+ varname + '_%03d.png' % (fhr),dpi=125, bbox_inches='tight')

   plt.close('all')



main()
