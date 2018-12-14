from timeit import default_timer as timer
tic=timer()
import sys,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

global pdy,cyc

try:
   filename="results_conv_ges.@pdy@@cyc@"
   datadir="/scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/anl/rh2018/201809/@pdy@/@cyc@/"
   pdy="@pdy@"
   cyc="@cyc@"
except:
   exit()
   
outputdir=datadir 
print(filename,pdy,cyc)
data_in = os.path.join(datadir,filename)   # name of analysis file
dom="CONUS"                                               # domain (can be CONUS, SC, etc.)
proj="gnom"                                               # map projection
#proj="cyl"
######################   USER DEFINED SETTINGS    ############################################

# Create the basemap
# create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

# Setup map corners for plotting.  This will give us CONUS
llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=ncepy.corners_res(dom,proj=proj)
if(proj=="cyl"):
  llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=-180,-80,180,80,'c'
#  llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=-80,30,-70,40,'h'
  #llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,res=-30,40,30,70,'l'
lon_0=-95.0
lat_0=25.0
offsetup=0.
offsetright=0.
m = Basemap(llcrnrlon=llcrnrlon+offsetright,   llcrnrlat=llcrnrlat+offsetup,
               urcrnrlon=urcrnrlon+offsetright,  urcrnrlat=urcrnrlat+offsetup,
               projection=proj, lat_0=lat_0,lon_0=lon_0,
               resolution=res,ax=ax)

# Map background stuff to make things look nice
parallels = np.arange(-80.,80,10.)
meridians = np.arange(-180,180.,10.)
m.drawcoastlines(linewidth=1.25)
m.drawcountries(linewidth=1.25)

varname='scatter'

def mkplot(varname):
    print("mkplot - "+str(multiprocessing.current_process()))
    print(data_in)
    import re
    f = open(data_in,"r")
    line = f.readlines()
    f.close

    for i in range(len(line)):
       line[i]=re.sub(' +',' ',line[i])
       lat,lon=float(line[i].split(" ")[7]),float(line[i].split(" ")[8])
       m.scatter(lon,lat,s=0.5,marker='o',color='r',latlon=True)
       if(i % 250 == 0):
         update_progress(float(i)/len(line))


    #m.scatter(lon,lat,s=125,marker='o',color='k',latlon=True)

    plt.title("Radial wind observation locations \n"+filename)

    plt.savefig(outputdir+'/'+filename+'.png',dpi=250, bbox_inches='tight')

    print("fig is located: "+outputdir)

    plt.close('all')

def update_progress(progress):
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__ == '__main__':
    mkplot(varname)
    toc=timer()
    time=toc-tic
    hrs=int(time/3600)
    mins=int(time%3600/60)
    secs=int(time%3600%60)
    print("Total elapsed time: "+str(toc-tic)+" seconds.")
    print("Total elapsed time: "+str(hrs).zfill(2)+":"+str(mins).zfill(2)+":"+str(secs).zfill(2))





