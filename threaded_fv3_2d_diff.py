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



######################SER DEFINED SETTINGS    ############################################
CDUMP='gdas' #gdas or gfs
#datadir='/gpfs/hps2/stmp/Donald.E.Lippi/fv3gfs_dl2rw_DAS_exp_001_2018050218/'+CDUMP+'.20180503/00/'
#datadir='/scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/2018050218/rw_001/gdas.20180503/06/'
#datadir='/scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/2018062800/rw_001/gfs.20180628/00/'
#datadir='/scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/2018091100/NATURE-2018091100-2018091800/gfs.20180911/00'
#datadir='/scratch4/NCEPDEV/stmp3/Donald.E.Lippi/fv3gfs_dl2rw/2018052418/NATURE-2018052418-2018060100/gdas.20180524/18'
global SOT
SOT='NA' # 90-deg or 00-deg or NA
global pdy,cyc,valpdy,valcyc,valtime,fhr
#data_in = os.path.join(datadir,'gdas.t06z.atminc.nc')   # name of analysis file
try:
   filename=str(sys.argv[1])
   datadir=str(sys.argv[2])
   pdy=str(int(sys.argv[3]))
   cyc=str(int(sys.argv[4])).zfill(2)
   valpdy=str(int(sys.argv[5]))
   valcyc=str(int(sys.argv[6])).zfill(2)
   valtime=str(int(sys.argv[7]))
   fhr=str(int(sys.argv[8])).zfill(3)

   nature_filename=str(sys.argv[9])
   naturedir=str(sys.argv[10])
   npdy=str(int(sys.argv[11]))
   ncyc=str(int(sys.argv[12])).zfill(2)
   nvalpdy=str(int(sys.argv[13]))
   nvalcyc=str(int(sys.argv[14])).zfill(2)
   nvaltime=str(int(sys.argv[15]))
   nfhr=str(int(sys.argv[16])).zfill(3)
except:
   exit()
#   filename='gfs.t00z.atmf000.nc4'
#   pdy="20180911"
#   cyc="00"
#   valpdy="20180911"
#   valcyc="00"
#   valtime="2018091100"
#   fhr="000"
outputdir=datadir 
print(filename,pdy,cyc,valpdy,valcyc,valtime)
print(nature_filename,npdy,ncyc,nvalpdy,nvalcyc,nvaltime)
data_in = os.path.join(datadir,filename)   # name of analysis file
nature_in=os.path.join(naturedir,nature_filename) 
#data_in = os.path.join(datadir,'gdas.t18z.atmf000.nc4')   # name of analysis file
dom="CONUS"                                               # domain (can be CONUS, SC, etc.)
proj="gnom"                                               # map projection
#proj="cyl"
varnames=[                                                # uncomment the desired variables below
#          'ugrd',#\
          'dbz',#\
#          'vgrd',\
#          'T_inc',\
         ]
######################   USER DEFINED SETTINGS    ############################################

# Create the basemap
# create figure and axes instances
fig = plt.figure(figsize=(8,8))
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
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
#m.drawstates(linewidth=1.25)
m.drawcountries(linewidth=1.25)
#m.drawparallels(parallels,labels=[1,0,0,1])
#m.drawmeridians(meridians,labels=[1,0,0,1],rotation=45)
#for m in meridians:
#    meridians[m][1][0].set_rotation(45)
#m.drawcounties(linewidth=0.2, color='k')

def mkplot(varname):
    print("mkplot - "+str(multiprocessing.current_process()))
    print(data_in)
    fnd = Dataset(data_in,mode='r')
    varnames2d=fnd.variables.keys()
    print(varnames2d)
    global lons,lats
    lons  = fnd.variables['lon'][:]
    lats  = fnd.variables['lat'][:]
    lons=lons-180
    nlon=len(lons)
    keep_ax_lst = ax.get_children()[:]
    # Transform lats and lons to map proj coords
    lon,lat=np.meshgrid(lons,lats)
    xi,yi = m(lon,lat)


    print(nature_in)
    nature_fnd = Dataset(nature_in,mode='r')
    global nature_lons,nature_lats
    nature_lons  = nature_fnd.variables['lon'][:]
    nature_lats  = nature_fnd.variables['lat'][:]
    nature_lons=nature_lons-180
    nature_nlon=len(nature_lons)
    keep_ax_lst = ax.get_children()[:]
    # Transform lats and lons to map proj coords
    nature_lon,nature_lat=np.meshgrid(nature_lons,nature_lats)
    nxi,nyi = m(nature_lon,nature_lat)





    dispatcher=plot_Dictionary()
    global model_level
    model_level='column max'
    if(model_level == 'column max'):
       var_n=              fnd.variables[str(varname)+'midlayer'][0,:,:,:]
       nature_var_n=nature_fnd.variables[str(varname)+'midlayer'][0,:,:,:]
       var_n=var_n.max(axis=0) #take max across axis 0, in this case, max at each point across the column.
       nature_var_n=nature_var_n.max(axis=0)
       # HERE IS WHERE WE COMPUTE THE DIFFERENCE BETWEEN EXPERIMENT AND TRUTH
#       var_n=np.ma.masked_where(var_n < 5,var_n)
#       nature_var_n=np.ma.masked_where(nature_var_n < 5,nature_var_n)
#       var_n=np.clip(var_n,0,100)
#       nature_var_n=np.clip(nature_var_n,0,100)
       
       
       diff_var_n=(var_n)-(nature_var_n)
       print(type(diff_var_n))
    else:
       var_n=fnd.variables[str(varname)+'midlayer'][0,64-model_level,:,:] 
       nature_var_n=nature_fnd.variables[str(varname)+'midlayer'][0,64-model_level,:,:] 
    var_n=np.roll(var_n,nlon/2,axis=1)
    nature_n=np.roll(nature_var_n,nlon/2,axis=1)
    print(np.max(var_n),np.min(var_n))
    print(np.max(nature_var_n),np.min(nature_var_n))
    print(np.max(diff_var_n),np.min(diff_var_n))
    try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
       #print(str(varname)+": Plotting forecast hour {:s} valid {:s}Z".format(str(fhr).zfill(3),outdate))
       function=dispatcher[varname]
       var_n,clevs,cticks,cm,units,longname,title=function(var_n)
       #nature_var_n,clevs,cticks,cm,units,longname,title=function(nature_var_n)

    except KeyError:
       raise ValueError("invalid varname:"+varname)


    option1=True
    option2=False
    if(option1):
       cm.set_under(color='white',alpha=0)
       cm.set_over(color='white',alpha=0)
       cs1=  m.contourf(nxi,nyi,nature_var_n,clevs,cmap=mrms_radarmap_grayscale(),extend='both')
       #cs1=  m.contourf(nxi,nyi,nature_var_n,clevs,cmap=cm,extend='both')
       cs2 = m.contourf(xi,yi,       var_n,clevs,cmap=cm,                       extend='both')
       cbar1 = m.colorbar(cs1,location='bottom',pad="-4%",extend="both",ticks=cticks)
       cbar2 = m.colorbar(cs2,location='bottom',pad="8%",extend="both",ticks=cticks)
       cbar1.ax.tick_params(labelsize=8.5)
       cbar2.ax.tick_params(labelsize=8.5)
       cbar2.set_label(varname+": "+longname+" ["+str(units)+"]")
    if(option2):
       #clevs  =[-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,\
       #           5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz
       clevs  =[-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40] # dbz
       clevmin,clevmax,inc=-40.0,40.0,5.
       N=int((-1.*clevmin+clevmax)/inc+1)
       clevs = np.linspace(clevmin,clevmax,N)
       cticks=clevs
       cmap=colormap.diff_colormap(clevs)

       h1="XX"
       h2=""
       hatches=[ h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,h1,\
                 h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2,h2]
       #cs = m.contourf(xi,yi,var_n,clevs,cmap=mrms_radarmap_diff(),hatches=hatches,extend='both')
       cs = m.contourf(xi,yi,diff_var_n,clevs,cmap=cmap,extend='both')
       cbar = m.colorbar(cs,location='bottom',pad="8%",extend="both",ticks=cticks)
       cbar.ax.tick_params(labelsize=8.5)
       cbar.set_label(varname+": "+longname+" ["+str(units)+"]")
       

    plt.title(title)

    # Make zoomed inset between title and save fig so that the title is placed correctly.
    #oblat,oblon=33.36333333,-84.56583333
    #ratio=16./9.
    #num=3.0 #numerator, also the half width of the plot (num=1 means 2 degrees lon wide).
    #x1,y1,x2,y2= oblon-num, oblat-(num/ratio), oblon+num, oblat+(num/ratio)
    #ratioaxinszoom=16./num #8./num #5./num
    #axins = zoomed_inset_axes(ax, ratioaxinszoom, loc=1)
    #map2 = Basemap(llcrnrlon=x1,llcrnrlat=y1,urcrnrlon=x2,urcrnrlat=y2,\
    #          rsphere=(6378137.00,6356752.3142),\
    #          resolution=res,projection=proj,\
    #          lat_0=lat_0,lon_0=lon_0,ax=ax)
    #map2.drawcoastlines(linewidth=1.25)
    #map2.drawstates(linewidth=1.25)
    #map2.drawcountries(linewidth=1.25)
    #map2.drawcounties(linewidth=0.2)
    #map2.contourf(xi,yi,var_n,clevs,cmap=cm,latlon=True,extend='both')
    #CS = map2.contour(xi,yi,var_n,clevs,colors='k')
    #plt.clabel(CS, inline=1, fontsize=10, colors='k')
    #map2.scatter(oblon,oblat,s=25,color='k',marker='.',latlon=True)
    #x1,y1=m(x1,y1)
    #x2,y2=m(x2,y2)
#    axins.set_xlim(x1,x2)
#    axins.set_ylim(y1,y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
#    m.scatter(-76.88,34.76,s=200,color='k',marker='*',latlon=True)
#    m.scatter(-78.49,35.67,s=200,color='k',marker='*',latlon=True)
#    m.scatter(-77.01,36.98,s=200,color='k',marker='*',latlon=True)
    #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", lw=1.75, ec="green")
    #for axis in ['top','bottom','left','right']:
    #    axins.spines[axis].set_linewidth(1.75)
    #    axins.spines[axis].set_color('g')
    ########### END make inset ##################

    #plt.savefig(outputdir+'/'+varname + '_%s.png' % (valtime),dpi=250, bbox_inches='tight')
#    plt.savefig(outputdir+'/gfs.t%sz.atmf%s_%s_%s_v%s.png' % (cyc,fhr,varname,pdy+cyc,valtime),dpi=250, bbox_inches='tight')
    plt.savefig(outputdir+'/gfs.t%sz.%s_v%s_atmf%s_%s_againstTruth.png' % (cyc,pdy+cyc,valtime,fhr,varname),dpi=250, bbox_inches='tight')

    print("fig is located: "+outputdir)

    plt.close('all')


############### useful functions ###########################################
def mrms_radarmap_diff():
    from matplotlib import colors
    rneg=[0.60,1.00,0.60,0.80,1.00,1.00,0.91,1.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00]
    rpos=[1.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.91,1.00,1.00,0.80,0.60,1.00,0.60]
    gneg=[0.20,0.00,0.00,0.20,0.00,0.56,0.75,1.00,0.56,0.78,1.00,0.00,0.63,0.93,1.00]
    gpos=[1.00,0.93,0.63,0.00,1.00,0.78,0.56,1.00,0.75,0.56,0.00,0.20,0.00,0.00,0.20]
    bneg=[0.80,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.96,0.96,0.93,1.00]
    bpos=[1.00,0.93,0.96,0.96,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.80]

    r=rneg+rpos
    g=gneg+gpos
    b=bneg+bpos


    rgb=zip(r,g,b)
    cmap=colors.ListedColormap(rgb,len(r))
    cmap.set_over(color='white')
    cmap.set_under(color='white')
    return cmap



def mrms_radarmap_grayscale():
    from matplotlib import colors
    r=[0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.91,1.00,1.00,0.80,0.60,1.00,0.60]
    g=[0.93,0.63,0.00,1.00,0.78,0.56,1.00,0.75,0.56,0.00,0.20,0.00,0.00,0.20]
    b=[0.93,0.96,0.96,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.80]
    gray=[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]

    for i in range(len(r)):
        red   = r[i]*0.2989
        green = g[i]*0.5870
        blue  = b[i]*0.1140
        gray[i] = red + green + blue

    
    rgb=zip(r,g,b)
    rgb=zip(gray,gray,gray)
    print(rgb)
    cmap=colors.ListedColormap(rgb,len(r))
    cmap.set_over(color='white')
    cmap.set_under(color='white')
    return cmap


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
def plot_ugrd(var_n): 
    """zonal wind [m/s]"""
    longname="zonal wind (u)"; units="m/s"
    if(SOT == 'NA'):
       clevmin,clevmax,inc=-2.,2.,.2
       cticks=[-2.,-1.6,-1.2,-0.8,-0.4,0.,0.4,0.8,1.2,1.6,2.]
    N=int((-1.*clevmin+clevmax)/inc+1)
    clevs = np.linspace(clevmin,clevmax,N)
    cm=colormap.diff_colormap(clevs)
    if(SOT == 'NA'):
      title="Increments (anl - bkgnd) of %s %d \nmodel level: %2d" % (longname,date,model_level)
    else:
      title="%s Single Observation Experiment \nIncrements (anl - bkgnd) of %s %d \nmodel level: %2d" \
           % (SOT,longname,date,model_level)
    return(var_n,clevs,cticks,cm,units,longname,title)

def plot_vgrd(var_n): 
    """meridional wind [m/s]"""
    longname="meridional wind (v)"; units="m/s"
    if(SOT == '00-deg'):
       clevmin,clevmax,inc=-1.,1.,.1
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]
    elif(SOT == '90-deg'):
       clevmin,clevmax,inc=-1./100,1./100,.1/100
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]/100
    elif(SOT == 'NA'):
       #clevmin,clevmax,inc=-5.,5.,.5
       #cticks=[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
       clevmin,clevmax,inc=-1.,1.,.1
       cticks=[-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.]
    N=int((-1.*clevmin+clevmax)/inc+1)
    clevs = np.linspace(clevmin,clevmax,N)
    cm=colormap.diff_colormap(clevs)
    if(SOT == 'NA'):
      title="Increments (anl - bkgnd) of %s %d \nmodel level: %2d" % (longname,date,model_level)
    else:
      title="%s Single Observation Experiment \nIncrements (anl - bkgnd) of %s %d \nmodel level: %2d" \
           % (SOT,longname,date,model_level)
    return(var_n,clevs,cticks,cm,units,longname,title)

def plot_dbz(var_n): 
    """reflectivity [dBz]"""
    longname="reflectivity"; units="dBZ"
    clevs=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cticks=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cm=ncepy.mrms_radarmap()
    if(SOT == 'NA'):
      if(model_level=='column max'):
         title="FV3GFS Composite Simulated Reflectivity F%s \n%s %sZ Valid %s %sZ" % (fhr,pdy,cyc,valpdy,valcyc)
      else:
         title="FV3GFS Simulated Reflectivity F%s \n%s %sZ Valid %s %sZ" % (fhr,pdy,cyc,valpdy,valcyc)
    else:
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
        'ugrd':plot_ugrd,
        'vgrd':plot_vgrd,
        'dbz':plot_dbz,
               }
    return dispatcher  

#def Make_Zoomed_Inset_Plot():

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





