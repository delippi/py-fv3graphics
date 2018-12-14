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

global pdy,cyc,valpdy,valcyc,valtime,fhr

try:
   filename=str(sys.argv[1])
   datadir=str(sys.argv[2])
   pdy=str(int(sys.argv[3]))
   cyc=str(int(sys.argv[4])).zfill(2)
   valpdy=str(int(sys.argv[5]))
   valcyc=str(int(sys.argv[6])).zfill(2)
   valtime=str(int(sys.argv[7]))
   fhr=str(int(sys.argv[8])).zfill(3)
except:
   exit()

outputdir=datadir 
print(filename,pdy,cyc,valpdy,valcyc,valtime)
data_in = os.path.join(datadir,filename)   # name of analysis file
dom="CONUS"                                               # domain (can be CONUS, SC, etc.)
proj="gnom"                                               # map projection
#proj="cyl"
varnames=[                                                # uncomment the desired variables below
#        'ugrd',   \
#        'vgrd',   \
#        'dzdt',   \
        'delz',   \
#        'tmp',    \
#        'dpres',  \
#        'spfh',   \
#        'clwmr',  \
#        'rwmr',   \
#        'icmr',   \
#        'snmr',   \
#        'grle',   \
#        'o3mr',   \
#        'cld_amt',\
#        'pressfc',\
#        'hgtsfc', \
#        'dbz',    #\
         ]
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
    dispatcher=plot_Dictionary()
    global model_level
    model_level='column max'
    model_level=40
    if(model_level == 'column max'):
       var_n=fnd.variables[str(varname)+'midlayer'][0,:,:,:]
       var_n=var_n.max(axis=0) #take max across axis 0, in this case, max at each point across the column.
    else:
       var_n=fnd.variables[str(varname)+'midlayer'][0,64-model_level,:,:] 
    var_n=np.roll(var_n,nlon/2,axis=1)
    print(np.max(var_n))
    try: # Doing it this way means we only have to supply a corresponding definition for cm,clevs,etc.
       function=dispatcher[varname]
       var_n,clevs,cticks,cm,units,longname,title=function(var_n)
    except KeyError:
       raise ValueError("invalid varname:"+varname)

    m.contour(xi,yi,var_n,color='k')

    #cs = m.contourf(xi,yi,var_n,clevs,cmap=cm,extend='both')
    #cbar = m.colorbar(cs,location='bottom',pad="5%",extend="both",ticks=cticks)
    #cbar.ax.tick_params(labelsize=8.5)
    #cbar.set_label(varname+": "+longname+" ["+str(units)+"]")
    plt.title(title)

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.savefig(outputdir+'/gfs.t%sz.%s_v%s_atmf%s_%s.png' % (cyc,pdy+cyc,valtime,fhr,varname),dpi=250, bbox_inches='tight')

    print("fig is located: "+outputdir)

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
#/gpfs/hps3/emc/meso/save/Donald.E.Lippi/fv3gfs-20181022/sorc/fv3gfs.fd/FV3/atmos_cubed_sphere/driver/fvGFS/fv_nggps_diag.F90

#"gfs_dyn",     "ucomp",       "ugrd",       zonal wind (m/sec) 
#"gfs_dyn",     "vcomp",       "vgrd",       meridional wind (m/sec)
#"gfs_dyn",     "sphum",       "spfh",       
#"gfs_dyn",     "temp",        "tmp",        temperature (K)
#"gfs_dyn",     "liq_wat",     "clwmr",      
#"gfs_dyn",     "ice_wat",     "icmr",       
#"gfs_dyn",     "snowwat",     "snmr",       
#"gfs_dyn",     "rainwat",     "rwmr",       
#"gfs_dyn",     "graupel",     "grle",       
##"gfs_dyn",     "ice_nc",      "nccice",    
##"gfs_dyn",     "rain_nc",     "nconrd",    
#"gfs_dyn",     "o3mr",        "o3mr",       
#"gfs_dyn",     "cld_amt",     "cld_amt",
#"gfs_dyn",     "delp",        "dpres",      pressure thickness (pa)
#"gfs_dyn",     "delz",        "delz",       height thickness (m)
##"gfs_dyn",     "pfhy",        "preshy",    hydrostatic pressure (pa)
##"gfs_dyn",     "pfnh",        "presnh",    non-hydrostatic pressure (pa)
#"gfs_dyn",     "w",           "dzdt",       vertical wind (m/sec)
#"gfs_dyn",     "ps",          "pressfc",    surface pressure (pa)
#"gfs_dyn",     "hs",          "hgtsfc",     surface geopotential height (gpm)
#"gfs_dyn",     "reflectivity","dbz",        Stoelinga simulated reflectivity (dBz)



def plot_delz(var_n):
    """height thickness [m]"""
    longname="height thickness"; units="dam"
    var_n=var_n/10. # convert to decameters
    clevs=np.arange(500,600,5).tolist() 
    cticks=clevs
    cm='k'
    title="NATURE height thickness \n Valid %s %sZ" % (valpdy,valcyc)
    return(var_n,clevs,cticks,cm,units,longname,title)


def plot_dbz(var_n): 
    """reflectivity [dBz]"""
    longname="reflectivity"; units="dBZ"
    clevs=[-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    cticks=[-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75] # dbz 
    #cm=ncepy.radarmap()
    cm=mrms_radarmap_with_gray()
    if(model_level=='column max'):
        title="NATURE Composite Simulated Reflectivity \n Valid %s %sZ" % (valpdy,valcyc)
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
#        'ugrd':plot_ugrd,
#        'vgrd':plot_vgrd,
#        'dzdt':plot_dzdt,
        'delz':plot_delz,
#        'tmp':plot_tmp,
#        'dpres':plot_dpres,
#        'spfh':plot_spfh,
#        'clwmr':plot_clwmr,
#        'rwmr':plot_rwmr,
#        'icmr':plot_icmr,
#        'snmr':plot_snmr,
#        'grle':plot_grle,
#        'o3mr':plot_o3mr,
#        'cld_amt':plot_cld_amt,
#        'pressfc':plot_pressfc,
#        'hgtsfc':plot_hgtsfc,
        'dbz':plot_dbz,
               }
    return dispatcher  

def mrms_radarmap_with_gray():
    from matplotlib import colors
    r=[0.66,0.41,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.91,1.00,1.00,0.80,0.60,1.00,0.60]
    g=[0.66,0.41,0.93,0.63,0.00,1.00,0.78,0.56,1.00,0.75,0.56,0.00,0.20,0.00,0.00,0.20]
    b=[0.66,0.41,0.93,0.96,0.96,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.80]
    rgb=zip(r,g,b)
    cmap=colors.ListedColormap(rgb,len(r))
    cmap.set_over(color='white')
    cmap.set_under(color='white')
    return cmap


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





