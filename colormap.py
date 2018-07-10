import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def diff_colormap(clevs):
    #assert len(clevs) % 2 == 0, 'N levels must be even.'
    #clevs = np.linspace(-10,10,20)
    size=len(clevs);    print('size=',size)
    sizeby2=(size)/2;   print('sizeby2=',sizeby2)
    pd=1./sizeby2;      print('pd=',pd)
    colors = [ () for i in range(size) ]; print('colors=',colors)
    incup=0;            print('incup',incup)
    incdown=1;          print('incdown',incdown)
    blue= (0,0,1);      print('blue=',blue)
    colors[0]=blue;
    for j in range(1,sizeby2-1):
        incup=incup+pd
        colors[j] = (incup,incup,1)
    colors[sizeby2-1]=(1.,1.,1.)
    colors[sizeby2  ]=(1.,1.,1.)
    colors[sizeby2+1]=(1.,1.,1.)
    red = (1,0,0);      print('red=',red)
    colors[-1]=red
    for k in range(sizeby2+2,size):
        incdown=incdown-pd
        colors[k] = (1,incdown,incdown)

    levs=1
    #print (len(list),type(list),list)
    #clevs = np.arange(-10,10,0.5)
    #levs = range(size)
    #assert len(clevs) % 2 == 1, 'N levels must be odd.'
    cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
                                                     colors = colors)

    #cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
    #                                                 colors =[(0, 0, 1),
    #                                                          (1, 1., 1),
    #                                                          (1, 0, 0)],
    #                                                 N=len(levs),
    #                                                 )
    return cmap

