from __future__ import print_function

import numpy as np
import math as math
from scipy.interpolate import UnivariateSpline

def int_fwhm(w,b,cw=np.nan):
    max2=np.max(b)/2
    b2=b-max2
    spline_flx= UnivariateSpline(w,b2, s=0)
    #print('Roots',spline_flx.roots())
    try:
        lb_flx, lr_flx = spline_flx.roots() # find the roots
    except:
        kk=spline_flx.roots()
        lb_flx=min(kk)
        lr_flx=max(kk)
    #print('My roots', lb_flx, lr_flx)
    fwhm_ang=np.abs(lb_flx-lr_flx)
    if cw ==0:
        print('you need to insert the central wavelegth')
        fwhm_vel=np.nan
    else:
        c=299792.458 #in km/s
        fwhm_vel=fwhm_ang*(c/cw)
    
    return fwhm_vel

