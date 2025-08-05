from __future__ import print_function


from astropy.io import fits 
from scipy import ndimage
import numpy as np
import glob,timeit
#from time import clock
import matplotlib.pyplot as plt
import ppxf_util_al_lab_wv as util
from scipy.optimize import leastsq
#from scipy import random
import time
from astropy.table import Table


def noise_sim(file_dir,exm=2.0):
    '''exm: # of dispersion velocity wavelenght to construct the limits'''
    #Read the noise
    hdu=fits.open(file_dir)
    t=hdu[1].data
    flx=t['best-fit']
    noise=t['spec-bestfit']
    wave=t['wave']
    scfact=hdu[0].header['scl_fact']
    #print('My scf:',scfact)
    noise=noise*scfact
    #ivar mask
    #ivar=t['ivar']
    #iv_mask=(ivar>0)
    #noise=noise[iv_mask]
    #wave=wave[iv_mask]
   # print('MY_stdv: ',np.nanstd(noise))
    #Get the narrow and broad velocity dispersion
    vd=hdu[2].data
    vd_n=vd['velocity dispersion(Km/s)'][0]
    vd_b=vd['velocity dispersion(Km/s)'][1]
    #Calculate the Sigma coef
    c=299792.458 #in Km/s
    
    sig_in=1-exm*vd_n/c
    sig_fn=1+exm*vd_n/c
    sig_ib=1-exm*vd_b/c
    sig_fb=1+exm*vd_b/c

    #Complex center(extrem center) lines
    #---
    lmg=2798.75 #MgII
    #---
    lhd=4101.76 #H_delta
    lhg=4340.47 #H_gamma
    #---
    lhe=4687.02  #HeII
    lo3=5007    #OIII
    #---
    lha=6562.8  #H_alpha
    #Create the limits list 
    exl=-c
    exu=c
    lim=[exl,lmg*sig_ib,lmg*sig_fb,lhd*sig_ib,lhg*sig_fb,lhe*sig_ib,lo3*sig_fn,lha*sig_ib,lha*sig_fb,exu]

    #Loop to construvt the sim_noise
    for ct in range(len(lim)-1):
        scut=2
        if ct%2==0:
            mask=(wave>lim[ct])&(wave<=lim[ct+1])
            noise_c=noise[mask]
            
            size=len(noise_c)
            stvd=np.nanstd(noise_c)
            mask2=(noise_c>-scut*stvd)&(noise_c<scut*stvd)
            noise_c=noise_c[mask2]
            stvd=np.nanstd(noise_c)
            s_noise=np.random.normal(scale=stvd,size=size)
            if ct==0:
                sim_noise=s_noise
                
            else:
                sim_noise=np.append(sim_noise,s_noise)
                
            print(stvd,size)
        else:
            mask=(wave>lim[ct])&(wave<=lim[ct+1])
            noise_c=noise[mask]
            size=len(noise_c)
            stvd=np.nanstd(noise_c)
            mask2=(noise_c>-scut*stvd)&(noise_c<scut*stvd)
            noise_c=noise_c[mask2]
            stvd=np.nanstd(noise_c)
            s_noise=np.random.normal(scale=stvd,size=size)
            if ct==0:
                sim_noise=s_noise
               
            else:
                sim_noise=np.append(sim_noise,s_noise)
            #print(stvd,size)    
    ckn=np.isfinite(sim_noise)            
    nct=0
    for ck in range(len(ckn)):
        if ckn[ck]==False:
            sim_noise[ck]=sim_noise[ck-1]
            print('Nan found in sim_noise-using the neighboor val')
        nct+=1
            
    
    return sim_noise    
    
def noise_sim_direct(residual,wave,exm=2.0):
    '''exm: # of dispersion velocity wavelenght to construct the limits'''
    noise=residual #use in total flux units ([pp.galaxy-pp.bestfit]*scal_factor)
    #ivar mask
    #ivar=t['ivar']
    #iv_mask=(ivar>0)
    #noise=noise[iv_mask]
    #wave=wave[iv_mask]
   # print('MY_stdv: ',np.nanstd(noise))
    #Get the narrow and broad velocity dispersion
    #vd=hdu[2].data
    vd_n=300
    vd_b=3000
    #Calculate the Sigma coef
    c=299792.458 #in Km/s
    
    #sig_in=1-exm*vd_n/c
    sig_fn=1+exm*vd_n/c
    sig_ib=1-exm*vd_b/c
    sig_fb=1+exm*vd_b/c

    #Complex center(extrem center) lines
    lciv=1550 #Civ
    #---
    lmg=2798.75 #MgII
    #---
    lhd=4101.76 #H_delta
    lhg=4340.47 #H_gamma
    #---
    lhe=4687.02  #HeII
    lo3=5007    #OIII
    #---
    lha=6562.8  #H_alpha
    #Create the limits list 
    exl=-c
    exu=c
    lim=[exl,lciv*sig_ib,lciv*sig_fb,lmg*sig_ib,lmg*sig_fb,lhd*sig_ib,lhg*sig_fb,lhe*sig_ib,lo3*sig_fn,lha*sig_ib,lha*sig_fb,exu]

    #Loop to construvt the sim_noise
    for ct in range(len(lim)-1):
        scut=2
        if ct%2==0:
            mask=(wave>lim[ct])&(wave<=lim[ct+1])
            noise_c=noise[mask]
            
            size=len(noise_c)
            stvd=np.nanstd(noise_c)
            mask2=(noise_c>-scut*stvd)&(noise_c<scut*stvd)
            noise_c=noise_c[mask2]
            stvd=np.nanstd(noise_c)
            s_noise=np.random.normal(scale=stvd,size=size)
            if ct==0:
                sim_noise=s_noise
                
            else:
                sim_noise=np.append(sim_noise,s_noise)
                
            #print(stvd,size)
        else:
            mask=(wave>lim[ct])&(wave<=lim[ct+1])
            noise_c=noise[mask]
            size=len(noise_c)
            stvd=np.nanstd(noise_c)
            mask2=(noise_c>-scut*stvd)&(noise_c<scut*stvd)
            noise_c=noise_c[mask2]
            stvd=np.nanstd(noise_c)
            s_noise=np.random.normal(scale=stvd,size=size)
            if ct==0:
                sim_noise=s_noise
               
            else:
                sim_noise=np.append(sim_noise,s_noise)
            #print(stvd,size)    
    ckn=np.isfinite(sim_noise)            
    nct=0
    for ck in range(len(ckn)):
        if ckn[ck]==False:
            sim_noise[ck]=sim_noise[ck-1]
            print('Nan found in sim_noise-using the neighboor val')
        nct+=1
            
    
    return sim_noise    
    
