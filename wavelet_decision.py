#!/usr/bin/env python
###########################################################################
####### WAVELET DECISION FOR AMARU v0.1.0 ########################################
#This is a code by Santiago Bernal to find deviations on the residuals of a
#spectrum fit, is applied for SDSS spectra and results from pPXF v 9.4.1.
# In particular this is used with the code AMARU v0.1.0
#Method
#We use the Discrete Wavelet Transform with Haar functions to detect deviations
#in a four levels-scale
#Results
#The identification of deviations in a partícular wavelength ranges
#The decision of the need of a new and complex model to fit the spectrum
# version 0.0.2 (February 2025)
##############################################################################
##################################################################################################################
"""
    Copyright (C) 2025, Santiago Alejandro Bernal Galarza

    E-mail: sbernal@das.uchile.cl
            santiago.bernal.astro@gmail.com

    Updated versions of the software are available under request
    to the authors

    If you have found this software useful for your research,
    I would appreciate if you cite:

    https://ui.adsabs.harvard.edu/abs/2025......

    This software is provided as is without any warranty whatsoever.
    Permission to use, for non-commercial purposes is granted.
    Permission to modify for personal or internal use is granted,
    provided this copyright and disclaimer are included unchanged
    at the beginning of the file. All other rights are reserved.
    In particular, redistribution of the code is not allowed.

"""
##################################################################################################################

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import timeit
import glob
from scipy import stats
from astropy.io import fits
import os,sys
import traceback
import matplotlib.gridspec as gridspec
from collections import Counter
pd.set_option('future.no_silent_downcasting', True)
#Valor a tabular for th x0.75x,x1.00x,x1.25x,x1.50x,x1.75x,x2.00x,x2.25x,2.50
def dwt_deviations(devdf_dit,my_residuals,z,wave,ppsol,idx_o3m,idx_belm,ppgalaxy,ppbestfit,fit_mask,level=4,threshold_ps=3.50,plot_dev=0):
    '''
    -devdf_dit: directory to save the deviations dataframe
    -residuals: spectrum - best-fit, residuals of the fitting
    -z: redshift
    - wave: rest-frame wavelength array with the pixels values, same length than residuals
    - ppsol: solution array from pPXF ususally pp.sol (pp=ppxf(params*)), it contains the 
             kinematic moments (velocity, velocity dispersion, h3, h4) [km/s]
    - fit_mask: mask used to fit the spectrum
    - level: Decomposition level (must be >=4) for wavelet coefficients calculation. The algorithm use 4 levels
    - threshold_ps: factor for the threshold of high values in the power spectrum of the wavelet decomposition
                    the threshold is defined as:
                    threshold=threshold_ps*std(pw)*sqrt(2*ln(len(pw))) where pw: power spectrum
                    and Default threshold_ps=1.75
    -plot_dev: If =1 plot the residual and the position of where deviations on the pw were found
    '''
#################################################################################
    #regions of interest to extract properties from the spectra for AGNs
    vel_disp_bels=2000 #in km/s estimated for BEL regions #Valor a tabular
    vel_disp_nels=300 #in km/s NELs upper lim for regions #Valor a tabular
    c=299792.458 #in Km/s
    #Halpha
    Ha_lc=6565
    fHa=3*Ha_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Ha_ll=Ha_lc-fHa
    Ha_ul=Ha_lc+fHa
    Ha_reg=[Ha_ll,Ha_ul]
    #Hbeta 
    Hb_lc=4863
    fHb=3*Hb_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Hb_ll=Hb_lc-fHb
    Hb_ul=Hb_lc+fHb
    Hb_reg=[Hb_ll,Hb_ul]
    #[OIII]
    O3_lc=5007
    O3_slc=4958
    fO3=3*O3_lc*vel_disp_nels/c #wavelength width of 3 vel_disp
    O3_ll=O3_slc-fO3
    O3_ul=O3_lc+fO3
    O3_reg=[O3_ll,O3_ul]
    #Winds region
    WO3_ll=O3_slc
    WO3_ul=O3_lc+fO3*(1/3) #consider cases where the NELs-tem fit the wind and the residual contains NELs in the rigth side of the fitted NEL
    reg_winds=[WO3_ll,WO3_ul]
    #MgII
    Mg2_lc=2799
    if z>1:
        fMg2=3.5*Mg2_lc*vel_disp_bels/c #wavelength width of 3.5 vel_disp
    else:
        fMg2=3*Mg2_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Mg2_ll=Mg2_lc-fMg2
    Mg2_ul=Mg2_lc+fMg2
    Mg_reg=[Mg2_ll,Mg2_ul]
    #CIV
    C4_lc=1550
    if z>1:
        fC4=3.5*C4_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    else:
        fC4=3*C4_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    C4_ll=C4_lc-fC4
    C4_ul=C4_lc+fC4
    C4_reg=[C4_ll,C4_ul]
    ## For MgII and CIV larger regions are considered because of absorptions and outflows
    
    
    #Regions
    regions_lam=[Ha_lc,O3_lc,WO3_ul,Hb_lc,Mg2_lc,C4_lc]
    regions=[Ha_reg,O3_reg,reg_winds,Hb_reg,Mg_reg,C4_reg]
    regions_name=['Halpha','OIII','Wind_OIII','Hbeta','MgII','CIV']



#################################################################################
#####################Prepare the signal (residuals)##############################
#Use the fit_mask to avoid masked pixels for the analysis
    #print('Into residuals filling')
    residuals=my_residuals.copy()
    #To fill the masked pixels with the std value or a fraction of it
    fill_val=np.std(residuals[fit_mask])
    residuals[~fit_mask]=0.5*fill_val #Tabular_val
    #print('Finished residuals filling')
#################################################################################
#################################################################################
# Perform DWT using Haar wavelet
    levels=level
    wvm=wave
    #DWT
    coeffs = pywt.wavedec(residuals, 'haar', level=levels, mode='sp1')
    
    # Calculate the wavelet power spectrum (square of wavelet coefficients)
    power_spectrum = [np.square(coeff) for coeff in coeffs]
    
    # Define a threshold for significant deviations in the power spectrum
    # This can be based on a percentage of the maximum power or a fixed value
    threshold_v = threshold_ps #* max([max(power) for power in power_spectrum])  
    #threshold_v = 1.75 #Default selected in the publication .........
    # Create a dictionary to store significant deviations for each level
    significant_deviations = {}
    
    # Identify significant deviations and their positions for each level
    for i, power in enumerate(power_spectrum):
        threshold= threshold_v*np.std(power)*np.sqrt(2*np.log(len(power))) #This worked well 
        # if threshold_v=1 is from https://academic.oup.com/biomet/article-abstract/81/3/425/256924
        significant_indices = np.where(power > threshold)[0]  # Find where power exceeds the threshold
        significant_deviations[i] = significant_indices  # Store significant indices for the level
    
#################################################################################
#################################################################################
#Save deviation in a DF
    #Find regions in the rest-frame wavelength range of the spectrum
    mask_reg=np.array((regions_lam>wvm[0]+50) & (regions_lam<wvm[-1]-50)) #Valor a tabular
    regions=np.array(regions)
    regions2=regions[mask_reg]
    regions_name=np.array(regions_name)
    regions_name2=regions_name[mask_reg]
    #regions2,regions_name2
    count=[]
    # Create the DF
    for i in range(levels+1):
        for k in significant_deviations[i]:
            ix=(k/len(power_spectrum[i]))*len(residuals)
            wv_dev=wvm[int(ix)]
            for r,rns in zip(regions2,regions_name2):
                if wv_dev>r[0] and wv_dev<r[1]:
                  #  print('deviation in region '+rns+' at level'+str(i))
                    count.append([i,rns])
                    #break

    if len(count)==0:  #For no deviations to fill with 0
        index_arange=np.arange(0,levels+1,1)
        index_list=list(index_arange)
        deviations_df=pd.DataFrame(index=index_list,columns=regions_name2)
        deviations_df=deviations_df.fillna(0)
    else:
        df = pd.DataFrame(count, columns=['level', 'element'])    
    
        # Create a pivot table to count occurrences
        deviations_df = df.pivot_table(index='level', columns='element', aggfunc='size', fill_value=0)
        
        # Ensure columns include all elements
        deviations_df = deviations_df.reindex(range(5),columns=regions_name2, fill_value=0) #Change_by_len

    deviations_df.to_csv(devdf_dit,index=False)
#################################################################################
    comment_wavelet='' #used later to comment about residuals significance
#################################################################################

    
    
    # from deviations - broad
    br_dev_f=0
    # #####for winds 
    wo3_dev_flag=0 
    
    for i in range(3): #to consider levels 0,1,2 #Tabular_valcalcular
        if 'Wind_OIII' in regions_name2:        
            if deviations_df[['Wind_OIII']].iloc[i].values[0]>0: #Consider only "Wind" region in OIII
                wo3_dev_flag+=1
    
   
    #Number of regions with deviations
    
    num_reg_dev=0 #
    for ns in regions_name2:
        sum_dev=0
        if ns!='OIII' and ns!='Wind_OIII': #Consider only BELs
            sum_dev=deviations_df[ns].iloc[:3].sum() #To consider levels 0,1 and 2
            if sum_dev>0:
                num_reg_dev+=1
    #For just one region with deviations
    sig_one_reg=np.nan
    #Flags
    if num_reg_dev>1:  # Condition to detect deviations in at leas 2 regions          
        broad_dev_flags=1
        for i in range(3): #to consider levels 0,1 and 2 #Tabular_val_calcular
            for r in regions_name2:
                if deviations_df[[r]].iloc[i].values[0]>0 and r!='OIII' and r!='Wind_OIII': #Consider only BELs
                    br_dev_f+=1
            
    elif num_reg_dev==1:
        for r,rns in zip(regions2,regions_name2):
            if deviations_df[rns].sum()>0 and rns!='OIII' and rns!='Wind_OIII': #Change for the region where the deviation were found
                wvm_rr_mask=(wvm>r[0]) & (wvm<r[1])
                wv_rr=wvm[wvm_rr_mask]
                residuals_rr=residuals[wvm_rr_mask]
                residuals_rr_abs=np.absolute(residuals_rr)
                flux_rr_res=np.trapezoid(residuals_rr_abs, x=wv_rr) #residual flux in the region range
                std_residuals=np.std(residuals_rr)
                #######
                std_flux=std_residuals*np.sqrt(len(residuals_rr))*(wv_rr[1]-wv_rr[0]) #Uncertanty in the integrated residual flux
                significance_rr=np.absolute(flux_rr_res)/std_flux #residual flux significance
                print(f'region {rns} with residual significance {significance_rr}')
                
                if significance_rr>10: #10 is a conserative value #Valor a tabular
                    comment_wavelet=comment_wavelet+' region '+str(rns)+' with res-sig '+str(round(significance_rr,2))
                    sig_one_reg=round(significance_rr,2)
                    broad_dev_flags=1
                    br_dev_f=4
                   
                else:
                    broad_dev_flags=0
    
    else:
        broad_dev_flags=0

                    
    
#################################################################################
#################################################################################
# from deviations - narrow
    nr_dev_flag=0
    o3n_dev_flag=0
    
    for i in range(2,5): #to consider levels 2,3,4 #Tabular_val_calculated
        for r in regions_name2:
            if deviations_df[[r]].iloc[i].values[0]>0 and r!='Wind_OIII':
                nr_dev_flag+=1
        if 'OIII' in regions_name2:
            if deviations_df[['OIII']].iloc[i].values[0]>0: #To later consider the OIII vels
                o3n_dev_flag+=1
        
    
    if nr_dev_flag>=2:
        narrow_dev_flags=1 #For narrow velocity
    else:
        narrow_dev_flags=0
#################################################################################
#################################################################################
#Significance for wind
    significance_wo3=-999
    if 'Wind_OIII' in regions_name2:
        wvm_o3_mask=(wvm>reg_winds[0]) & (wvm<reg_winds[1])
        wv_o3=wvm[wvm_o3_mask]
        residuals_o3=residuals[wvm_o3_mask]
        flux_o3_res=np.trapezoid(residuals_o3, x=wv_o3) #residual flux in the wind_O3 range
        std_residuals=np.std(residuals_o3)
        std_flux=std_residuals*np.sqrt(len(residuals_o3))*(wv_o3[1]-wv_o3[0]) #Uncertanty in the integrated residual flux
        significance_wo3=np.absolute(flux_o3_res)/std_flux #residual flux significance
    else:
        significance_wo3=0.0
#################################################################################
#################################################################################
#Decision
    #For broad regions:
    case_bels=0 #To save which case is the one indicating the 2BELs model
    #- Check if h3 and h4 moments are large
    h3_f=ppsol[idx_belm][2]  #index are from the model used to fit, related with the components indexing
    h4_f=ppsol[idx_belm][3]  #
    if (np.abs(h3_f)>0.05 and np.abs(h4_f)>0.04) or (np.abs(h4_f)>0.05 and np.abs(h3_f)>0.04): #Valor a tabular #Tabular_val
        ng_mom_flag=1
    else:
        ng_mom_flag=0
    #add a second set of BELs
    if  ng_mom_flag==1 and broad_dev_flags==1: #This is good for few deviations but in more than one region
        add_2bels=1
        case_bels=1
        #print('Broad deviations case1')
    #For deviations in a BEL region o BEL regions, considering that 3 or more deviations in one region can 
    #indicate a problem, later the comparison between the fit with the new model can determine
    #which is better
    elif br_dev_f>=4:      #Tabular_val_calculated            
        add_2bels=1
        #For checking the case of 2BELs decision
        if num_reg_dev>1:
            case_bels=2
        else:
            case_bels=3
    else:
        add_2bels=0
    
    #add winds
    case_winds=0 #To save the case where the winds are added
    if 'Wind_OIII' in regions_name2:
    #using the flux significance and deviations to avoid small narrow deviations mixed with residuals
     # significance, recomended 3 but winds could be small in residuals because of NELS, then using sig>2
        if significance_wo3>2.5 and wo3_dev_flag>0: #Valor a tabular #Tabular_val
            add_wind=1
            case_winds=1
        elif (ppsol[idx_o3m][1]>=300 and wo3_dev_flag>=0): #Valor a tabular #Tabular_val
            # new condition of vel_disp for NEL-templates that fit the wind and not the NELs in [OIII]
            #Note that the limit of vel disp in the model is 400km/s but 300km/s can mimic NELs+Winds(low vel)
            #Again the index 1 in ppsol comes from the indexing of templates for pPXF fitting
            add_wind=1
            case_winds=2
        elif (ppsol[idx_o3m][1]>=300 and significance_wo3>2.5): #Valor a tabular #Tabular_val_calculated
            add_wind=1
            case_winds=3
        else:
            add_wind=0
    else:
        add_wind=0
    
    models_add=['2Bel','2Bel_winds','winds','no_new_model']
    
    if add_2bels==1 and add_wind==0:
        new_model_index=0
    elif add_2bels==1 and add_wind==1:
        new_model_index=1
    elif add_2bels==0 and add_wind==1:
        new_model_index=2
    else:
        new_model_index=-1 #means no new model
#################################################################################
#################################################################################
#Decision about initial velocity narrow lines
    if o3n_dev_flag+wo3_dev_flag>0:
        wvm_o3n_mask=(wvm>O3_reg[0]) & (wvm<O3_reg[1])
        wv_o3n=wvm[wvm_o3n_mask]
        gal_o3=ppgalaxy[wvm_o3n_mask]
        bsft_o3=ppbestfit[wvm_o3n_mask]
        max_gal_idx=list(gal_o3).index(max(gal_o3))
        max_bstf_idx=list(bsft_o3).index(max(bsft_o3))
        c = 299792.458
        dv_nr = np.log(wv_o3n[max_gal_idx]/wv_o3n[max_bstf_idx])*c
        if  np.absolute(dv_nr)>100: #Valor a tabular #Tabular_val
            new_inti_vel_narrow=ppsol[idx_o3m][0]+dv_nr
            
            print(f'new init vel for NELs from {ppsol[idx_o3m]}')
            flag_vel_o3=1
        else:
            flag_vel_o3=0
            new_inti_vel_narrow=ppsol[idx_o3m][0]
    else:
        dv_nr=0
        flag_vel_o3=0
        new_inti_vel_narrow=ppsol[idx_o3m][0]
        
    
#################################################################################
#################################################################################
#Add recomendation if no_new_model and flux residuals have a large significance
    #Added because the rutine don't know how recognize bad NELs fitted with incorrect-high flux
    
    if models_add[new_model_index]=='no_new_model':
        for r,rns in zip(regions2,regions_name2):
            if deviations_df[rns].sum()>0:
                wvm_rr_mask=(wvm>r[0]) & (wvm<r[1])
                wv_rr=wvm[wvm_rr_mask]
                residuals_rr=residuals[wvm_rr_mask]
                flux_rr_res=np.trapezoid(residuals_rr, x=wv_rr) #residual flux in the region range
                std_residuals=np.std(residuals_rr)
                std_flux=std_residuals*np.sqrt(len(residuals_rr))*(wv_rr[1]-wv_rr[0]) #Uncertanty in the integrated residual flux
                significance_rr=np.absolute(flux_rr_res)/std_flux #residual flux significance     
                if significance_rr>2.5: #Valor a tabular
                    print(f'Check region {rns} with residual significance {significance_rr}')
                    comment_wavelet=comment_wavelet+' Check region'+str(rns)

#################################################################################
#################################################################################
#Add the residual significance of each region
    residuals_significance_regs=[]
    for r,rns in zip(regions2,regions_name2):
        wvm_rr_mask=(wvm>r[0]) & (wvm<r[1])
        wv_rr=wvm[wvm_rr_mask]
        residuals_rr=residuals[wvm_rr_mask]
        flux_rr_res=np.trapezoid(residuals_rr, x=wv_rr) #residual flux in the region range
        std_residuals=np.std(residuals_rr)
        std_flux=std_residuals*np.sqrt(len(residuals_rr))*(wv_rr[1]-wv_rr[0]) #Uncertanty in the integrated residual flux
        significance_rr=np.absolute(flux_rr_res)/std_flux #residual flux significance
        residuals_significance_regs.append(significance_rr)

    sig_regs=pd.DataFrame()
    sig_regs['region']=regions_name2
    sig_regs['residual_sig']=residuals_significance_regs
    #####
    rgsn1=devdf_dit.split(sep='idf-')
    #Used to save the deviations DataFrame
    sig_regs.to_csv('../residuals_significance/res_sig-'+rgsn1[-1],index=False)
#################################################################################
#################################################################################
    if plot_dev==1:
        dvc=0
        for i in range(3):
            for k in significant_deviations[i]:
                ix=(k/len(power_spectrum[i]))*len(residuals)
                if dvc==0:
                    plt.axvline(x=wvm[int(ix)], color='green', linestyle='--', label=f'l=0-3')
                    dvc=1
                else:
                    plt.axvline(x=wvm[int(ix)], color='green', linestyle='--')
        dvc2=0
        for i in range(3,5):
            for k in significant_deviations[i]:
                ix=(k/len(power_spectrum[i]))*len(residuals)
                if dvc2==0:
                    plt.axvline(x=wvm[int(ix)], color='grey', linestyle='--', label=f'l=3-4')
                    dvc2=1
                else:
                    plt.axvline(x=wvm[int(ix)], color='grey', linestyle='--')
        plt.plot(wvm,residuals,c='k',label='residuals')
        plt.legend()
        dir_save_deviations_plot='./deviations.png'
        plt.savefig(dir_save_deviations_plot)
        
    plt.close('all')
#################################################################################

#################################################################################
#################################################################################

#Returning decision and initial vel recomendation
    return models_add[new_model_index],flag_vel_o3, new_inti_vel_narrow, comment_wavelet,sig_one_reg,significance_wo3,dv_nr,\
            case_winds,case_bels
    

#################################################################################

#To create only the regions
def my_regions(lam_range):
    #regions of interest to extract properties from the spectra for AGNs
    vel_disp_bels=2000 #in km/s estimated for BEL regions
    vel_disp_nels=300 #in km/s NELs upper lim for regions
    c=299792.458 #in Km/s
    #Halpha
    Ha_lc=6565
    fHa=3*Ha_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Ha_ll=Ha_lc-fHa
    Ha_ul=Ha_lc+fHa
    Ha_reg=[Ha_ll,Ha_ul]
    #Hbeta 
    Hb_lc=4863
    fHb=3*Hb_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Hb_ll=Hb_lc-fHb
    Hb_ul=Hb_lc+fHb
    Hb_reg=[Hb_ll,Hb_ul]
    #[OIII]
    O3_lc=5007
    O3_slc=4958
    fO3=3*O3_lc*vel_disp_nels/c #wavelength width of 3 vel_disp
    O3_ll=O3_slc-fO3
    O3_ul=O3_lc+fO3
    O3_reg=[O3_ll,O3_ul]
    
    #MgII
    Mg2_lc=2799
    fMg2=3*Mg2_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    Mg2_ll=Mg2_lc-fMg2
    Mg2_ul=Mg2_lc+fMg2
    Mg_reg=[Mg2_ll,Mg2_ul]
    #CIV
    C4_lc=1550
    fC4=3*C4_lc*vel_disp_bels/c #wavelength width of 3 vel_disp
    C4_ll=C4_lc-fC4
    C4_ul=C4_lc+fC4
    C4_reg=[C4_ll,C4_ul]
    ##
    #Regions
    regions_lam=[Ha_lc,O3_lc,Hb_lc,Mg2_lc,C4_lc]
    regions=[Ha_reg,O3_reg,Hb_reg,Mg_reg,C4_reg]
    regions_name=['Halpha','OIII','Hbeta','MgII','CIV']

    mask_reg=np.array((regions_lam>lam_range[0]+50) & (regions_lam<lam_range[1]-50))
    regions=np.array(regions)
    regions2=regions[mask_reg]
    regions_name=np.array(regions_name)
    regions_name2=regions_name[mask_reg]

    return regions2, regions_name2

