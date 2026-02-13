########################################### AMARU_v0.2.1
#A.M.A.R.U. – "Automatic Multiscale Analysis for Recommendation of AGN Models"
#Santiago A. Bernal G. August 2025----- AMARU version 0.2.0
#This version was an update of the first algorithm and is used
# in the publication of the article: Automated model selection for the spectral fitting of large samples of active galactic nucleus spectra
#ADS: 
###############################################
#     █████╗ ███╗   ███╗ █████╗ ██████╗ ██╗   ██╗
#    ██╔══██╗████╗ ████║██╔══██╗██╔══██ ██║   ██║
#    ███████║██╔████╔██║███████║██████╔╝██║   ██║
#    ██╔══██║██║╚██╔╝██║██╔══██║██  ██║ ██╚═══██║
#    ██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║  █████╔╝
#    ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚════╝    

#  A U T O M A T I C   M U L T I S C A L E
#     A N A L Y S I S   F O R   T H E
#   R E C O M M E N D A T I O N   O F 
#        A G N   M O D E L S

##################################################################################################################
##################################################################################################################
#El “Amaru” o serpiente cósmica ... la serpiente simboliza el principio de la vida, el alma, la libido y la fecundidad, también las fuerzas #opuestas complementarias de la naturaleza, la sabiduría y el conocimiento. Este animal totémico es el representante del “Ukhu pacha” o mundo #interior, subconciente, del mundo andino, el mundo subterráneo, que está “dentro y debajo". (Cosmovisión andina, Jym Qhapaq Amaru, 2012).
#El "Amaru" también representa la conexión entre el cielo y la tierra.
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
"""
    Copyright (C) 2025, Santiago Alejandro Bernal Galarza

    E-mail: sbernal@das.uchile.cl
            santiago.bernal.astro@gmail.com

    Updated versions of the software are available under request
    to the authors

    If you have found this software useful for your research,
    I would appreciate if you cite:

    Title: Automated model selection for the spectral fitting of large samples of active galactic nucleus spectra
    

    This software is provided as is without any warranty whatsoever.
    Permission to use, for non-commercial purposes is granted.
    Permission to modify for personal or internal use is granted,
    provided this copyright and disclaimer are included unchanged
    at the beginning of the file. All other rights are reserved.
    In particular, redistribution of the code is not allowed.

"""
##################################################################################################################
##################################################################################################################
#INPUT 
######### Required parameters  ##########
#     file_dir: Directory of the .fits archive. Must be an SDSS type .fits

#     EMILES_Dir: Directory of the E-MILES stellar population templates. You can donwload them 
#                and more information from 
#                https://research.iac.es/proyecto/miles/pages/spectral-energy-distributions-seds/e-miles.php
#                and cite: UV-extended E-MILES stellar population models: young components in massive early-type 
#                galaxies, Vazdekis, A.; Koleva, M.; Ricciardelli, E.; Röck, B.; Falcón-Barroso, J.
#                2016MNRAS.463.3409V

#     fe2_temps: Directory of the FeII templates. You can donwload them from
#                https://drive.google.com/drive/folders/1NT7dRKSnumhc1OxYtw0E8CDgrUYCr4aw?usp=sharing
#                https://drive.google.com/drive/folders/1P_Vv3mBySidM5VtGugyPGwpNYOn2ZnN_?usp=sharing

#     Balmer_tem: Directory of the FeII templates. You can donwload them from
#                 https://drive.google.com/drive/folders/1DAn3gwDXhcptkG-Q7SwdAKkehuIFrxmj?usp=sharing
#
#NOTE: If the links are not working you can request them by email.

######### Optional parameters  ##########
#     lwl and upl: are the lower and upper limits for the observed wavelentgh that is going to be used to
#                  produce the spectrum fit. Default values are lwl=3800 and upl=10400

#     islope and fslope: are the lowerand upper limit values for the power-law set of templates. Default 
#                        values are islope=-3 and fslope=0

#     ivar_lim: limit value of the inverse variance from SDSS archive to make a mask. Values close to zero
#               are pixels with large error. Default value is ivar_lim=0.005

#     zz: Optional redshift value. It is used only when is provided and replace the redshift from the .fits
#         archive. Useful when the redshift is incorrect in the .fits archive

#     st_func:  If st_func=0 use models with stellar templates, If st_func=1 use models without stellar templates
#               When z>1, st_func is fixed to st_func=1

#     str_op: When st_func=0; if str_op=0 uses all the templates in the EMILES_Dir. If str_op!=0, you need to 
#             pass list_st=[array] as an array with the directories of the stellar populations or individual 
#             templates that you whant to inclde in the model. Useful when to limit the number of stellar 
#             templates.

#     model_name: Use this parameter if you want to start with a model different from the initial one suggested
#                 The options are in the function <<model_constructor>>

#     nnels_comp: If nnels_comp=1 all the Narrow Emission Lines (NELs) with the same velocity. If nnels_comp=2 
#                divide the NELs in two set at the 6000A in observed wavelength, this allows to fit better the 
#                NELs when there is error i the wavelength, which is observed in some cases. In general the
#                resulting velocity of the two sets is the same or with a maximun diference of ~100km/s, but 
#                assuring a good fit of the NELs.

#    my_plot_dir: directory where to save the plots

#    plot_model_ini: If equal to 1, the plot of the fit using the initial model is saved (used for comparison)

#    plot_best: If equal True, the plot of the fit using the best model is saved (used for comparison)

#    devdf_dir: directory where to save the tables containing the number of deviations.

#    save_op: If save_op=1 the results of the best-fit are saved in .fits archive

#    nsim: Number of the Monte Carlo simulations to constraint the errors. Recommended number is >=50. 
#          When nsim>0, nsim fits are performed using the best-fit + random error, each fit is
#          saved in a .fits archive. 
##################################################################################################################
##################################################################################################################
##################################################################################################################

from __future__ import print_function

from astropy.io import fits 
import numpy as np
import glob,timeit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ppxf import ppxf 
import ppxf_util_al_lab_wv as util #modified and provided
from astropy.table import Table
import os
import sys
from scipy.stats import f
#For dereddening
from sfdmap2 import sfdmap
import extinction
### AMARU packeges
import noise_sim as nosim #Used for Monte Carlo Simulations
from two_copm_fwhm_interpol import int_fwhm as fwhm_tc #Used to get the FWHM or velocity dispersion of a two component BELs
from wavelet_decision import dwt_deviations,my_regions #Here the use of the DWT is implemented



class amaru:


    def __init__(self,file_dir, lwl=3800,upl=10400,islope=-3,fslope=0,str_op=0,ivar_lim=0.005,sdfmaps=None,
                 zz=None,list_st=None,save_op=0,nsim=None,EMILES_Dir=None,
                 fe2_temps=None,Balmer_tem=None,st_func=0,nnels_comp=2,model_name=None,plot_model_ini=0,
                 my_plot_dir=None,plot_best=True,devdf_dir='../dev'):


        self.file_dir=file_dir
        self.lwl=lwl
        self.upl=upl
        self.islope=islope
        self.fslope=fslope
        self.str_op=str_op
        self.ivar_lim=ivar_lim
        self.zz=zz
        self.list_st=list_st
        self.save_op=save_op
        self.nsims=nsim
        self.EMILES_Dir=EMILES_Dir
        self.fe2_temps=fe2_temps
        self.Balmer_tem=Balmer_tem
        self.model_name=model_name
        self.best_model=model_name
        self.time_pp2=0
        self.time_compar=0
        self.devdf_dir=devdf_dir
        #Read the data
        print(file_dir)
        self.file_reading(file_dir,lwl,upl,ivar_lim,my_sdfmaps=sdfmaps)
        cond_plot2=0
        #### Choose a model name
        if model_name is None:
            if self.z<1:
                model_name='model_st_1'
            else:
                model_name='model_nst_1'
                st_func=1
        if self.z>1:
            st_func=1
            
        ####################### Model #####################
        start_time = timeit.default_timer() #start time to check
        
        #Use the stellar templates
        if st_func==0:
            self.stellar_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op,list_st)
        else:
            self.set_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op,list_st)
        #Create the FeII templates
        self.feii_temps(fe2_temps,self.Templates_size,self.lamRange_temp,self.velscale)
        #Create the Balmer templates
        self.balmer_templates(Balmer_tem,self.Templates_size,self.lamRange_temp,self.velscale)
        #Create gas templates
        #Condition for NELs
        if nnels_comp==1:
            self.gas_templates, self.line_names, self.line_wave=self.nels_templates(np.min(self.lam_gal),np.max(self.lam_gal),self.logLam2)
        if nnels_comp==2:
            wave_sep=6000/(1+self.z) #based on the resolution of BOSS spectrograph
            if wave_sep<1250:
                wave_sep=1250
            self.gas_templates1, self.line_names1, self.line_wave1=self.nels_templates(np.min(self.lam_gal),wave_sep,self.logLam2)
            self.gas_templates2, self.line_names2, self.line_wave2=self.nels_templates(wave_sep,np.max(self.lam_gal),self.logLam2)
            self.gas_templates=[self.gas_templates1,self.gas_templates2]
           
        # BELS
        self.bels_templates(self.lam_gal,self.logLam2)

        #Civ BELS
        self.civ_templates_Bels(self.lam_gal,self.logLam2)
        #winds
        self.my_wind_templates(self.lam_gal,self.logLam2)
        #AGN
        self.my_agn_templates(self.logLam2,islope,fslope)                
        #print(len(self.stars_templates))
        
        #Model constructor for the first model (usually model_*_1)
        self.model_constructor(model_name,self.stars_templates,self.gas_templates,self.Broad_templates,self.civ_templates,self.wind_templates,
                               self.bc_templates, self.fe_templates,self.agn_templates,self.z)

        
        end_time = timeit.default_timer()  #end time to check
        self.time_model1=np.round(end_time-start_time)      #time of code section
        ###############pPXF running ##############
        print('using: '+model_name)
        start_time = timeit.default_timer() #start time to check
         #-----------
        try:
            self.pp = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
              plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
              vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
              bias=0,quiet=True)
        except:
            time.sleep(1)
            self.pp = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
              plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
              vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
              bias=0,quiet=True)

        self.chi21=self.pp.chi2
        print('pp1_chi2',self.chi21)
        end_time = timeit.default_timer()  #end time to check
        self.time_pp1=np.round(end_time-start_time)      #time of code section
        print('Time pp1=',self.time_pp1)
        ################# My initial chi2 #######
        residuals1=(self.pp.galaxy-self.pp.bestfit)*self.scale_factor
        residuals1=residuals1[self.iv_mask]
        uncer=self.noise[self.iv_mask]
        my_dof=sum(self.iv_mask)
        res_dstd1=residuals1/uncer
        self.my_ini_chi2=(res_dstd1@res_dstd1)/my_dof
        
        ################# Plot First model #######
        # Only for comparison and visual inspection to test the method
        if plot_model_ini==1:
            self.plot_fit_model(file_dir,self.z,model_name,self.pp,self.lam_gal,self.nTemps,self.nLines1,
                                self.nLines2,self.nBlines,self.nClines,self.nwinds,
                            self.nBc,self.nFe,self.nAgns,option=5,save_fig_dir=my_plot_dir,ini=1)       
            #--------
         ################# Save results model-1 first fit to compare #######
    
        if save_op==1: 
            self.fits_constructor(file_dir,self.z,self.scale_factor,model_name,self.pp,self.ivar,self.lam_gal,self.wdisp,
                              self.vazdekis,self.nTemps,
                              self.line_names1, self.line_wave1,self.nLines1,self.line_names2,self.line_wave2,self.nLines2,
                              self.Bline_names, self.Bline_wave,self.nBlines,
                              self.nClines, self.Cline_names,self.Cline_wave,
                              self.wind_line_names, self.wind_line_wave,self.nwinds,
                              self.nBc,self.nFe,self.slopes,self.nAgns,self.iv_mask,wts=2,fit_type=None,units=None,my_chi2=self.my_ini_chi2)

        
        ################## Wavelet method ##########
        start_time = timeit.default_timer() #start time to check
        #Find the index for [OIII] needed into dwt_deviations
        if '[OIII]5007d' in np.array(self.line_names1):
            self.idx_o3=1
        else:
            self.idx_o3=2
        #Find the index if z>0.8 (no stellar temps)
        if st_func==1:
            self.idx_o3=self.idx_o3-1
            self.idx_bels=2
        else:
            self.idx_bels=3
            
        #-------------
        self.residuals=self.pp.galaxy-self.pp.bestfit
        self.new_model,self.flag_vel_o3, self.new_inti_vel_narrow,self.comment_wavelet,\
        self.sig_one_reg,self.wo3_sig,self.dv_nr,self.case_winds,self.case_bels=dwt_deviations(self.devdf_dir,self.residuals,
                                                                                self.z,self.lam_gal,self.pp.sol,self.idx_o3,
                                                                                 self.idx_bels,self.pp.galaxy,self.pp.bestfit,
                                                                                 self.iv_mask,level=4,threshold_ps=1.25,plot_dev=1)
        
        end_time = timeit.default_timer()  #end time to check
        #print('DWT finished')
        self.time_dwt=np.round(end_time-start_time)      #time of code section
        ##################Action after decision ######
        
            
        #-----------
        # decision is no new model    
        if self.new_model=='no_new_model':
            print('DWT suggestion: no_new_model')
            self.best_pp=self.pp
            self.best_model=model_name
            
            
            if '[OIII]5007d' in self.gas_templates1:
                o3gc=1
            elif '[OIII]5007d' in self.gas_templates2:
                o3gc=2
            else:
                #print('No [OIII]5007d in the wavelengt range')
                o3gc=None
#Check for new init vel suggestion
            if self.flag_vel_o3!=0 and o3gc!=None and np.absolute(self.new_inti_vel_narrow-self.pp.sol[o3gc][0])>50:
                    
                #Use the stellar templates fitted in the first model
                #As here we are not interested in ssp properties we assume that ssp selected in the first fit
                #are enough to obtain a good fit of the stellar component, also check the value of mdegree
                if st_func==0:
                    mask_ssp_2md=(self.pp.weights[:self.nTemps]>0)
                    self.list_st_2md=self.vazdekis[mask_ssp_2md]
                    if len(self.list_st_2md)>2:
                        str_op_2md=2
                        self.stellar_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op_2md,self.list_st_2md)
            #Same model but change in initial velocity of [OIII]d5007 if present    
                self.model_constructor(model_name,self.stars_templates,self.gas_templates,
                                           self.Broad_templates,self.civ_templates,self.wind_templates,self.bc_templates,
                                 self.fe_templates,self.agn_templates,self.z,vel_o3=self.new_inti_vel_narrow,o3gc=o3gc)
                 ###############pPXF running ##############
                start_time = timeit.default_timer() #start time to check

                #-----------#########
                    
                try:
                    self.pp2 = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
                except:
                    time.sleep(1)
                    self.pp2 = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
                self.chi22=self.pp2.chi2
    
                end_time = timeit.default_timer()  #end time to check
                self.time_pp2=np.round(end_time-start_time)      #time of code section
                ################# Comparison ##################
        #------------------------
                start_time = timeit.default_timer() #start time to check

                self.comparison_two_fits(self.pp.galaxy-self.pp.bestfit,self.pp.dof,self.pp2.galaxy-self.pp2.bestfit,self.pp2.dof,
                                         self.lam_gal,self.iv_mask,cp_ty=0)
                print(f'comparison score:{self.comparison_sc}')
                if self.comparison_sc>1:
                    print('Desicion:'+model_name+' init_vel_mod')
                    self.best_pp=self.pp2
                    cond_plot2=1
                    self.best_model=model_name
                else:
                    print('Desicion:'+model_name)
                    self.best_pp=self.pp
                    self.best_model=model_name
        
                end_time = timeit.default_timer()  #end time to check
                self.time_compar=np.round(end_time-start_time)      #time of code section        
                #------------------------
            else:
                self.comparison_sc=-999
                self.chi22=np.nan
        else: #Here the new model and the comparison are computed
            print('DWT suggestion:'+self.new_model)
            models_suggestion=['2Bel','winds','2Bel_winds']
            models_to_const_st=['model_st_2','model_st_3','model_st_4']
            models_to_const_nst=['model_nst_2','model_nst_3','model_nst_4']
            idxmodel=models_suggestion.index(self.new_model)
            if st_func==0:
                model_name2=models_to_const_st[idxmodel]
                bel_vel=self.pp.sol[3][0]
            else:
                model_name2=models_to_const_nst[idxmodel]
                bel_vel=self.pp.sol[2][0]
            if self.flag_vel_o3!=0:
                new_init_vel_o3=self.new_inti_vel_narrow
                if '[OIII]5007d' in self.gas_templates1:
                    o3gc=1
                elif '[OIII]5007d' in self.gas_templates2:
                    o3gc=2
                else:
                    #print('No [OIII]5007d in the wavelengt range')
                    o3gc=None
            else:
                new_init_vel_o3=None
                o3gc=None

            #Use the stellar templates fitted in the first model
            #As here we are not interested in ssp properties we assume that ssp selected in the first fit
            #are enough to obtain a good fit of the stellar component, also check the value of mdegree
            if st_func==0:
                mask_ssp_2md=(self.pp.weights[:self.nTemps]>0)
                self.list_st_2md=self.vazdekis[mask_ssp_2md]
                if len(self.list_st_2md)>2:
                    str_op_2md=2
                    self.stellar_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op_2md,self.list_st_2md)
                        
            #New model considering init vel suggestion        
            self.model_constructor(model_name2,self.stars_templates,self.gas_templates,
                                           self.Broad_templates,self.civ_templates,self.wind_templates,self.bc_templates,
                                 self.fe_templates,self.agn_templates,self.z,vel_o3=new_init_vel_o3,o3gc=o3gc,bel_vel=bel_vel)
            ###############pPXF running ##############
            start_time = timeit.default_timer() #start time to check
            
            try:
                self.pp2 = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
            except:
                time.sleep(1)
                self.pp2 = ppxf(self.templates, self.galaxy, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
                
            self.chi22=self.pp2.chi2
            
            end_time = timeit.default_timer()  #end time to check
            self.time_pp2=np.round(end_time-start_time)      #time of code section
            print('Finished pp2')
            ################# Plot best model 2####### Clean this part is only for test
            cond_plot22=10
            if self.new_model!='no_new_model' and cond_plot22==0:
                self.plot_fit_model(file_dir,self.z,model_name2,self.pp2,self.lam_gal,self.nTemps,self.nLines1,
                            self.nLines2,self.nBlines,self.nClines,self.nwinds,
                        self.nBc,self.nFe,self.nAgns,option=5,save_fig_dir=my_plot_dir,ini=2) 
            ################# Comparison ##################
        #------------------------
        #------------------------
            start_time = timeit.default_timer() #start time to check
            self.comparison_two_fits(self.pp.galaxy-self.pp.bestfit,self.pp.dof,self.pp2.galaxy-self.pp2.bestfit,self.pp2.dof,
                                         self.lam_gal,self.iv_mask,cp_ty=0)
        #-------
            print('finished comparison')
        ############# Score result and best model decision #################
        print(f'comparison score:{self.comparison_sc}')
        if self.comparison_sc>=0.5:
            print('Desicion:'+model_name2)
            self.best_model=model_name2
            self.best_pp=self.pp2
            cond_plot2=1
        else:
            print('Desicion:'+model_name)
            self.best_model=model_name
            self.best_pp=self.pp
            # The model values and all templates need to be reconstructed to use the save function, it is no needed to fit again
            if st_func==0:
                self.stellar_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op,list_st)
            else:
                self.set_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op,list_st)
            self.model_constructor(model_name,self.stars_templates,self.gas_templates,self.Broad_templates,
                                               self.civ_templates,self.wind_templates,self.bc_templates,
                             self.fe_templates,self.agn_templates,self.z)
        
        end_time = timeit.default_timer()  #end time to check
        self.time_compar=np.round(end_time-start_time)      #time of code section
        #--------
        ################# My best chi2 #######
        best_residuals=(self.best_pp.galaxy-self.best_pp.bestfit)*self.scale_factor
        best_residuals=best_residuals[self.iv_mask]
        uncer=self.noise[self.iv_mask]
        my_dof=sum(self.iv_mask)
        res_dstd=best_residuals/uncer
        self.my_chi2=(res_dstd@res_dstd)/my_dof
        #--------
        ################# Plot best model #######
        if plot_best==True and cond_plot2==1:
            self.plot_fit_model(file_dir,self.z,self.best_model,self.pp2,self.lam_gal,self.nTemps,self.nLines1,
                            self.nLines2,self.nBlines,self.nClines,self.nwinds,
                        self.nBc,self.nFe,self.nAgns,option=5,save_fig_dir=my_plot_dir)   

        
        #--------
        ################# Check if comparison was performed #######    
        # For test
        if hasattr(self, 'comparison_sc'):
            self.comp_re=1
        else:
            self.comp_re=0
        #--------
        ################# Save results #######
        start_time = timeit.default_timer() #start time to check
    
        if save_op==1: 
            print('starting save')
            self.fits_constructor(file_dir,self.z,self.scale_factor,self.best_model,self.best_pp,self.ivar,self.lam_gal,self.wdisp,
                              self.vazdekis,self.nTemps,
                              self.line_names1, self.line_wave1,self.nLines1,self.line_names2,self.line_wave2,self.nLines2,
                              self.Bline_names, self.Bline_wave,self.nBlines,
                              self.nClines, self.Cline_names,self.Cline_wave,
                              self.wind_line_names, self.wind_line_wave,self.nwinds,
                              self.nBc,self.nFe,self.slopes,self.nAgns,self.iv_mask,wts=0,fit_type=None,units=None,my_chi2=self.my_chi2)
    
        end_time = timeit.default_timer()  #end time to check
        self.time_save=np.round(end_time-start_time)      #time of code section

            #--------
        ################# Simulations #######
        if nsim!=None:
            if nsim>0:
                print(f'Performing {nsim} simulations with '+self.best_model)
                #Best fit spectrum
                flux_pp=self.best_pp.bestfit*self.scale_factor
                #residuals
                residuals_fs=(self.best_pp.galaxy-self.best_pp.bestfit)*self.scale_factor
                #Use only ssp from best-fit model
                if st_func==0:
                    mask_ssp_sim=(self.best_pp.weights[:self.nTemps]>0)
                    self.list_st_sim=self.vazdekis[mask_ssp_sim]
                    if len(self.list_st_sim)>0:
                        str_op_sim=2
                        self.stellar_temp(EMILES_Dir,self.lam_gal,self.velscale,self.fwhm_gals,str_op_sim,self.list_st_sim)
                        self.model_constructor(self.best_model,self.stars_templates,self.gas_templates,self.Broad_templates,
                                                   self.civ_templates,self.wind_templates,self.bc_templates,
                                 self.fe_templates,self.agn_templates,self.z)
                start_time = timeit.default_timer() #start time to check
                for j in range(nsim):
                    print(f'performing simulation {j} of {nsim}')
                    #produce the simulated flux
                    
                    sim_noise=nosim.noise_sim_direct(residuals_fs,self.lam_gal)
                    flux_sim=flux_pp+sim_noise
                    self.scale_fact_sim=np.median(flux_sim)
                    galaxy_sim=flux_sim/self.scale_fact_sim
                    # run pPXF
                    try:
                        self.pp_sim = ppxf(self.templates, galaxy_sim, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
                    except:
                        time.sleep(1)
                        self.pp_sim = ppxf(self.templates, galaxy_sim, self.noise, self.velscale, self.start,
                      plot=False, moments=self.moments, degree=-1, mdegree=-1,mask=self.iv_mask,
                      vsyst=self.dv, clean=True, component=self.component, bounds=self.bounds,tied=self.tied,
                      bias=0,quiet=True)
                    #Save the simulation
                    self.fits_constructor(file_dir,self.z,self.scale_fact_sim,self.best_model,self.pp_sim,self.ivar,self.lam_gal,self.wdisp,
                              self.vazdekis,self.nTemps,
                              self.line_names1, self.line_wave1,self.nLines1,self.line_names2,self.line_wave2,self.nLines2,
                              self.Bline_names, self.Bline_wave,self.nBlines,
                              self.nClines, self.Cline_names,self.Cline_wave,
                              self.wind_line_names, self.wind_line_wave,self.nwinds,
                              self.nBc,self.nFe,self.slopes,self.nAgns,self.iv_mask,fit_type='sim',units=None)
                    

                end_time = timeit.default_timer()  #end time to check
                self.time_sim=np.round(end_time-start_time)      #time of code section







############################Read the SDSS fits file ###################################
    def file_reading(self, file_dir,lwl,upl,ivar_lim,zz=None,my_sdfmaps=None):
        
        self.file_dir=file_dir

        hdu = fits.open(self.file_dir)
        hm=hdu[0].header
        
        #self.units=hdu[1].header['TUNIT1'] #For SDSS-V pipeline v6.1.3
        t = hdu[1].data
    
        h2=hdu[2].data
        self.z =h2['Z'][0]   # SDSS redshift estimate
        self.mjd=h2['MJD'][0] #SDSS MJD epoch
        #plateid=hdu[0].header['plateid'] # as new spectra don't have plateid just comment
        #if self.z>3:  #Change later for but redshift in the SDSS
         #   self.z=zz
        if zz is not None:
            self.z=zz
        if self.z<0:
            if zz is not None:
                print(f'WARNING: redshift set by zz option')
            else:
                print(f'WARNING: negative redshift {self.z}\n use the zz option to set a different redshift')
            
        #For new templates uses a range (observed) where the SDSS data is not very noisy SB
        # in the observed wavelength we are going to use lwl=3800 and upl=10400. This means 
        # only a cut in the left part of the wavelength that is usually the one with less
        # realible pixels. The right wavelength extreme is usually masked later by the 'ivar' mask

        #Check for wavelengt small than 1200A

        if lwl/(1+self.z)<1200:
            lwl=1200*(1+self.z)
        
    
        mask = (t['loglam'] >= np.log10(lwl)) & (t['loglam'] <= np.log10(upl))
        loglam_gal = t['loglam'][mask]
        lam_gal = 10**loglam_gal
        lam_gal = np.array(lam_gal, dtype=np.float64)
        flux = t['flux'][mask]      
        #Use the ivar for noise
        self.ivar=t['ivar'][mask]
        self.wdisp = t['wdisp'][mask]        # Intrinsic dispersion of every pixel, in pixels units
    
        #Dereddening
        if my_sdfmaps is not None:
            ra_deg = hdu[0].header['plug_ra']     # RA 
            dec_deg = hdu[0].header['plug_dec']   #Dec
            map_dr = sfdmap.SFDMap(my_sdfmaps)
            ebv = map_dr.ebv(ra_deg, dec_deg)
            # Extinction curve
            A_lambda = extinction.fitzpatrick99(lam_gal, 3.1 * ebv) #Rv=3.1 typical Milky Way value
    
            # Deredden flux
            corr_dered = 10 ** (0.4 * A_lambda)
            flux_dered = flux * corr_dered
        else:
            flux_dered=flux
        
        self.scale_factor =  np.median(flux_dered)
       # print('SCLFACT',self.scale_factor)
        self.galaxy = flux/np.median(flux_dered)   # Normalize spectrum to avoid numerical issues
    
        ######MASK using the ivar
        
        self.iv_mask=(self.ivar>ivar_lim)  #Use ivar>0.0 in all the cases if not using a higher value
    
        #Replace the zero values in ivar by the median just to avoid inf values
        #These values are going to be masked in the fit process
        mivar=np.mean(self.ivar[self.ivar>0])
        self.ivar[self.ivar==0]=mivar
        if my_sdfmaps is not None:
            self.noise=np.sqrt(corr_dered**2/self.ivar)
            self.noise=self.noise/self.scale_factor
        else:
            self.noise=np.sqrt(1/self.ivar)
            self.noise=self.noise/self.scale_factor
        # noise = galaxy*0 + 0.0166       # Assume constant noise per pixel here if not having 'ivar' or equivalent
    
        ####add the Balmer continuum not well constrained range in templates in the mask
        #Balmer Continuum 3650 region to mask
        bcll=3610*(1+self.z) #in observed wavelength
        bcul=3690*(1+self.z) #in observed wavelength
        mask_bc = (lam_gal< bcll) | (lam_gal > bcul)
        for pm in range(len(self.iv_mask)):
            if mask_bc[pm]==False:
                self.iv_mask[pm]=False
    
    
        self.c = 299792.458                  # speed of light in km/s
        frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
        self.velscale = np.log(frac)*self.c       # Constant velocity scale in km/s per pixel
    
        dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom
    
        fwhm_gal = 2.355*self.wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms
    
    
        # If the galaxy is at a significant redshift (z > 0.03), one would need to apply
        # a large velocity shift in PPXF to match the template to the galaxy spectrum.
        # This would require a large initial value for the velocity (V > 1e4 km/s)
        # in the input parameter START = [V,sig]. This can cause PPXF to stop!
        # The solution consists of bringing the galaxy spectrum roughly to the
        # rest-frame wavelength, before calling PPXF. In practice there is no
        # need to modify the spectrum in any way, given that a red shift
        # corresponds to a linear shift of the log-rebinned spectrum.
        # One just needs to compute the wavelength range in the rest-frame
        # and adjust the instrumental resolution of the galaxy observations.
        # This is done with the following three commented lines:
        #
        #Condition for high redshift # using always for further analysis and functions
        if self.z>=0.0:
            self.lam_gal = lam_gal/(1+self.z)  # Compute approximate restframe wavelength
            self.fwhm_gals = fwhm_gal/(1+self.z)   # Adjust resolution in Angstrom
            zck=0
        else:
            zck=1

        #return scale_factor
#################################### Set the stellar templates ########################################           
    def stellar_temp(self,dir_temps,lam_gal,velscale,fwhm_gals,str_op_temps,list_st):
        
        # Read the list of filenames from the Single Stellar Population library
        str_op_temps=0 #use any other number to use the option with particular stellar models, also use the list_st
        #list_st=[]  #This maybe an array with the stellar population directories that you want to use
        if str_op_temps==0:
            vdk1=glob.glob(dir_temps)
            self.vazdekis=np.asarray(vdk1)
            ustemps='all_eMstr'
        
        else:
            ustemps='selc_eMstr'
            self.vazdekis=[]
            for tmp in list_st:
                line_ta=tmp
                self.vazdekis=np.append(self.vazdekis,line_ta)
        self.vazdekis.sort()
        fwhm_tem =2.51 #2.51 #Use the same Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
                        #Because I don't know the exact value and apparently is similar
    
        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        #
        hdu2 = fits.open(self.vazdekis[0])
        ssp = hdu2[0].data
        h2 = hdu2[0].header
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        stwvmk=(lam_temp>lam_gal[0]-10)&(lam_temp<lam_gal[-1]+10)
        lam_temp=lam_temp[stwvmk]
        self.lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
        delt_lam=lam_temp[1]-lam_temp[0]
        ssp=ssp[stwvmk]
    
    
        sspNew,self.logLam2,velscale2 = util.log_rebin(self.lamRange_temp, ssp, velscale=velscale)
        self.stars_templates = np.empty((sspNew.size, len(self.vazdekis)))
        #Define the size for templates
        self.Templates_size=sspNew.size
        #print(lamRange_temp,lam_gal[0],lam_gal[-1])
        #print(np.mean(ssp),np.mean(sspNew))
    
        # Interpolates the galaxy spectral resolution at the location of every pixel
        # of the templates. Outside the range of the galaxy spectrum the resolution
        # will be extrapolated, but this is irrelevant as those pixels cannot be
        # used in the fit anyway.
        fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gals)
    
        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the SDSS and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.
    
        # Quadratic sigma difference in pixels Vazdekis --> SDSS
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        #
        # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
        # In principle it should never happen and a higher resolution template should be used.
        #
        fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
        #fwhm_gal2=2.76
        #fwhm_dif = np.sqrt(fwhm_gal2**2 - fwhm_tem**2)
    
        sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
        #sigma=sigma*0
        for j, fname in enumerate(self.vazdekis):
            hdu2 = fits.open(fname)
            ssp = hdu2[0].data
            ssp=ssp[stwvmk]
            ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma        
            sspNew, self.logLam2, velscale = util.log_rebin(self.lamRange_temp, ssp, velscale=velscale)
            self.stars_templates[:, j] = sspNew#/np.median(sspNew) # Normalizes templates
    
        # The galaxy and the template spectra do not have the same starting wavelength.
        # For this reason an extra velocity shift DV has to be applied to the template
        # to fit the galaxy spectrum. We remove this artificial shift by using the
        # keyword VSYST in the call to PPXF below, so that all velocities are
        # measured with respect to DV. This assume the redshift is negligible.
        # In the case of a high-redshift galaxy one should de-redshift its
        # wavelength to the rest frame before using the line below (see above).
        #
        c = 299792.458
        self.dv = np.log(np.exp(self.logLam2[0])/lam_gal[0])*c    # km/s
#################################### Use one stellar template to later set other templates ########################################           
    def set_temp(self,dir_temps,lam_gal,velscale,fwhm_gals,str_op_temps,list_st):
        # Read the list of filenames from the Single Stellar Population library
        str_op_temps=0 #use any other number to use the option with particular stellar models, also use the list_st
        #list_st=[]  #This maybe an array with the stellar population directories that you want to use
        if str_op_temps==0:
            vdk1=glob.glob(dir_temps)
            self.vazdekis=np.asarray(vdk1[0:1])
            ustemps='all_eMstr'
        
        else:
            ustemps='selc_eMstr'
            self.vazdekis=[]
            line_ta=list_st[0]
            self.vazdekis=np.append(self.vazdekis,line_ta)
        self.vazdekis.sort()
        fwhm_tem =2.51 #2.51 #Use the same Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
                        #Because I don't know the exact value and apparently is similar
    
        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the SDSS galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        #
        hdu2 = fits.open(self.vazdekis[0])
        ssp = hdu2[0].data
        h2 = hdu2[0].header
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        #Fill for blue side for high redshift
        if lam_temp[0]>lam_gal[0]:
            flt=lam_temp[1]-lam_temp[0]
            dlt=np.absolute(lam_gal[0]-lam_temp[0])
            npfs=dlt/flt
            extsspr=[lam_temp[0]-(flt*ik) for ik in range(int(npfs)+1)]
            extssp=extsspr[::-1]
            lam_temp=np.append(np.array(extssp),lam_temp)
            ssp=np.append(np.zeros(len(extssp)),ssp)
            
            
        stwvmk=(lam_temp>lam_gal[0]-10)&(lam_temp<lam_gal[-1]+10)
        lam_temp=lam_temp[stwvmk]
        self.lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
        delt_lam=lam_temp[1]-lam_temp[0]
        ssp=ssp[stwvmk]
        
    
        sspNew,self.logLam2,velscale2 = util.log_rebin(self.lamRange_temp, ssp, velscale=velscale)
        #Define the size for templates
        self.Templates_size=sspNew.size
        #Define stars_templates for the constructor
        self.stars_templates=sspNew/np.mean(sspNew)
        
        # The galaxy and the template spectra do not have the same starting wavelength.
        # For this reason an extra velocity shift DV has to be applied to the template
        # to fit the galaxy spectrum. We remove this artificial shift by using the
        # keyword VSYST in the call to PPXF below, so that all velocities are
        # measured with respect to DV. This assume the redshift is negligible.
        # In the case of a high-redshift galaxy one should de-redshift its
        # wavelength to the rest frame before using the line below (see above).
        #
        c = 299792.458
        self.dv = np.log(np.exp(self.logLam2[0])/lam_gal[0])*c    # km/s
#################################### Set FeII templates ######################################## 
    
    def feii_temps(self,fe2_temps,Templates_size,lamRange_temp,velscale):
        
        #fe1=glob.glob(fe2_temps[0])
        #fe2=glob.glob(fe2_temps[1])
        #self.fe=np.append(fe1,fe2)
        self.fe=fe2_temps
    
        #Define the FeII shape templates
        self.fe_templates = np.empty((Templates_size, len(self.fe)))
    
        #FWHM from km/s to AA
        #fwhm_bc=4000
        #sig_bc=(fwhm_bc/(c*2.355))*lam_bc
    
        for j, fnames in enumerate(self.fe):
            ssp = np.loadtxt(fnames)
            ssbc1=ssp[:,1]
            lam_bc1=ssp[:,0]
            deltbc=lam_bc1[1]-lam_bc1[0]
            #add zeros or cut to work with the same wavelenght as the masked stellar templates
            if lam_bc1[0]>lamRange_temp[0]:
                wzle=np.arange(-lam_bc1[0],-lamRange_temp[0],deltbc)
                fzle=wzle*0
                wzle=np.sort(-wzle)
                lam_bc=np.append(wzle,lam_bc1)
                sspbc=np.append(fzle,ssbc1)
            else:
                lmks=(lam_bc1>=lamRange_temp[0])
                lam_bc=lam_bc1[lmks]
                sspbc=ssbc1[lmks]
    
    
            if lam_bc1[-1]<lamRange_temp[-1]:
                wrge=np.arange(lam_bc1[-1]+deltbc,lamRange_temp[-1]+deltbc,deltbc)
                frge=wrge*0
                lam_bc=np.append(lam_bc,wrge)
                sspbc=np.append(sspbc,frge)
            else:
                hmks=(lam_bc<=lamRange_temp[-1])
                lam_bc=lam_bc[hmks]
                sspbc=sspbc[hmks]
        #Define the sigma difference in pixels as zero
        sigma2d=sspbc*0
    
        sspbc = util.gaussian_filter1d(sspbc, sigma2d)  # perform convolution with variable sigma
        sspNew, logLam3, velscale = util.log_rebin(lamRange_temp, sspbc, velscale=velscale)
        mfm=(sspNew>0) #To avoid ~zero median
        sspfm=sspNew[mfm]
        #In some cases the template size is different
        if len(sspNew)!=Templates_size:
            dftl=int(len(sspNew)-Templates_size)
            if dftl>0:
                sspNew=sspNew[:-dftl]
            else:
                sadf=np.zeros([np.abs(dftl)])
                sspNew=np.append(sspNew,sadf)
        self.fe_templates[:, j] = sspNew/np.median(sspfm) # Normalized templates
###############BALMER and BALMER High Order Continuum Templates ####################
    def balmer_templates(self,Balmer_tem,Templates_size,lamRange_temp,velscale):
    #Read the files
        #bc1=glob.glob('BalmerCont/BalCont/Bal*')
        #bc2=glob.glob('BalmerCont/BalHiOrd/*M1000.dat') #Use only the template with FWHM1000km/s
        bc1=glob.glob(Balmer_tem[0])#('BalmerCont/BalCont/Bal*')
        bc2=glob.glob(Balmer_tem[1])#('BalmerCont/BalHiOrd/*M1000.dat') #Use only the template with FWHM1000km/s
        self.bc=np.append(bc1,bc2)
    
        #Define the BC shape templates
        self.bc_templates = np.empty((Templates_size, len(self.bc)))
    
        #FWHM from km/s to AA
        #fwhm_bc=4000
        #sig_bc=(fwhm_bc/(c*2.355))*lam_bc
    
        for j, fname in enumerate(self.bc):
            ssp = np.loadtxt(fname)
            ssbc1=ssp[:,1]
            lam_bc1=ssp[:,0]
            deltbc=lam_bc1[1]-lam_bc1[0]
            #add zeros or cut to work with the same wavelenght as the masked stellar templates
            if lam_bc1[0]>lamRange_temp[0]:
                wzle=np.arange(-lam_bc1[0],-lamRange_temp[0],deltbc)
                fzle=wzle*0
                wzle=np.sort(-wzle)
                lam_bc=np.append(wzle,lam_bc1)
                sspbc=np.append(fzle,ssbc1)
            else:
                lmks=(lam_bc1>=lamRange_temp[0])
                lam_bc=lam_bc1[lmks]
                sspbc=ssbc1[lmks]
    
            if lam_bc1[-1]<lamRange_temp[-1]:
                wrge=np.arange(lam_bc1[-1]+deltbc,lamRange_temp[-1]+deltbc,deltbc)
                frge=wrge*0
                lam_bc=np.append(lam_bc,wrge)
                sspbc=np.append(sspbc,frge)
            else:
                hmks=(lam_bc<=lamRange_temp[-1])
                lam_bc=lam_bc[hmks]
                sspbc=sspbc[hmks]
            #Define the sigma difference in pixels as zero
            sigma2d=sspbc*0
            sspbc = util.gaussian_filter1d(sspbc, sigma2d)  # perform convolution with variable sigma
            sspNew, logLam3, velscale = util.log_rebin(lamRange_temp, sspbc, velscale=velscale)
            mfm=(sspNew>0) #To avoid ~zero median
            sspfm=sspNew[mfm]
            #sspNew=np.append(sspNew,0) #In some cases the template size is different
             #In some cases the template size is different
            if len(sspNew)!=Templates_size:
                dftl=int(len(sspNew)-Templates_size)
                if dftl>0:
                    sspNew=sspNew[:-dftl]
                else:
                    sadf=np.zeros([np.abs(dftl)])
                    sspNew=np.append(sspNew,sadf)
    
            self.bc_templates[:, j] = sspNew  #No Normalized templates
    
#####################Emission Lines################################
    def nels_templates(self,min_lam_gal,max_lam_gal,logLam2):
        FWHM_galx =2.69 #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
        lamRange_gal = np.array([min_lam_gal, max_lam_gal])
        gas_templates, line_names,line_wave = \
            util.emission_lines(logLam2, lamRange_gal, FWHM_galx)
        return gas_templates, line_names,line_wave
        
        
    def bels_templates(self,lam_gal,logLam2):
        FWHM_galx =2.69 #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
        lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
        self.Broad_templates, self.Bline_names, self.Bline_wave = \
            util.Broad_emission_lines(logLam2, lamRange_gal, FWHM_galx)

    def civ_templates_Bels(self,lam_gal,logLam2):
        FWHM_galx =2.69 #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
        lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
        self.civ_templates, self.Cline_names, self.Cline_wave = \
            util.Broad_civ_lines(logLam2, lamRange_gal, FWHM_galx)

    def my_wind_templates(self,lam_gal,logLam2):
        FWHM_galx =2.69 #2.69A (2.5A) #BOSS red(blue) chanel instrumental resolution
        lamRange_gal = np.array([np.min(lam_gal), np.max(lam_gal)])
        self.wind_templates, self.wind_line_names, self.wind_line_wave=util.emission_o3_winds(logLam2, lamRange_gal, FWHM_galx)
        #self.wind_templates, self.wind_line_names, self.wind_line_wave=util.emission_lines(logLam2, lamRange_gal, FWHM_galx)
        #When using a second semi-broad component for the NELs pPXF can use it to model part of the broad component in the
        #BELs regions, then we restrict the winds to the prominent [OIII] region
 
##########################AGN templates##################################           
    def my_agn_templates(self,logLam2,islope,fslope):                
    
        lambda_norm=5000
        self.agn_templates, self.slopes=util.AGN(logLam2,lambda_norm,stp=0.1,islope=islope,fslope=fslope)
########################## MODEL CONSTRUCTOR ##################################           

    def model_constructor(self,model_name,stars_templates,gas_templates,Broad_templates,civ_templates,wind_templates,bc_templates,
                    fe_templates,agn_templates,redshift,vel_o3=None,o3gc=None,bel_vel=None):
        '''- model_name: define the components that are going to be used in the model, the options are listed below:
                model_*_1: *, NELs, BELs, Balmer High_order and Balmer continuum, FeII, AGN 
                model_*_2: *, NELs, BELs, BELs, Balmer High_order and Balmer continuum, FeII, AGN 
                model_*_3: *, NELs, BELs, winds [O_III]d5007, Balmer High_order and Balmer continuum, FeII, AGN 
                model_*_4: *, NELs, BELs, BELs, winds [O_III]d5007, Balmer High_order and Balmer continuum, FeII, AGN 
                *: st or nst, st means using stellar templates (for z<1[0.8]) and nst means not ssp (for z>1[0.8])
        '''

        print('construction of: '+model_name)
        # Define the initial values for all the posible components
        #You can manipulate this input values to improve the fit
        vel=0
        start_s = [vel, 10.,0.,0.] # (km/s), starting guess for [V,sigma] stellar
        start_g1 = [vel, 100.] # (km/s), starting guess for [V,sigma] Gas narrow left
        start_g2 = [vel, 100.] # (km/s), starting guess for [V,sigma] Gas narrow right
        start_wn = [vel-100, 500.] # (km/s), starting guess for [V,sigma] Gas winds
        start_gb = [vel, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
        start_bc=[vel,5.]
        start_fe=[vel,5.]
        agn_start = [vel, 5.] # (km/s), starting guess for [V,sigma] AGN

        if vel_o3 is not None and\
        (model_name=='model_st_3' or model_name=='model_st_4' or model_name=='model_nst_3' or model_name=='model_nst_4'):
            #if o3gc==1: #commented, before used to indicated in which component the [OIII] line was
            #Use the same initial values for both NELs components
           # print(vel_o3)
            start_g1 = [vel_o3, 90.] # (km/s), starting guess for [V,sigma] Gas narrow left
        #else:
            start_g2 = [vel_o3, 90.] # (km/s), starting guess for [V,sigma] Gas narrow right
       # print('INT VEL NELs')
       # print(start_g1,start_g2)
        #Define bouns for kinematic vals for each component
        ##########BOUNDS: Use this to limit the kinematic values 
        #vel stars
        v1=-300 #-300
        v2=300 #300
        #vel_disp stars
        vd1=10
        vd2=500 #1000
        #stars
        sbn=[[v1,v2],[vd1,vd2],[-0.3,0.3],[-0.3,0.3]]

        #vel winds
        vwg1=-800 #-600
        vwg2=600 #400
        #vel_disp narrow
        vdwd1=400
        vdwd2=800 #400 #1000
        #narrow lines / agn
        wbn=[[vwg1,vwg2],[vdwd1,vdwd2]]
        
        #Define two different bounds for NELs, BELs at 1=z because
        #it is more probable that the pipeline redshift is not well 
        #constrained at z>1 because NELs are poorly detected

        if redshift>1:
            #vel narrow gas
            vng1=-600 #-400
            vng2=600   #400
            #vel_disp narrow
            vdd1=5
            vdd2=300 #300
            #narrow lines / agn
            nbn=[[vng1,vng2],[vdd1,vdd2]]
    
            
        
            #vel broad
            vbg1=-1500 #-600
            vbg2=1500 #600
            #vel_disp Broad_1
            vdb1=500 #400
            vdb2=10000 #2000
            #broad lines
            bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
    
            
        else:
            #vel narrow gas
            vng1=-400 #-400
            vng2=400   #400
            #vel_disp narrow
            vdd1=5
            vdd2=300 #300
            #narrow lines / agn
            nbn=[[vng1,vng2],[vdd1,vdd2]]
    
            
        
            #vel broad
            vbg1=-600 #-600
            vbg2=600 #600
            #vel_disp Broad_1
            vdb1=500 #400
            vdb2=10000 #2000
            #broad lines
            bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
    
        
        #vel Civ broad
        vbc1=-5000 #-1000
        vbc2=2000  #1000
        #vel_disp Broad_1
        vdc1=500 #400
        vdc2=10000 #2000
        #broad lines
        bbc=[[vbc1,vbc2],[vdc1,vdc2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
        #Bc
        bcbn=[[-500,500],[1.,1000]]
        #FeII
        #febn=[[-500,500],[1,1000]] #-+100 ;1-300
        febn=[[vbg1,vbg2],[1,300]] #-+100 ;1-300
        #(AGN)
        #agnbn=[[-500,500],[1,100]]#-+100; 1-10
        agnbn=[[-100,100],[1,10]]#-+100; 1-10
        ############tied option to tied the Broad emission lines and the Balmer High order velocity########################
        tied_ssp=['','','','']
        tied_nel1=['','']
        tied_nel2=['','']
        tied_wind=['','']
        tied_bel=['','','','']
        tied_belc=['','','','']
        #tied_bc=['p[indx_BELS_vel]',''] #defined for each case
        #tied_fe2=['p[indx_BELS_vel]',''] # FeII is also tied in each case
        tied_agn=['','']
        
                
        
        if len(gas_templates)>1:
            gas_templates1=gas_templates[0]
            gas_templates2=gas_templates[1]
            try:
                self.nTemps = stars_templates.shape[1]
            except:
                self.nTemps = 0
            self.nLines1 = gas_templates1.shape[1]
            self.nLines2 = gas_templates2.shape[1]
            self.nBlines=Broad_templates.shape[1]
            self.nClines=civ_templates.shape[1]
            try:
                self.nwinds=wind_templates.shape[1]
            except:
                self.nwinds=1
            self.nFe=fe_templates.shape[1]
            self.nBc=bc_templates.shape[1]
            self.nAgns=agn_templates.shape[1]
            #print(self.nBc,self.nFe,self.nAgns)
            if model_name=='model_st_1':
                
                self.templates = np.column_stack([stars_templates ,gas_templates1,gas_templates2,Broad_templates,bc_templates,
                                 fe_templates,agn_templates])
                self.component = [0]*self.nTemps + [1]*self.nLines1+ [2]*self.nLines2+[3]*self.nBlines+[4]*self.nBc+[5]*self.nFe+[6]*self.nAgns
                self.moments= [4,2,2,4,2,2,2]
                self.start= [start_s, start_g1,start_g2,start_gb,start_bc,start_fe,agn_start]
                self.bounds=[sbn,nbn,nbn,bbn,bcbn,febn,agnbn]
                tied_nel2=['','p[5]']
                tied_bc=['p[8]','']
                tied_fe2=['p[8]','']
                self.tied=[tied_ssp,tied_nel1,tied_nel2,tied_bel,tied_bc,tied_fe2,tied_agn]       
                self.nnc=7

                
                    

            elif model_name=='model_st_2':
                
                self.templates = np.column_stack([stars_templates ,gas_templates1,gas_templates2,Broad_templates,Broad_templates,bc_templates,
                                 fe_templates,agn_templates])
                self.component = [0]*self.nTemps + [1]*self.nLines1+ [2]*self.nLines2+[3]*self.nBlines+[4]*self.nBlines+\
                [5]*self.nBc+[6]*self.nFe+[7]*self.nAgns
                self.moments= [4,2,2,4,4,2,2,2]

                if bel_vel is not None:
                    #vel broad
                    vbg1=bel_vel-300
                    vbg2=bel_vel+300
                    #vel_disp Broad_1
                    vdb1=500 #400
                    vdb2=5000 #2000
                    #broad lines
                    bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                    start_gb = [bel_vel-100, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
                    start_gb = [bel_vel+100, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
                    

                
                self.start= [start_s, start_g1,start_g2,start_gb,start_gb,start_bc,start_fe,agn_start]
                self.bounds=[sbn,nbn,nbn,bbn,bbn,bcbn,febn,agnbn]
                tied_nel2=['','p[5]']
                tied_bc=['p[8]','']
                tied_fe2=['p[8]','']
                self.tied=[tied_ssp,tied_nel1,tied_nel2,tied_bel,tied_bel,tied_bc,tied_fe2,tied_agn]       
                self.nnc=8
            elif model_name=='model_st_3':
                
                self.templates = np.column_stack([stars_templates ,gas_templates1,gas_templates2,Broad_templates,wind_templates,bc_templates,
                                 fe_templates,agn_templates])
                self.component = [0]*self.nTemps + [1]*self.nLines1+ [2]*self.nLines2+[3]*self.nBlines+[4]*self.nwinds+\
                [5]*self.nBc+[6]*self.nFe+[7]*self.nAgns
                self.moments= [4,2,2,4,2,2,2,2]
                self.start= [start_s, start_g1,start_g2,start_gb,start_wn,start_bc,start_fe,agn_start]
                self.bounds=[sbn,nbn,nbn,bbn,wbn,bcbn,febn,agnbn]
                tied_nel2=['','p[5]']
                tied_bc=['p[8]','']
                tied_fe2=['p[8]','']
                self.tied=[tied_ssp,tied_nel1,tied_nel2,tied_bel,tied_wind,tied_bc,tied_fe2,tied_agn] 
                self.nnc=8
            elif model_name=='model_st_4':
                
                self.templates = np.column_stack([stars_templates ,gas_templates1,gas_templates2,Broad_templates,
                                                  Broad_templates,wind_templates,bc_templates,fe_templates,agn_templates])
                self.component = [0]*self.nTemps + [1]*self.nLines1+ [2]*self.nLines2+[3]*self.nBlines+[4]*self.nBlines+\
                [5]*self.nwinds+[6]*self.nBc+[7]*self.nFe+[8]*self.nAgns
                self.moments= [4,2,2,4,4,2,2,2,2]
                if bel_vel is not None:
                    #vel broad
                    vbg1=bel_vel-300
                    vbg2=bel_vel+300
                    #vel_disp Broad_1
                    vdb1=500 #400
                    vdb2=5000 #2000
                    #broad lines
                    bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                    start_gb = [bel_vel-100, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
                    start_gb = [bel_vel+100, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines

                self.start= [start_s, start_g1,start_g2,start_gb,start_gb,start_wn,start_bc,start_fe,agn_start]
                self.bounds=[sbn,nbn,nbn,bbn,bbn,wbn,bcbn,febn,agnbn]
                tied_nel2=['','p[5]']
                tied_bc=['p[8]','']
                tied_fe2=['p[8]','']
                self.tied=[tied_ssp,tied_nel1,tied_nel2,tied_bel,tied_bel,tied_wind,tied_bc,tied_fe2,tied_agn]       
                self.nnc=9

            
            # When no using stars_templates
            if model_name=='model_nst_1':
                if self.nClines==0:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,bc_templates,
                                     fe_templates,agn_templates])
                    self.component =  [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nBc+[4]*self.nFe+[5]*self.nAgns
                    self.moments= [2,2,4,2,2,2]
                    self.start= [start_g1,start_g2,start_gb,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_bc,tied_fe2,tied_agn] 
                    self.nnc=6

                else:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,civ_templates,bc_templates,
                                 fe_templates,agn_templates])
                    self.component =  [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nClines+[4]*self.nBc+\
                    [5]*self.nFe+[6]*self.nAgns
                    self.moments= [2,2,4,4,2,2,2]
                    self.start= [start_g1,start_g2,start_gb,start_gb,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbc,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_belc,tied_bc,tied_fe2,tied_agn] 
                    self.nnc=7

            elif model_name=='model_nst_2':
                if self.nClines==0:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,Broad_templates,bc_templates,
                                     fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nBlines+\
                    [4]*self.nBc+[5]*self.nFe+[6]*self.nAgns
                    self.moments= [2,2,4,4,2,2,2]
                    if bel_vel is not None:
                        #vel broad
                        vbg1=bel_vel-200
                        vbg2=bel_vel+200
                        #vel_disp Broad_1
                        vdb1=500 #400
                        vdb2=5000 #2000
                        #broad lines
                        bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                        start_gb = [bel_vel, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
    
                    self.start= [start_g1,start_g2,start_gb,start_gb,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_bel,tied_bc,tied_fe2,tied_agn]
                    self.nnc=7
                else:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,Broad_templates,
                                 civ_templates,civ_templates,bc_templates,fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nBlines+\
                    [4]*self.nClines+[5]*self.nClines+[6]*self.nBc+[7]*self.nFe+[8]*self.nAgns
                    self.moments= [2,2,4,4,4,4,2,2,2]
                    if bel_vel is not None:
                        #vel broad
                        vbg1=bel_vel-200
                        vbg2=bel_vel+200
                        #vel_disp Broad_1
                        vdb1=500 #400
                        vdb2=5000 #2000
                        #broad lines
                        bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                        start_gb = [bel_vel, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
                        start_civ= [vel, 1000.,0.,0.]
    
                    self.start= [start_g1,start_g2,start_gb,start_gb,start_civ,start_civ,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbn,bbc,bbc,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_bel,tied_belc,tied_belc,tied_bc,tied_fe2,tied_agn]
                    self.nnc=9
            elif model_name=='model_nst_3':
                if self.nClines==0:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,wind_templates,bc_templates,
                                     fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nwinds+\
                    [4]*self.nBc+[5]*self.nFe+[6]*self.nAgns
                    self.moments= [2,2,4,2,2,2,2]
                    self.start= [ start_g1,start_g2,start_gb,start_wn,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,wbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_wind,tied_bc,tied_fe2,tied_agn]
                    self.nnc=7
                else:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,civ_templates,wind_templates,
                                     bc_templates,fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nClines+[4]*self.nwinds+\
                    [5]*self.nBc+[6]*self.nFe+[7]*self.nAgns
                    self.moments= [2,2,4,4,2,2,2,2]
                    self.start= [ start_g1,start_g2,start_gb,start_gb,start_wn,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbc,wbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_belc,tied_wind,tied_bc,tied_fe2,tied_agn]
                    self.nnc=8
            elif model_name=='model_nst_4':
                if self.nClines==0:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,
                                                      Broad_templates,wind_templates,bc_templates,fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nBlines+\
                    [4]*self.nwinds+[5]*self.nBc+[6]*self.nFe+[7]*self.nAgns
                    self.moments= [2,2,4,4,2,2,2,2]
                    if bel_vel is not None:
                        #vel broad
                        vbg1=bel_vel-200
                        vbg2=bel_vel+200
                        #vel_disp Broad_1
                        vdb1=500 #400
                        vdb2=5000 #2000
                        #broad lines
                        bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                        start_gb = [bel_vel, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
    
                    self.start= [ start_g1,start_g2,start_gb,start_gb,start_wn,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbn,wbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_bel,tied_wind,tied_bc,tied_fe2,tied_agn] 
                    self.nnc=8
                else:
                    self.templates = np.column_stack([gas_templates1,gas_templates2,Broad_templates,
                                                      Broad_templates,wind_templates,civ_templates,civ_templates,
                                                      bc_templates,fe_templates,agn_templates])
                    self.component = [0]*self.nLines1+ [1]*self.nLines2+[2]*self.nBlines+[3]*self.nBlines+\
                    [4]*self.nClines+[5]*self.nClines+[6]*self.nwinds+[7]*self.nBc+[8]*self.nFe+[9]*self.nAgns
                    self.moments= [2,2,4,4,4,4,2,2,2,2]
                    if bel_vel is not None:
                        #vel broad
                        vbg1=bel_vel-200
                        vbg2=bel_vel+200
                        #vel_disp Broad_1
                        vdb1=500 #400
                        vdb2=5000 #2000
                        #broad lines
                        bbn=[[vbg1,vbg2],[vdb1,vdb2],[-0.1,0.1],[-0.1,0.1]] #h3:symmetry  h4:shape
                        start_gb = [bel_vel, 1000.,0.,0.] # (km/s), starting guess for [V,sigma] Broad lines
                        start_civ= [vel, 1000.,0.,0.]
    
                    self.start= [ start_g1,start_g2,start_gb,start_gb,start_civ,start_civ,start_wn,start_bc,start_fe,agn_start]
                    self.bounds=[nbn,nbn,bbn,bbn,bbc,bbc,wbn,bcbn,febn,agnbn]
                    tied_nel2=['','p[1]']
                    tied_bc=['p[4]','']
                    tied_fe2=['p[4]','']
                    self.tied=[tied_nel1,tied_nel2,tied_bel,tied_bel,tied_belc,tied_belc,tied_wind,tied_bc,tied_fe2,tied_agn] 
                    self.nnc=8


########################## COMPARISON ################################## 
    def comparison_two_fits(self,residuals1,dof_1,residuals2,dof_2,lam_gal,f_mask,cp_ty=0):
        '''cp_ty: comparison type
            cp_ty=0 for global and local
            cp_ty=1 for only global
            cp_ty=2 for only local
            f_mask: mask used during the fit
            WARNING: residuals_1, and dof_1 are always from the less complex model, meaning less components and templates'''
        #mask the residuals
       # print('starting comparison')
        #print(len(residuals1),len(residuals2),len(f_mask))
        residuals_1=residuals1[f_mask]
        residuals_2=residuals2[f_mask]
       # print('starting comparison pass')
        #Global comparison
        if cp_ty==1 or cp_ty==0:
           # print('starting global comparison')
            #RMS
            self.rms_m1=np.sqrt(np.mean(residuals_1**2))
            self.rms_m2=np.sqrt(np.mean(residuals_2**2))
            self.Rdif=(self.rms_m1-self.rms_m2)/self.rms_m1
            #print('starting scoring comparison')
            # qualitative comparison and score
            #From the RMS six cases are going to be consider
            if self.Rdif>0.1:
                self.cp_gl_sc=1
                self.comment_cp='significative'
            elif 0.01<self.Rdif<0.1:
                self.cp_gl_sc=0.5
                self.comment_cp='partially_significative'
            elif 0.0<=self.Rdif<0.01:
                self.cp_gl_sc=0.0
                self.comment_cp='no_significative'
            elif -0.01<self.Rdif<0.0:
                self.cp_gl_sc=0.0
                self.comment_cp='no_significative'
            elif -0.1<self.Rdif<-0.01:
                self.cp_gl_sc=-1
                self.comment_cp='partially_negative_significative'
            elif self.Rdif<-0.1:
                self.cp_gl_sc=-100
                self.comment_cp='negative_significative'
            
            #F-test
            #print('starting F-test comparison')
            self.ssr_1=np.sum(residuals_1**2)
            self.ssr_2=np.sum(residuals_2**2)
            self.F_stat=((self.ssr_1-self.ssr_2)/(dof_1-dof_2))/(self.ssr_2/dof_2)
            self.p_val_gl=1-f.cdf(self.F_stat,dof_1-dof_2,dof_2)
            if self.p_val_gl<0.05:
                self.comment_cp=self.comment_cp+'_F_test_significative'
                self.cp_gl_fts_sc=1
            else:
                self.comment_cp=self.comment_cp+'_F_test_no_significative'
                #self.cp_gl_fts_sc=-1
                self.cp_gl_fts_sc=0

            self.global_score_comp=self.cp_gl_sc+self.cp_gl_fts_sc

        #Local comparison
        if cp_ty==2 or cp_ty==0:
            #print('starting local comparison')
            lam_range_r=[lam_gal[0],lam_gal[-1]]
            self.regions_comp,self.regions_comp_name=my_regions(lam_range_r)
            self.local_score_comp=0
            lam_gal_mk=lam_gal[f_mask]
            for my_lam in self.regions_comp:
               # print('local loop comparison')
                mask_reg=((lam_gal_mk>my_lam[0])&(lam_gal_mk<my_lam[1]))
                residuals_reg_1=residuals_1[mask_reg]
                residuals_reg_2=residuals_2[mask_reg]
            
                #RMS
                self.rms_lc_m1=np.sqrt(np.mean(residuals_reg_1**2))
                self.rms_lc_m2=np.sqrt(np.mean(residuals_reg_2**2))
                self.Rdif_lc=(self.rms_lc_m1-self.rms_lc_m2)/self.rms_lc_m1
                #print('starting local scoring comparison')
                # qualitative comparison and score
                #From the RMS six cases are going to be consider
                if self.Rdif_lc>0.1:
                    self.cp_lc_sc=1
                    self.comment_lc_cp='significative'
                elif 0.01<self.Rdif_lc<0.1:
                    self.cp_lc_sc=0.5
                    self.comment_lc_cp='partially_significative'
                elif 0.0<=self.Rdif_lc<0.01:
                    self.cp_lc_sc=0.0
                    self.comment_lc_cp='no_significative'
                elif -0.01<self.Rdif_lc<0.0:
                    self.cp_lc_sc=0.0
                    self.comment_lc_cp='no_significative'
                elif -0.1<self.Rdif_lc<-0.01:
                    self.cp_lc_sc=-1
                    self.comment_lc_cp='partially_negative_significative'
                elif self.Rdif_lc<-0.1:
                    self.cp_lc_sc=-100
                    self.comment_lc_cp='negative_significative'
    
                #F-test need the DOF for each region, as these values are not easy to extract for each region
                #the F-test is not performed but is commented 
                #self.ssr_lc_1=np.sum(residuals_reg_1**2)
                #self.ssr_lc_2=np.sum(residuals_reg_2**2)
                #self.F_stat_lc=((self.ssr_lc_1-self.ssr_lc_2)/(dof_lc_1-dof_lc_2))/(self.ssr_lc_2/dof_lc_2)
                #self.p_val_lc=1-f.cdf(self.F_stat_lc,dof_lc_1-dof_lc_2,dof_lc_2)
                #if self.p_val_lc<0.05:
                 #   self.comment_lc_cp=self.comment_lc_cp+'_F_test_significative'
                 #   self.cp_lc_fts_sc=2
                #else:
                 #   self.comment_lc_cp=self.comment_lc_cp+'_F_test_no_significative'
                 #   self.cp_lc_fts_sc=-1
                
                self.local_score_comp=self.local_score_comp+self.cp_lc_sc
        if cp_ty==0:
            self.comparison_sc=self.local_score_comp+self.global_score_comp
        elif cp_ty==1:
            self.comparison_sc=self.global_score_comp
        elif cp_ty==2:
            self.comparison_sc=self.local_score_comp
        #print('finished comparison internal')
                
                
########################## Plot fit results ##################################                 
    def plot_fit_model(self,file_dir,z,model_name,pp,lam_gal,nTemps,nLines1,nLines2,nBlines,nClines,nwinds,
                        nBc,nFe,nAgns,option=5,save_fig_dir=None,ini=0):
        print(f'Ploting model {model_name}')
        #Components extraction
        if model_name in np.array(['model_st_2','model_st_4','model_nst_2','model_nst_4']):
            nBlines2=nBlines
            nClines2=nClines
        else:
            nBlines2=0
            nClines2=0
        if model_name in np.array(['model_st_3','model_st_4','model_nst_3','model_nst_4']):
            nwinds2=nwinds
        else:
            nwinds2=0
       # print(nBlines2,nwinds2)    
        #define arrays to loop an extract the compoents
        nValues=[nLines1,nLines2,nBlines,nBlines2,nClines,nClines2,nwinds2,nBc,nFe,nAgns]
        comp_name=['gas_narrow_1f','gas_narrow_2f','gas_broad_1f','gas_broad_2f','gas_Civ_1f','gas_Civ_2f',
                   'winds_f','Bal_cont_f','fe2_f','agn_f']
        comp_vals={}
        
        g0=nTemps
        for nval, c_name in zip(nValues,comp_name):
            comp_vals[c_name]=pp.matrix[:,g0:g0+nval]@pp.weights[g0:g0+nval]
            g0=g0+nval

        #plot 
        plt.figure(figsize=(15,10))
        
        #Plot regions
        lam_range_r=[lam_gal[0],lam_gal[-1]]
        regions_comp1,regions_comp_name1=my_regions(lam_range_r)
        regions_comp1=np.array(regions_comp1)
        regions_comp_name1=np.array(regions_comp_name1)
        
        regions_comp=regions_comp1[(regions_comp_name1!='OIII')]
        regions_comp_name=regions_comp_name1[(regions_comp_name1!='OIII')]
        #define the grid
        if len(regions_comp)>2:
            gs = gridspec.GridSpec(7, 2)
            lgs=6
        else:
            gs = gridspec.GridSpec(5, 2)
            lgs=4

        #Plot all spectrum
        plt.subplot(gs[lgs-2:lgs,:])
        plt.plot(lam_gal,pp.galaxy,label='spectrum')
        plt.plot(lam_gal,pp.bestfit,label='best-fit')
        if nTemps>0:
            stars_f=pp.matrix[:,:nTemps]@pp.weights[:nTemps]
            plt.plot(lam_gal,stars_f,label='stars')
        plt.plot(lam_gal,comp_vals['gas_narrow_1f'],label='NELs-lf')
        plt.plot(lam_gal,comp_vals['gas_narrow_2f'],label='NELs-rg')
        plt.plot(lam_gal,comp_vals['gas_broad_1f'],label='BELs-1')
        if nClines!=0:
            plt.plot(lam_gal,comp_vals['gas_Civ_1f'],label='Civ-1')
        if model_name in np.array(['model_st_2','model_st_4','model_nst_2','model_nst_4']):
            plt.plot(lam_gal,comp_vals['gas_broad_2f'],label='BELs-2')
            if nClines!=0:
                plt.plot(lam_gal,comp_vals['gas_Civ_2f'],label='Civ-2')
        if model_name in np.array(['model_st_3','model_st_4','model_nst_3','model_nst_4']):
            plt.plot(lam_gal,comp_vals['winds_f'],label='winds')
        plt.plot(lam_gal,comp_vals['Bal_cont_f'],label='Bal-cont-BHO')
        plt.plot(lam_gal,comp_vals['fe2_f'],label='FeII')
        plt.plot(lam_gal,comp_vals['agn_f'],label='AGN-cont')
        
        if np.max(pp.galaxy)>2.8*np.max(pp.bestfit):
            maxspec=np.max(pp.bestfit)
        elif np.max(pp.bestfit)>2.8*np.max(pp.galaxy):
            maxspec=np.max(pp.galaxy)
        else:
            maxspec=np.max([np.max(pp.galaxy),np.max(pp.bestfit)])
        plt.ylim(-0.5,maxspec+0.1*maxspec)
        plt.ylabel('Flux [u]')
        # Add a legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1), borderaxespad=0)
        #plt.legend()
        #Plot residuals
        plt.subplot(gs[lgs,:])
        plt.plot(lam_gal,pp.galaxy-pp.bestfit,label='residuals')
        plt.xlabel('Rest-frame wavelength ($\\AA$)')

        
        for i, (my_lam,my_reg_nm) in enumerate(zip(regions_comp,regions_comp_name)):
            #my_lam=my_lam.tolist()
            #my_lam=np.array(my_lam)
            #print(my_lam)
            i2=i+1
            idxpl=i2%2
            adidx=(i//2)*2
            if  my_reg_nm=='Hbeta':  
                my_lam[1]=5010
            plt.subplot(gs[lgs-4-adidx:lgs-2-adidx,idxpl])
            mask_reg=((lam_gal>my_lam[0]-100)&(lam_gal<my_lam[1]+100))
            plt.title(my_reg_nm)
            plt.plot(lam_gal[mask_reg],pp.galaxy[mask_reg],label='spectrum',linewidth=3)
            plt.plot(lam_gal[mask_reg],pp.bestfit[mask_reg],label='best-fit')
            if nTemps>0:
                stars_f=pp.matrix[:,:nTemps]@pp.weights[:nTemps]
                plt.plot(lam_gal[mask_reg],stars_f[mask_reg],label='stars')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_narrow_1f'])[mask_reg],label='NELs-lf')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_narrow_2f'])[mask_reg],label='NELs-rg')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_broad_1f'])[mask_reg],label='BELs-1')
            if nClines!=0:
                plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_Civ_1f'])[mask_reg],label='Civ-1')
            if model_name in np.array(['model_st_2','model_st_4','model_nst_2','model_nst_4']):
                plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_broad_2f'])[mask_reg],label='BELs-2')
                if nClines!=0:
                    plt.plot(lam_gal[mask_reg],np.array(comp_vals['gas_Civ_2f'])[mask_reg],label='Civ-2')
            if model_name in np.array(['model_st_3','model_st_4','model_nst_3','model_nst_4']):
                plt.plot(lam_gal[mask_reg],np.array(comp_vals['winds_f'])[mask_reg],label='winds')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['Bal_cont_f'])[mask_reg],label='Balmer-cont-BHO')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['fe2_f'])[mask_reg],label='FeII')
            plt.plot(lam_gal[mask_reg],np.array(comp_vals['agn_f'])[mask_reg],label='AGN-cont')
            #maxspec=np.max([np.max(pp.galaxy[mask_reg]),np.max(pp.bestfit[mask_reg])])
            plt.ylim(-0.5,np.max(pp.galaxy[mask_reg])+0.1*np.max(pp.galaxy[mask_reg]))
            #plt.legend()
        #save plot
        
        dir1_plt=file_dir.split(sep='/spec-')
        dir1_plt2=dir1_plt[-1].split(sep='.f')
        if save_fig_dir is None:
            sf_dir='./'
        else:
            sf_dir=save_fig_dir

        print(sf_dir+'fit-'+model_name+'-'+dir1_plt2[0]+'_ini.png')
        if ini==1:
            plt.savefig(sf_dir+'fit-'+model_name+'-'+dir1_plt2[0]+'_ini.png')
        elif ini==2:
            plt.savefig(sf_dir+'fit-'+model_name+'-'+dir1_plt2[0]+'_model_suggs.png')
        else:
            plt.savefig(sf_dir+'fit-'+model_name+'-'+dir1_plt2[0]+'.png')
        print('Plot saved in: '+sf_dir+'fit-'+model_name+'-'+dir1_plt2[0]+'.png')
        
        plt.close()
            
            
                
                

            




    
########################## Save results ################################## 

    def fits_constructor(self,file_dir,z,scale_fact,model_name,pp,ivar,lam_gal,wdisp,st_names,nTemps,
                        line_names1,line_wave1,nLines1,line_names2,line_wave2,nLines2,
                        Bline_names, Bline_wave,nBlines,nClines, Cline_names,Cline_wave,wind_line_names,
                         wind_line_wave,nwinds,nBc,nFe,slopes,nAgns,fit_mask,wts=0,fit_type=None,units=None,my_chi2=None):
        '''wts>0 to indicate inthe fits name that the model used was the initial model'''
        #Create the Primary and header
        hdu=fits.PrimaryHDU()
        hdr=hdu.header
        #file_name fro SDSS-V spec or fit-spec from SDSS-V
        n1=file_dir.split(sep='/spec-')
        if len(n1)==1:
            n1=file_dir.split(sep='/fit-')
        if len(n1)==1:
            n1=file_dir.split(sep='/') #Consider other cases

        n2=n1[-1].split(sep='.fit')
        if fit_type is not None:
            nm=fit_type+'-'+n2[0]
        else:
            nm='spec-'+n2[0]
        hdr['ID_name']=(nm,'type of fit') #name in fits
        #Other values
        hdr['Z']=(z,'SDSS pipeline redshift')
        hdr['Scl_fact']=(scale_fact, 'ScaleFactor-median-flux_org')
        if units is not None:
            hdr['F_units']=(units,'Flux units from spectrum')
        hdr['model']=(model_name,'Model fitted name software')
        #Software version
        soft_v='AMARU_v0.2,0'
        hdr['software']=(soft_v,'name of the code used')
        #Chi2 from fit
        chi2ppxf=round(pp.chi2,2)
        hdr['chi2_pp']=(chi2ppxf,'chi2 from ppxf')
        hdr['fit_chi2']=(my_chi2,'reduced chi2 from fit')
        ###############
        # Arrays with components: Here the flux units are flux/median(flux)
        #Consider thet differen models have different components
        #Components extraction
        if model_name in np.array(['model_st_2','model_st_4','model_nst_2','model_nst_4']):
            nBlines2=nBlines
            nClines2=nClines
        else:
            nBlines2=0
            nClines2=0
        if model_name in np.array(['model_st_3','model_st_4','model_nst_3','model_nst_4']):
            nwinds2=nwinds
        else:
            nwinds2=0
       # print(nBlines2,nwinds2)    
        #define arrays to loop an extract the compoents
        nValues=[nLines1,nLines2,nBlines,nBlines2,nClines,nClines2,nwinds2,nBc,nFe,nAgns]
        comp_name=['gas_narrow_1f','gas_narrow_2f','gas_broad_1f','gas_broad_2f','gas_Civ_1f','gas_Civ_2f',
                   'winds_f','Bal_cont_f','fe2_f','agn_f']
        comp_vals={}
        
        g0=nTemps
        for nval, c_name in zip(nValues,comp_name):
            comp_vals[c_name]=pp.matrix[:,g0:g0+nval]@pp.weights[g0:g0+nval]
            g0=g0+nval
        
        if nTemps>0:
            stars_f=pp.matrix[:,:nTemps]@pp.weights[:nTemps]
        else:
            stars_f=0*pp.bestfit

        if nClines==0:
            comp_vals['gas_Civ_1f']=0*np.array(comp_vals['gas_Civ_1f'])
            comp_vals['gas_Civ_2f']=0*np.array(comp_vals['gas_Civ_1f'])
            
            
        #Residuals for all
        residuals=pp.galaxy-pp.bestfit
        

        #Now consider the different models
        if model_name=='model_st_1' or model_name=='model_nst_1':
            print('Into_saving model 1')
            my_comp_names=('Best-fit','stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_Civ','Bal_Cont',
            'Fe_II','AGN','residuals','ivar','wave','wdisp','fit_mask')
            tcomp=Table([pp.bestfit,stars_f,np.array(comp_vals['gas_narrow_1f']),np.array(comp_vals['gas_narrow_2f']),
                    np.array(comp_vals['gas_broad_1f']),np.array(comp_vals['gas_Civ_1f']),np.array(comp_vals['Bal_cont_f']),
                    np.array(comp_vals['fe2_f']),np.array(comp_vals['agn_f']),residuals,ivar,lam_gal,wdisp,fit_mask],names=my_comp_names)
            print('Saved model 1')
            ###############
            #Kinematic results
            names1=('Component','velocity(Km/s)', 'velocity dispersion(Km/s)','h3','h4')
            names11=np.array(['stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Bal_Cont','Fe_II','AGN'])
            if nClines>0:
                names11=np.array(['stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_Civ','Bal_Cont','Fe_II','AGN'])
        #Generate the vel array
            vels=[]
            vels_disp=[]
            mnh3=[]
            mnh4=[]
            len_comp=len(names11)
            if nTemps==0:
                vels=[-9999999]
                vels_disp=[-9999999]
                mnh3=[-9999999]
                mnh4=[-9999999]
                len_comp=len_comp-1
                
            for jj in range(len_comp):
                vels=np.append(vels,pp.sol[jj][0])
                vels_disp=np.append(vels_disp,pp.sol[jj][1])
                try:
                    mnh3=np.append(mnh3,pp.sol[jj][2])
                    mnh4=np.append(mnh4,pp.sol[jj][3])
                except:
                    mnh3=np.append(mnh3,0.0)
                    mnh4=np.append(mnh4,0.0)
            tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)

        elif model_name=='model_st_2' or model_name=='model_nst_2':
            print('Into_saving model 2')
            
            my_comp_names=('Best-fit','stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_broad_2','Gas_Civ','Gas_Civ_2',
                           'Bal_Cont','Fe_II','AGN','residuals','ivar','wave','wdisp','fit_mask')
            #print(len(pp.bestfit),len(stars_f),len(np.array(comp_vals['gas_narrow_1f'])),len(np.array(comp_vals['gas_narrow_2f'])),
             #            len(np.array(comp_vals['gas_broad_1f'])),len(np.array(comp_vals['gas_broad_2f'])),
             #     len(np.array(comp_vals['gas_Civ_1f'])),
             #            len(np.array(comp_vals['gas_Civ_2f'])),len(np.array(comp_vals['Bal_cont_f'])),len(np.array(comp_vals['fe2_f'])),
             #            len(np.array(comp_vals['agn_f'])),len(residuals),len(ivar),len(lam_gal),len(wdisp),len(fit_mask))

            
            tcomp=Table([pp.bestfit,stars_f,np.array(comp_vals['gas_narrow_1f']),np.array(comp_vals['gas_narrow_2f']),\
                         np.array(comp_vals['gas_broad_1f']),np.array(comp_vals['gas_broad_2f']),np.array(comp_vals['gas_Civ_1f']),\
                         np.array(comp_vals['gas_Civ_2f']),np.array(comp_vals['Bal_cont_f']),np.array(comp_vals['fe2_f']),\
                         np.array(comp_vals['agn_f']),residuals,ivar,lam_gal,wdisp,fit_mask],names=my_comp_names)
            print('Saved model 2')
            ###############
            #Kinematic results
            names1=('Component','velocity(Km/s)', 'velocity dispersion(Km/s)','h3','h4')
            names11=np.array(['stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_broad_2',
                              'Bal_Cont','Fe_II','AGN'])
            if nClines>0:
                names11=np.array(['stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_broad_2','Gas_Civ','Gas_Civ_2',
                              'Bal_Cont','Fe_II','AGN'])
        #Generate the vel array
            vels=[]
            vels_disp=[]
            mnh3=[]
            mnh4=[]
            len_comp=len(names11)
            if nTemps==0:
                vels=[-9999999]
                vels_disp=[-9999999]
                mnh3=[-9999999]
                mnh4=[-9999999]
                len_comp=len_comp-1
            for jj in range(len_comp):
                vels=np.append(vels,pp.sol[jj][0])
                vels_disp=np.append(vels_disp,pp.sol[jj][1])
                try:
                    mnh3=np.append(mnh3,pp.sol[jj][2])
                    mnh4=np.append(mnh4,pp.sol[jj][3])
                except:
                    mnh3=np.append(mnh3,0.0)
                    mnh4=np.append(mnh4,0.0)

            #Add the two broad components to get the FWHM of the composition:
            #Look for a BEL with weight>0 in herarchical order
            
      
            bels_h_names=['Hbeta','Halpha','MgII2798.75'] 
            for belsn in bels_h_names:
                try:
                    indx_broad11=np.where(Bline_names==belsn)
                    indx_broad1=indx_broad11[0][0]
                    wb1=pp.weights[nTemps+nLines1+nLines2+indx_broad1]
                    wb2=pp.weights[nTemps+nLines1+nLines2+nBlines+indx_broad1]
                except:
                    two_comp_track=0
                    continue

                if wb1>0 or wb2>0:
                    two_comp_track=1 #To check if we have the listed emission lines
                    rt_wb=np.min([wb1,wb2])/np.max([wb1,wb2])
                    if rt_wb>=0.1: #Check that the second component is at least 10% as intense as the most intense emission
                        br111=pp.matrix[:,nTemps+nLines1+nLines2+indx_broad1].dot(pp.weights[nTemps+nLines1+nLines2+indx_broad1])
                
                        br112=pp.matrix[:,nTemps+nLines1+nLines2+nBlines+indx_broad1].dot(pp.weights[nTemps+nLines1
                                                                                                         +nLines2+nBlines+indx_broad1])
                            
                        btotal=br111+br112
                        bels_wv=Bline_wave[indx_broad1]
                        fwhm_two_cop=fwhm_tc(lam_gal,btotal,cw=bels_wv)
                    
                        vels=np.append(vels,-9999999)
                        vels_disp=np.append(vels_disp,fwhm_two_cop/2.35)
                        mnh3=np.append(mnh3,0.0)
                        mnh4=np.append(mnh4,0.0)
                        names11=np.append(names11,'Tot_broad')
                        #break
                    else:
                        if wb1>wb2:
                            indx_tbw=np.where(names11=='Gas_broad')[0][0]
                        else:
                            indx_tbw=np.where(names11=='Gas_broad_2')[0][0]
                        if nTemps==0:
                            indx_tbw=indx_tbw-1

                        vels=np.append(vels,pp.sol[indx_tbw][0])
                        vels_disp=np.append(vels_disp,pp.sol[indx_tbw][1])
                        mnh3=np.append(mnh3,pp.sol[indx_tbw][2])
                        mnh4=np.append(mnh4,pp.sol[indx_tbw][3])
                        names11=np.append(names11,'Tot_broad')
                    #Create the table       
                    tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                    break
                else:
                    two_comp_track=0 #To check if we have the listed emission lines
                    tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
            if two_comp_track==0 and nClines>0:
                bels_civ_names=['CIV1548','CIV1550'] 
                #print(bels_civ_names)
                for belcn in bels_civ_names:
                   # print('into loop')
                   # print(Cline_names)
                    try:
                        indx_broadC11=np.where(Cline_names==belcn)
                        indx_broadC1=indx_broadC11[0][0]
                        wb1=pp.weights[nTemps+nLines1+nLines2+2*nBlines+indx_broadC1]
                        wb2=pp.weights[nTemps+nLines1+nLines2+2*nBlines+nClines+indx_broadC1]
                      #  print('Civ',wb1,wb2)
                    except:
                        continue
    
                    if wb1>0 or wb2>0:
                        rt_wb=np.min([wb1,wb2])/np.max([wb1,wb2])
                        if rt_wb>=0.1:
                            br111=pp.matrix[:,nTemps+nLines1+nLines2+2*nBlines+indx_broadC1].dot(pp.weights[nTemps+nLines1+\
                                                                                                 nLines2+2*nBlines+indx_broadC1])
                    
                            br112=pp.matrix[:,nTemps+nLines1+nLines2+2*nBlines+nClines+indx_broad1].dot(pp.weights[nTemps+
                                                                                                        nLines1+nLines2+2*nBlines
                                                                                                        +nClines+indx_broad1])
                                
                            btotal=br111+br112
                            bels_wv=Cline_wave[indx_broadC1]
                            fwhm_two_cop=fwhm_tc(lam_gal,btotal,cw=bels_wv)
                        
                            vels=np.append(vels,-9999999)
                            vels_disp=np.append(vels_disp,fwhm_two_cop/2.35)
                            mnh3=np.append(mnh3,0.0)
                            mnh4=np.append(mnh4,0.0)
                            names11=np.append(names11,'Tot_broad')
                           # break
                        else:
                            if wb1>wb2:
                                indx_tbw=np.where(names11=='Gas_Civ')[0][0]
                            else:
                                indx_tbw=np.where(names11=='Gas_Civ_2')[0][0]
                            if nTemps==0:
                                indx_tbw=indx_tbw-1
    
                            vels=np.append(vels,pp.sol[indx_tbw][0])
                            vels_disp=np.append(vels_disp,pp.sol[indx_tbw][1])
                            mnh3=np.append(mnh3,pp.sol[indx_tbw][2])
                            mnh4=np.append(mnh4,pp.sol[indx_tbw][3])
                            names11=np.append(names11,'Tot_broad')
                        #Create the table       
                        tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                        break
                    else:
                        tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
            else:
                tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)



        #Now consider the different models
        elif model_name=='model_st_3' or model_name=='model_nst_3':
            my_comp_names=('Best-fit','stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_Civ','winds','Bal_Cont',
            'Fe_II','AGN','residuals','ivar','wave','wdisp','fit_mask')
            tcomp=Table([pp.bestfit,stars_f,np.array(comp_vals['gas_narrow_1f']),np.array(comp_vals['gas_narrow_2f']),
                    np.array(comp_vals['gas_broad_1f']),np.array(comp_vals['gas_Civ_1f']),np.array(comp_vals['winds_f']),
                         np.array(comp_vals['Bal_cont_f']),np.array(comp_vals['fe2_f']),np.array(comp_vals['agn_f']),
                         residuals,ivar,lam_gal,wdisp,fit_mask],names=my_comp_names)
    
            ###############
            #Kinematic results
            names1=('Component','velocity(Km/s)', 'velocity dispersion(Km/s)','h3','h4')
            names11=np.array(['stars','Gas_narrow1','Gas_narrow2','Gas_broad','winds','Bal_Cont','Fe_II','AGN'])
            if nClines>0:
                names11=np.array(['stars','Gas_narrow1','Gas_narrow2','Gas_broad','Gas_Civ','winds','Bal_Cont','Fe_II','AGN'])
        #Generate the vel array
            vels=[]
            vels_disp=[]
            mnh3=[]
            mnh4=[]
            len_comp=len(names11)
            if nTemps==0:
                vels=[-9999999]
                vels_disp=[-9999999]
                mnh3=[-9999999]
                mnh4=[-9999999]
                len_comp=len_comp-1
            
            for jj in range(len_comp):
                vels=np.append(vels,pp.sol[jj][0])
                vels_disp=np.append(vels_disp,pp.sol[jj][1])
                try:
                    mnh3=np.append(mnh3,pp.sol[jj][2])
                    mnh4=np.append(mnh4,pp.sol[jj][3])
                except:
                    mnh3=np.append(mnh3,0.0)
                    mnh4=np.append(mnh4,0.0)
            tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)

        elif model_name=='model_st_4' or model_name=='model_nst_4':
            my_comp_names=('Best-fit','stars','Gas_narrow_1','Gas_narrow_2','Gas_broad','Gas_broad_2','Gas_Civ','Gas_Civ_2',
            'winds','Bal_Cont','Fe_II','AGN','residuals','ivar','wave','wdisp','fit_mask')
            tcomp=Table([pp.bestfit,stars_f,np.array(comp_vals['gas_narrow_1f']),np.array(comp_vals['gas_narrow_2f']),
                    np.array(comp_vals['gas_broad_1f']),np.array(comp_vals['gas_broad_2f']),np.array(comp_vals['gas_Civ_1f']),
                    np.array(comp_vals['gas_Civ_2f']),np.array(comp_vals['winds_f']),
                    np.array(comp_vals['Bal_cont_f']),np.array(comp_vals['fe2_f']),np.array(comp_vals['agn_f']),
                        residuals,ivar,lam_gal,wdisp,fit_mask],names=my_comp_names)
    
            ###############
            #Kinematic results
            names1=('Component','velocity(Km/s)', 'velocity dispersion(Km/s)','h3','h4')
            names11=np.array(['stars','Gas_narrow1','Gas_narrow2','Gas_broad','Gas_broad_2',
                              'winds','Bal_Cont','Fe_II','AGN'])
            if nClines>0:
                names11=np.array(['stars','Gas_narrow1','Gas_narrow2','Gas_broad','Gas_broad_2','Gas_Civ','Gas_Civ_2'
                              'winds','Bal_Cont','Fe_II','AGN'])
        #Generate the vel array
            vels=[]
            vels_disp=[]
            mnh3=[]
            mnh4=[]
            len_comp=len(names11)
            if nTemps==0:
                vels=[-9999999]
                vels_disp=[-9999999]
                mnh3=[-9999999]
                mnh4=[-9999999]
                len_comp=len_comp-1
            for jj in range(len_comp):
                vels=np.append(vels,pp.sol[jj][0])
                vels_disp=np.append(vels_disp,pp.sol[jj][1])
                try:
                    mnh3=np.append(mnh3,pp.sol[jj][2])
                    mnh4=np.append(mnh4,pp.sol[jj][3])
                except:
                    mnh3=np.append(mnh3,0.0)
                    mnh4=np.append(mnh4,0.0)

            #Add the two broad components to get the FWHM of the composition:
            #Look for a BEL with weight>0 in herarchical order
            
      
            bels_h_names=['Hbeta','Halpha','MgII2798.75'] 
            for belsn in bels_h_names:
                try:
                    indx_broad11=np.where(Bline_names==belsn)
                    indx_broad1=indx_broad11[0][0]
                    wb1=pp.weights[nTemps+nLines1+nLines2+indx_broad1]
                    wb2=pp.weights[nTemps+nLines1+nLines2+nBlines+indx_broad1]
                except:
                    two_comp_track=0
                    continue

                if wb1>0 or wb2>0:
                    two_comp_track=1 #To check if we have the listed emission lines
                    rt_wb=np.min([wb1,wb2])/np.max([wb1,wb2])
                    if rt_wb>=0.1: #Check that the second component is at least 10% as intense as the most intense emission
                        br111=pp.matrix[:,nTemps+nLines1+nLines2+indx_broad1].dot(pp.weights[nTemps+nLines1+nLines2+indx_broad1])
                
                        br112=pp.matrix[:,nTemps+nLines1+nLines2+nBlines+indx_broad1].dot(pp.weights[nTemps+nLines1
                                                                                                         +nLines2+nBlines+indx_broad1])
                            
                        btotal=br111+br112
                        bels_wv=Bline_wave[indx_broad1]
                        fwhm_two_cop=fwhm_tc(lam_gal,btotal,cw=bels_wv)
                    
                        vels=np.append(vels,-9999999)
                        vels_disp=np.append(vels_disp,fwhm_two_cop/2.35)
                        mnh3=np.append(mnh3,0.0)
                        mnh4=np.append(mnh4,0.0)
                        names11=np.append(names11,'Tot_broad')
                        #break
                    else:
                        if wb1>wb2:
                            indx_tbw=np.where(names11=='Gas_broad')[0][0]
                        else:
                            indx_tbw=np.where(names11=='Gas_broad_2')[0][0]
                        if nTemps==0:
                            indx_tbw=indx_tbw-1

                        vels=np.append(vels,pp.sol[indx_tbw][0])
                        vels_disp=np.append(vels_disp,pp.sol[indx_tbw][1])
                        mnh3=np.append(mnh3,pp.sol[indx_tbw][2])
                        mnh4=np.append(mnh4,pp.sol[indx_tbw][3])
                        names11=np.append(names11,'Tot_broad')
                    #Create the table       
                    tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                    break
                else:
                    tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                    two_comp_track=0 #To check if we have the listed emission lines
                    
            if two_comp_track==0 and nClines>0:
                bels_civ_names=['CIV1548','CIV1550'] 
                for belcn in bels_civ_names:
                    try:
                        indx_broadC11=np.where(Cline_names==belcn)
                        indx_broadC1=indx_broadC11[0][0]
                        wb1=pp.weights[nTemps+nLines1+nLines2+2*nBlines+indx_broadC1]
                        wb2=pp.weights[nTemps+nLines1+nLines2+2*nBlines+nClines+indx_broadC1]
                    except:
                        continue
    
                    if wb1>0 or wb2>0:
                        rt_wb=np.min([wb1,wb2])/np.max([wb1,wb2])
                        if rt_wb>=0.1:
                            br111=pp.matrix[:,nTemps+nLines1+nLines2+2*nBlines+indx_broadC1].dot(pp.weights[nTemps+nLines1+\
                                                                                                 nLines2+2*nBlines+indx_broadC1])
                    
                            br112=pp.matrix[:,nTemps+nLines1+nLines2+2*nBlines+nClines+indx_broad1].dot(pp.weights[nTemps+
                                                                                                        nLines1+nLines2+2*nBlines
                                                                                                        +nClines+indx_broad1])
                                
                            btotal=br111+br112
                            bels_wv=Cline_wave[indx_broadC1]
                            fwhm_two_cop=fwhm_tc(lam_gal,btotal,cw=bels_wv)
                        
                            vels=np.append(vels,-9999999)
                            vels_disp=np.append(vels_disp,fwhm_two_cop/2.35)
                            mnh3=np.append(mnh3,0.0)
                            mnh4=np.append(mnh4,0.0)
                            names11=np.append(names11,'Tot_broad')
                           # break
                        else:
                            if wb1>wb2:
                                indx_tbw=np.where(names11=='Gas_Civ')[0][0]
                            else:
                                indx_tbw=np.where(names11=='Gas_Civ_2')[0][0]
                            if nTemps==0:
                                indx_tbw=indx_tbw-1
    
                            vels=np.append(vels,pp.sol[indx_tbw][0])
                            vels_disp=np.append(vels_disp,pp.sol[indx_tbw][1])
                            mnh3=np.append(mnh3,pp.sol[indx_tbw][2])
                            mnh4=np.append(mnh4,pp.sol[indx_tbw][3])
                            names11=np.append(names11,'Tot_broad')
                        #Create the table       
                        tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                        break
                    else:
                        tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)
                        
            else:
                tkr=Table([names11,vels,vels_disp,mnh3,mnh4],names=names1)

    ###################
    ######Stars Templates with weight>0 Table
        names3=('Template','weight')
     #generate the template weight mask
        if nTemps>0:
            sww=pp.weights[:nTemps]
            swmk=(sww>0)
         #Template names array
            stnm=st_names[swmk]
            #loop to extract only the name and not the directory
            rstnm=[]
            for vdnm in stnm:
                otnm=vdnm.split(sep='/')
                rstnm=np.append(rstnm,otnm[-1])
         #weights array
            swar=sww[swmk]
        else:
            rstnm=['no_ssp']
            swar=[-9999999]
        tst=Table([rstnm,swar],names=names3)

#AGN Templates with weight>0 Table
        names6=('AGN_Template','weight')
     #generate the template weight mask
        wagn=pp.weights[-nAgns:]
        agnmk=(wagn>0)
     #AGN templates names and weights arrays
        nagna=slopes[agnmk]
        nwagn=wagn[agnmk]
        tagn=Table([nagna,nwagn],names=names6)
        

####################################### fluxes and EW
        #Continuum 
        if nBlines2>0:
            Bels2=np.array(comp_vals['gas_broad_2f'])
            Civ2=np.array(comp_vals['gas_Civ_2f'])
        else:
            Bels2=0*np.array(comp_vals['gas_broad_2f'])
            Civ2=0*np.array(comp_vals['gas_Civ_2f'])
        if nwinds2>0:
            xwnds=np.array(comp_vals['winds_f'])
        else:
            xwnds=0*np.array(comp_vals['winds_f'])
            
        F_cont=(pp.bestfit-np.array(comp_vals['gas_narrow_1f'])-np.array(comp_vals['gas_narrow_2f'])-
                    np.array(comp_vals['gas_broad_1f'])-Bels2-np.array(comp_vals['gas_Civ_1f'])-Civ2-xwnds)*scale_fact
        #Narrow lines
        Nrr_flux=[]
        Nrr_EW=[]
        line_names=np.append(line_names1,line_names2)
        line_wave=np.append(line_wave1,line_wave2)   
        names8=('Line','wave','Flux','EW')
        for t in range(len(line_names)):
            L_flux=pp.matrix[:, nTemps+t].dot(pp.weights[nTemps+t])
            L_flux=L_flux*scale_fact
            I_flux=np.trapezoid(L_flux,x=lam_gal)
            Nrr_flux.append(I_flux)
            #EW
            nnL_flux=L_flux+F_cont
            nF_int=1-(nnL_flux/F_cont)
            ew_mask=(lam_gal>line_wave[t]-1000)&(lam_gal<line_wave[t]+1000)
            mFx=nF_int[ew_mask]
            ew_lam=lam_gal[ew_mask]
            EW_nrr=np.absolute(np.trapezoid(mFx,x=ew_lam))
            Nrr_EW.append(EW_nrr)
            
        
        tnr=Table([line_names,line_wave,Nrr_flux,Nrr_EW],names=names8)

        #Broad Lines
        Br_flux=[]
        Br_EW=[]
        names9=('Line','wave','Flux','EW')
        if nClines>0:
            Bline_names=np.append(Bline_names,Cline_names)
            Bline_wave=np.append(Bline_wave,Cline_wave)
        for t in range(len(Bline_names)):
            L_flux=pp.matrix[:, nTemps+nLines1+nLines2+t].dot(pp.weights[nTemps+nLines1+nLines2+t])
            L_flux=L_flux*scale_fact
            I_flux=np.trapezoid(L_flux,x=lam_gal)
            Br_flux.append(I_flux)
        ##EW####
            nL_flux=L_flux+F_cont
            F_int=1-(nL_flux/F_cont)
            ew_mask=(lam_gal>Bline_wave[t]-1000)&(lam_gal<Bline_wave[t]+1000)
            mFx=F_int[ew_mask]
            ew_lam=lam_gal[ew_mask]
            EW_b=np.absolute(np.trapezoid(mFx,x=ew_lam))
            Br_EW.append(EW_b)
               
        ###### Broad Second Comp
        if nBlines2>0:
            sc_Bline_names=[]
            #F_cont=(pp.bestfit-gas_narrow-gas_broad-gas_broadSc-Bch-Fe2)*scale_factor
            for t in range(len(Bline_names)):
                L_flux=pp.matrix[:, nTemps+nLines1+nLines2+nBlines+nClines+t].dot(pp.weights[nTemps+nLines1+nLines2+nBlines+nClines+t])
                L_flux=L_flux*scale_fact
                I_flux=np.trapezoid(L_flux,x=lam_gal)
                Br_flux.append(I_flux)
            ##EW####
                nL_flux=L_flux+F_cont
                F_int=1-(nL_flux/F_cont)
                ew_mask=(lam_gal>Bline_wave[t]-1000)&(lam_gal<Bline_wave[t]+1000)
                mFx=F_int[ew_mask]
                ew_lam=lam_gal[ew_mask]
                EW_b=np.absolute(np.trapezoid(mFx,x=ew_lam))
                Br_EW.append(EW_b)
            
                sc_Bline_names.append(Bline_names[t]+'_2')
            Bline_wave22=np.append(Bline_wave,Bline_wave)
            Bline_names22=np.append(Bline_names,sc_Bline_names)
            tewb=Table([Bline_names22,Bline_wave22,Br_flux,Br_EW],names=names9)
        else:
            tewb=Table([Bline_names,Bline_wave,Br_flux,Br_EW],names=names9)        
        
        ####################################### Wind lines fluxes
        wind_flux=[]
        EW_wd=[]
        names8=('Line','wave','Flux','EW')
        if nwinds2>0:
            for t in range(len(wind_line_names)):
                L_flux=pp.matrix[:, nTemps+nLines1+nLines2+nBlines+nBlines2+t].dot(pp.weights[nTemps+nLines1+nLines2+nBlines+nBlines2+t])
                L_flux=L_flux*scale_fact
                I_flux=np.trapezoid(L_flux,x=lam_gal)
                wind_flux.append(I_flux)

                nL_flux=L_flux+F_cont
                F_int=1-(nL_flux/F_cont)
                ew_mask=(lam_gal>wind_line_wave[t]-1000)&(lam_gal<wind_line_wave[t]+1000)
                mFx=F_int[ew_mask]
                ew_lam=lam_gal[ew_mask]
                EW_w=np.absolute(np.trapezoid(mFx,x=ew_lam))
                EW_wd.append(EW_w)
            
        else:
            wind_flux=np.full((len(wind_line_names)),-9999999)
            EW_wd=np.full((len(wind_line_names)),-9999999)

        tfwd=Table([wind_line_names,wind_line_wave,wind_flux,EW_wd],names=names8)
            


        #Transform all tablest to a BinTableHDU
        t111=fits.BinTableHDU(tcomp)
        t112=fits.BinTableHDU(tkr)
        t113=fits.BinTableHDU(tst)
        t114=fits.BinTableHDU(tagn)
        t115=fits.BinTableHDU(tnr)
        t116=fits.BinTableHDU(tewb)
        t117=fits.BinTableHDU(tfwd)
        hdutt=fits.HDUList([hdu,t111,t112,t113,t114,t115,t116,t117])

        ########Name of fits archival
        #file_name fro SDSS-V spec or fit-spec from SDSS-V
        
        if fit_type is not None:
            nm1=fit_type+'-'+n2[0]
        else:
            nm1='fit-sc-'+n2[0]
            
        nt1=file_dir.split(sep='/') #Consider all cases
        save_dir=nt1[0]
        for ns_fl in range(1,len(nt1)-1):
            save_dir=save_dir+'/'+nt1[ns_fl]

        if wts>0:
            nm1='fit-ff-'+n2[0]
            save_dir=save_dir+'/'+nm1+'_model_1_ff.fits'    
        else:
            save_dir=save_dir+'/'+nm1+'.fits'

        if fit_type=='sim':
            save_dir_sim=save_dir.split(sep=nm1)
            save_dir=save_dir_sim[0]+n2[0]+'/'
           # print(save_dir)
            if os.path.exists(save_dir):
                sim_fls=glob.glob(save_dir+'sim*.fits')
                sim_number=len(sim_fls)+1
                save_dir=save_dir+'sim'+str(sim_number)+'-'+n2[0]+'.fits'
            else:
                cmd='mkdir '+save_dir
                os.system(cmd)
                save_dir=save_dir+'sim1'+'-'+n2[0]+'.fits'

        hdutt.writeto(save_dir, overwrite=True)
        print('Results saved in:'+save_dir)










        
