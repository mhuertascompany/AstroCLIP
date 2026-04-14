import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import shutil
import database_v7 as db
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u          
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

# Une colonne pour chaque type avec 0/1 dans les visual morpho à transformer en une seule colonne qui liste le type
def assign_type(row): 
    if row['ELL_REGULAR'] == 1:
        return 'ELL_REGULAR'
    elif row['ELL_DISTURB'] == 1:
        return 'ELL_DISTURB'
    elif row['ELL_INTER']==1:
        return 'ELL_INTER'
    elif row['S0_REGULAR'] == 1:
        return 'S0_REGULAR'
    elif row['S0_INTER'] == 1:
        return 'SO_INTER'
    elif row['S0_DISTURB'] == 1:
        return 'S0_DISTURB'
    elif row['EDISK_REGULAR'] == 1:
        return 'EDISK_REGULAR'
    elif row['EDISK_INTER'] == 1:
        return 'EDISK_INTER'
    elif row['EDISK_DISTURB'] == 1:
        return 'EDISK_DISTURB'
    elif row['LDISK_REGULAR'] == 1:
        return 'LDISK_REGULAR'
    elif row['LDISK_INTER'] == 1:
        return 'LDISK_INTER'
    elif row['LDISK_DISTURB'] == 1:
        return 'LDISK_DISTURB'
    else:
        return 'UNKNOWN'

# Read the database with the visual classification
def read_visual_morpho():
  
  outputdb  = db.AutoUpdateSQLiteTable('visualmorpho_COSMOSWeb_v7.db', 'morphology')
  results = outputdb.get_all_rows()
  # convert in panda dataframe, associating the right name of the columns
  col_names = [
    'id',
    'FAKE', 'PHOTOMETRY_OFF', 'SERSIC_OFF', 'SUBCOMPONENT', 'BLENDED',
    'TOO_FAINT', 'TOO_SMALL', 'UNCERTAIN', 'BRIGHT_FOREGROUND',
    'ELL_REGULAR', 'ELL_INTER', 'ELL_DISTURB',
    'S0_REGULAR', 'S0_INTER', 'S0_DISTURB',
    'EDISK_REGULAR', 'EDISK_INTER', 'EDISK_DISTURB',
    'LDISK_REGULAR', 'LDISK_INTER', 'LDISK_DISTURB',
    'EDGE_ON', 'ASYMETRY', 'ARMS', 'BAR', 'LSB_DISK',
    'CLUMP', 'MANY_CLUMPS', 'IS_A_CLUMP', 'CHAIN', 'COMPACT',
    'IRR', 'POINT_LIKE', 'POWERLAW',
    'MINOR_MERGER', 'MINOR_CLOSE', 'MINOR_PAIR',
    'MAJOR_MERGER', 'MAJOR_CLOSE', 'MAJOR_PAIR',
    'IS_SMALL_COMPANION', 'CONSISTENT_Z', 'DRY', 'REMNANT',
    'LENS', 'GROUPE', 'INFO', 'ADDI', 'VERSION'
  ]
  df_visual_morpho = pd.DataFrame(results, columns=col_names)

  # Define column type_visual
  df_visual_morpho['type_visual'] = df_visual_morpho.apply(assign_type, axis=1)
    
  #print("VISUAL ",df_visual_morpho.head())
  return df_visual_morpho


def load_selected_columns(hdu_data, selected_cols):
    table = Table(hdu_data)
    uni_cols = [col for col in table.colnames if len(table[col].shape) <= 1]
    valid_cols = [col for col in selected_cols if col in uni_cols]
    return table[valid_cols].to_pandas()



# read the huge master COSMOS2025 catalogue
def read_cosmos2025():

 # Selection uniquement de certaines données à spécifier de COSMOS-Web, sinon le temps d'exécution est trop long
 photom_cols = ['id','tile','ra','dec','a_image','b_image','theta_image','theta_world','mag_auto_hst-f814w','mag_auto_f115w','mag_auto_f150w','mag_auto_f277w','mag_auto_f444w','mag_auto_f770w','flag_star','flag_blend','ra_model','dec_model','radius_sersic','radius_sersic_err','axratio_sersic','axratio_sersic_err','sersic','sersic_err','angle_sersic','angle_sersic_err','e1','e1_err','e2','e2_err','mag_model_f115w','mag_model_f150w','mag_model_f277w','mag_model_f444w','mag_model_hst-f814w','mag_model_f770w','warn_flag']
 lephare_cols = ['zfinal','type','zpdf_med','zpdf_l68','zpdf_u68','zchi2','chi2_best','nbfilt','zp_agn','chi2_agn','mod_agn','mod_star','chi_star','mod_minchi2_phys','ebv_minchi2','law_minchi2','age_minchi2','mass_minchi2','sfr_minchi2','ssfr_minchi2','age_l68','age_med','age_u68','mass_l68','mass_med','mass_u68','sfr_l68','sfr_med','sfr_u68','ssfr_l68','ssfr_med','ssfr_u68']
 cigale_cols = ['mass','sfr_100myr','sfr_inst']
 bd_cols = ['ra_detec_bd','dec_detec_bd','disk_radius_deg','disk_radius_deg_err','bulge_radius_deg','bulge_radius_deg_err','angle_bd','angle_bd_err','disk_axratio','disk_axratio_err','bulge_axratio','bulge_axratio_err','b/t_f115w','b/t_f150w','b/t_f277w','b/t_f444w','b/t_err_f115w','b/t_err_f150w','b/t_err_f277w','b/t_err_f444w','b/t_hst-f814w','b/t_f770w','b/t_err_hst-f814w','b/t_err_f770w',]
    
 # Read the master catalogue (probably old version given the name lp_zfinal
 with fits.open('COSMOSWeb_mastercatalog_v1.fits', memmap=True) as hdu:
    cat_photom  = load_selected_columns(hdu[1].data, photom_cols)
    cat_lephare = load_selected_columns(hdu[2].data, lephare_cols)
    cat_cigale  = load_selected_columns(hdu[4].data, cigale_cols)
    cat_BD      = load_selected_columns(hdu[6].data, bd_cols)
    flux_aper_f150w_1 = (hdu[1].data['flux_aper_f150w'])[:,1]
    flux_aper_f150w_2 = (hdu[1].data['flux_aper_f150w'])[:,2]
    flux_aper_f277w_1 = (hdu[1].data['flux_aper_f277w'])[:,1]
    flux_aper_f277w_2 = (hdu[1].data['flux_aper_f277w'])[:,2]
    flux_aper_f444w_1 = (hdu[1].data['flux_aper_f444w'])[:,1]
    flux_aper_f444w_2 = (hdu[1].data['flux_aper_f444w'])[:,2]
 
 # Merge
 df_cosmosweb_cat= pd.concat([
    cat_photom.reset_index(drop=True),
    cat_lephare.reset_index(drop=True),
    cat_cigale.reset_index(drop=True),
    cat_BD.reset_index(drop=True)
 ], axis=1)
 # Add the column ssfr_100myr
 df_cosmosweb_cat = df_cosmosweb_cat.assign(ssfr_100myr=lambda x: x['sfr_100myr'] / x['mass'])
 # Add concentrations
 df_cosmosweb_cat = df_cosmosweb_cat.assign(concentration150=lambda x: (flux_aper_f150w_1 / flux_aper_f150w_2))
 df_cosmosweb_cat = df_cosmosweb_cat.assign(concentration277=lambda x: (flux_aper_f277w_1 / flux_aper_f277w_2))
 df_cosmosweb_cat = df_cosmosweb_cat.assign(concentration444=lambda x: (flux_aper_f444w_1 / flux_aper_f444w_2))
 # radius in arcsec
 df_cosmosweb_cat = df_cosmosweb_cat.assign(radius_arcsec=lambda x: x['radius_sersic']*3600.)
 

    
 #print("Read cosmos2025 ",df_cosmosweb_cat.head())
 return df_cosmosweb_cat 


# Read the marc Huertas catalogue
def read_CNN_morpho():

  hdul = fits.open('morphology_v7.fits')
  table = Table(hdul[1].data)
  incat_morpho = table.to_pandas()

  # Add the column with the number of Irregular CNN as a first choice
  first_CNN   = np.column_stack(((incat_morpho['morph_flag_f150w']),(incat_morpho['morph_flag_f277w']),(incat_morpho['morph_flag_f444w'])))
  count_CNN_Irr  = np.sum(first_CNN == 2, axis=1)
  incat_morpho = incat_morpho.assign(CNN_N_Irr=lambda x: count_CNN_Irr)
  #print("inside CNN",incat_morpho.shape)
 
  #print("Read compilation",incat_morpho.head())
  return incat_morpho

# Read the parametric Lilang morphological parameters
def read_lilang_morpho():

  hdul = fits.open('CW_sersic_cas_matchsepp3.1.0.fits')
  table = Table(hdul[1].data)
  uni_cols = [col for col in table.colnames if len(table[col].shape) <= 1]
  cas_cols = ['SE++_source_id', 'nsersic_f115w', 'asymmetry_f115w', 'smoothness_f115w','concentration_f115w', 'gini_f115w', 'm20_f115w', 'cas_flag_f115w', 'nsersic_f150w', 'asymmetry_f150w', 'smoothness_f150w','concentration_f150w', 'gini_f150w', 'm20_f150w', 'cas_flag_f150w', 'nsersic_f277w', 'asymmetry_f277w', 'smoothness_f277w','concentration_f277w', 'gini_f277w', 'm20_f277w', 'cas_flag_f277w', 'nsersic_f444w', 'asymmetry_f444w', 'smoothness_f444w','concentration_f444w', 'gini_f444w', 'm20_f444w', 'cas_flag_f444w']
  valid_cols = [col for col in cas_cols if col in uni_cols]
  incat_cas = table[valid_cols].to_pandas()
  incat_cas = incat_cas.rename(columns={
    'SE++_source_id': 'id',
  })
    
  #print("Read lilang ",incat_cas.head())
  return incat_cas

def find_neighbour(incat):
    
  ra = u.Quantity(incat['ra'], unit=u.deg)
  dec = u.Quantity(incat['dec'], unit=u.deg)
  c_master = SkyCoord(ra=ra, dec=dec)

  idxself_1, d2dself_1, d3dself_1 = c_master.match_to_catalog_sky(c_master, nthneighbor=2)
  sepself_1 = c_master.separation(c_master[idxself_1])
  sepself_1.arcsec

  idxself_2, d2dself_2, d3dself_2 = c_master.match_to_catalog_sky(c_master, nthneighbor=3)
  sepself_2 = c_master.separation(c_master[idxself_2])
  sepself_2.arcsec
  print(len(idxself_1),len(idxself_2))

  z_ref = incat['zfinal'].values  
  z_neighbour_1 = incat['zfinal'].values[idxself_1]  
  z_neighbour_2 = incat['zfinal'].values[idxself_2]

  # Masques 
  mask_invalid_1 = (z_ref <= 0) | (z_neighbour_1 <= 0) | (z_ref > 20) | (z_neighbour_1 > 20)
  mask_invalid_2 = (z_ref <= 0) | (z_neighbour_2 <= 0) | (z_ref > 20) | (z_neighbour_2 > 20)

  # Calcul des différences (delta_z = 99 si invalide)
  delta_z_1 = np.where(mask_invalid_1, 99, np.abs(z_neighbour_1 - z_ref))
  delta_z_2 = np.where(mask_invalid_2, 99, np.abs(z_neighbour_2 - z_ref))

  # Masque pour sélectionner le meilleur voisin
  mask = delta_z_1 < delta_z_2
  # Sélection des propriétés du meilleur voisin
  idx_best    = np.where(mask, idxself_1, idxself_2)
  sep_best    = np.where(mask, sepself_1.arcsec, sepself_2.arcsec)
  z_best      = np.where(mask, z_neighbour_1, z_neighbour_2)
  z_delta     = np.where(mask, delta_z_1, delta_z_2)
  mass_best   = np.where(mask, incat['mass_med'][idxself_1], incat['mass_med'][idxself_2])
  mag277_best = np.where(mask, incat['mag_model_f277w'][idxself_1], incat['mag_model_f277w'][idxself_2])
  mag444_best = np.where(mask, incat['mag_model_f444w'][idxself_1], incat['mag_model_f444w'][idxself_2])
  ra_best     = np.where(mask, incat['ra'][idxself_1], incat['ra'][idxself_2])
  dec_best    = np.where(mask, incat['dec'][idxself_1], incat['dec'][idxself_2])
  radius_best = np.where(mask, incat['radius_sersic'][idxself_1], incat['radius_sersic'][idxself_2])

  # Crée le DataFrame final
  inter_self = pd.DataFrame({
    'id': incat['id'].values,
    'id_neighbour': idx_best,
    'separation_neighbour': sep_best,
    'zfinal_neighbour': z_best,
    'deltaz_neighbour': z_delta,
    'mass_neighbour': mass_best,
    'mag277_neighbour': mag277_best,
    'mag444_neighbour': mag444_best,
    'ra_neighbour': ra_best,
    'dec_neighbour': dec_best,
    'radius_neighbour': radius_best,
  })
  # Affiche les résultats
  #print("Id :")
  #print(inter_self['id'].values)
  #print("Id_neighbour :")
  #print(inter_self['id_neighbour'].values)
  #print("Séparations angulaires (arcsec) :")
  #print(inter_self['separation_neighbour'].values)
  #print("\nRayons des voisins :")
  #print(inter_self['radius_neighbour'].values)
  #print("\nDelta Redshifts des voisins :")
  #print(inter_self['deltaz_neighbour'].values)
  #print("Length self inter ",len(inter_self))

  return inter_self


# Read all the file and merge them according to the Id
def create_df_master():
    
 df_visual     = read_visual_morpho()
 df_CNN        = read_CNN_morpho()
 df_lilang     = read_lilang_morpho()
 df_cosmos2025 = read_cosmos2025()
 df_neighbour  = find_neighbour(df_cosmos2025)

 # merge catalogue based on Id
 df_master = df_cosmos2025.merge(df_visual, on='id', how='left')
 df_master = df_master.merge(df_lilang, on='id', how='left')
 df_master = df_master.merge(df_CNN, on='id', how='left')
 df_master = df_master.merge(df_neighbour, on='id', how='left')
 print("master",df_master.shape)
 print("cosmos2025",df_cosmos2025.shape)
 print("CNN",df_CNN.shape)
 print("visual",df_visual.shape)
 print("lilang",df_lilang.shape)
 print("neighbour",df_neighbour.shape)

 # Add new columns
 # Combine columns to try selecting disturbed galaxies (prob merger from Marc + asymetry + M20 from Lilan)
 df_master = df_master.assign(sum_disturb=lambda x: x['p_merger_mean']  + x['asymmetry_f277w'] + x['asymmetry_f444w'] + 0.5*x['m20_f277w'])

 # Calcul des distances to isolate mergers/close
 dist_close  = (df_master['separation_neighbour']) - 1.*3600.*((df_master['radius_sersic'])+(df_master['radius_neighbour']))
 dz_neighbour = np.abs((df_master['zfinal']) - (df_master['zfinal_neighbour']))
 dist_close = np.where(dz_neighbour > 0.5, 999, dist_close )
 dist_close = np.where(dist_close <0, 0, dist_close )

 # Assignation au DataFrame
 df_master = df_master.assign(dist_close=dist_close, dz_neighbour=dz_neighbour)
 df_master = df_master.assign(merger_metric=lambda x: x['p_merger_mean']*2. - x['dist_close'] )

 return df_master

