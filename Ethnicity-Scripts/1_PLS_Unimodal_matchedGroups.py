# %%
# %%
#libraries
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
# from datetime import date, datetime
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import scipy.stats as st
from scipy.stats import zscore as  zscore
from joblib import Parallel, delayed
# from multiprocessing import Manager
from sklearn.cross_decomposition import PLSRegression
import gc
import traceback
import logging
from sklearn.model_selection import KFold
import random


# %%
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 4
print(fold_num)
# fold_num = 0
# Paths
abcd_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/' #/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/ML_Tables/Ethnicity/' 
tables_path = root_dir
fold_base_path = root_dir + 'folds/tsp/'
fold_dir = fold_base_path  + 'Fold_' + str(fold_num) +'/'
if not os.path.isdir(fold_dir):
    os.mkdir(fold_dir) 
pls_dir = fold_dir + 'pls/' 
if not os.path.isdir(pls_dir):
    os.mkdir(pls_dir) 



# # Ensure subjects align
# assert len(subjects) == 5093, "Subject count mismatch!"


# %%
# Load features
features_name = 'abccCntr' # changes to run on other modalities (sMRI, FC, etc)
featuresd = joblib.load(abcd_dir + features_name + '.joblib')
demo = pd.read_csv(abcd_dir + 'demo_nesi.csv', index_col=0).dropna()
targs = pd.read_csv(abcd_dir + 'cog_all.csv', index_col=0, low_memory=False).dropna()
targs.index = targs.index.str.replace('_', '')
# %%
# Function to run PLS with standardization
def run_pls(path_out, target_name, pls_dict, x_tr, x_te, y_tr, y_te):
    try:
        # Standardize features
        scalerx = StandardScaler()
        x_tr_scaled = scalerx.fit_transform(x_tr.values)
        x_tr_scaleddf = pd.DataFrame(x_tr_scaled, index=x_tr.index)
        x_te_scaled = scalerx.transform(x_te.values)
        x_te_scaleddf = pd.DataFrame(x_te_scaled, index=x_te.index)
        scalery = StandardScaler()
        y_tr_scaled = scalery.fit_transform(y_tr.values)
        y_tr_scaleddf = pd.DataFrame(y_tr_scaled, index=y_tr.index)

        y_te_scaled = scalery.transform(y_te.values)
        y_te_scaleddf = pd.DataFrame(y_te_scaled, index=y_te.index)
        print(np.where(y_te_scaled == np.nan), np.where(y_tr_scaled == np.nan))
        # Performance dict
        perf = {'nmse': [], 'r2': [], 'pr': [], 'mae': []}
        
        # Set params: use all components if < 30, else up to 30
        nc = list(range(1, x_tr.shape[1] + 1, 1)) if x_tr.shape[1] < 30 else list(range(1, 31, 1))
        param_grid = {'n_components': nc, 'scale': [False]}  # Scaling handled manually
        
        # Set random seed
        seed = 42
        np.random.seed(seed)
        
        # Create PLS model
        pls = PLSRegression()
        # Perform grid search with 5-fold CV on training data
        grid_search = GridSearchCV(pls, param_grid, scoring='r2', 
                                cv=KFold(5, shuffle=True, random_state=seed))
        grid_search.fit(x_tr_scaled, y_tr_scaled)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Print results
        print(f"Fold {fold_num} - Best n_components: {grid_search.best_params_['n_components']}, "
            f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
        
        # Plot performance across components
        n_components = grid_search.cv_results_['param_n_components'].data
        mean_test_scores = grid_search.cv_results_['mean_test_score']

        # Transform and predict on training set
        tr_x_pls = pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index)
        tr_y_p = best_model.predict(x_tr_scaled)#.flatten()
        tr_y_pls = pd.DataFrame(tr_y_p, index=y_tr.index)
        
        # Training performance
        perf['nmse'].append(mean_squared_error(y_tr_scaled, tr_y_p))
        perf['r2'].append(r2_score(y_tr_scaled, tr_y_p))
        pearson_corr, _ = pearsonr(y_tr_scaled.flatten(), tr_y_p.flatten())
        perf['pr'].append(pearson_corr)
        perf['mae'].append(mean_absolute_error(y_tr_scaled, tr_y_p))
        
        # Transform and predict on test set
        te_x_pls = pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index)
        te_y_p = best_model.predict(x_te_scaled)#.flatten()
        te_y_pls = pd.DataFrame(te_y_p, index=y_te.index)
        
        # Test performance
        perf['nmse'].append(mean_squared_error(y_te_scaled, te_y_p))
        perf['r2'].append(r2_score(y_te_scaled, te_y_p))
        p_corr, _ = pearsonr(y_te_scaled.flatten(), te_y_p.flatten())
        perf['pr'].append(p_corr)
        perf['mae'].append(mean_absolute_error(y_te_scaled, te_y_p))
        
        # SelAA Test performance
        # if not (y_te_scaleddf.index.equals(y_te.index) and set(fold['selWA']).issubset(y_te_scaleddf.index)):
        #     print(key, 'Mismatch')
        
        y_te_scaled_aa = y_te_scaleddf.loc[test_AAidx].values
        te_y_p_aa = te_y_pls.loc[test_AAidx].values
        perf['nmse'].append(mean_squared_error(y_te_scaled_aa, te_y_p_aa))
        perf['r2'].append(r2_score(y_te_scaled_aa, te_y_p_aa))
        p_corr, _ = pearsonr(y_te_scaled_aa.flatten(), te_y_p_aa.flatten())
        perf['pr'].append(p_corr)
        perf['mae'].append(mean_absolute_error(y_te_scaled_aa, te_y_p_aa))

        # SelWA Test performance
        y_te_scaled_wa = y_te_scaleddf.loc[test_WAidx].values
        te_y_p_wa = te_y_pls.loc[test_WAidx].values
        perf['nmse'].append(mean_squared_error(y_te_scaled_wa, te_y_p_wa))
        perf['r2'].append(r2_score(y_te_scaled_wa, te_y_p_wa))
        p_corr, _ = pearsonr(y_te_scaled_wa.flatten(), te_y_p_wa.flatten())
        perf['pr'].append(p_corr)
        perf['mae'].append(mean_absolute_error(y_te_scaled_wa, te_y_p_wa))

	
        # Save performance
        performance = pd.DataFrame(perf, index=['train', 'test', 'test_aa', 'test_wa'])

        # fill output dict
        if key not in pls_dict:
            pls_dict[key] = {'model': {}, 'data': {}}
        pls_dict[key]['data'].update({'Xtrain': tr_x_pls, 'Xtest1': te_x_pls, 'yptrain': tr_y_pls, 'yptest': te_y_pls, 'yttrain': y_tr_scaleddf, 'yttest': y_te_scaleddf}) 
        pls_dict[key]['model'].update({'perf': performance, 'cv_results': grid_search.cv_results_, 'best_model': best_model}) 
        
        gc.collect()
    except Exception as e:
        logging.error(traceback.format_exc())



print(f"\nStarting Fold {fold_num}...")
# matched groups 
Fold = joblib.load(root_dir + 'abcc_2split.joblib')
# get local indices for train and test
train_index = Fold[fold_num][0]
test_index = Fold[fold_num][1]

targ_list= ['nihtbx_totalcomp_uncorrected', 'nihtbx_cryst_uncorrected', 'nihtbx_fluidcomp_uncorrected']
for targ in targ_list:
    target_name =  targ.split('_')[1][0:5] + '_'
    # pls output dict for each target
    pls_dict = {}
    pls_dict_wa = {}
    pls_dict_aa = {}
    pls_dict_ha = {}
    pls_dict_aawa = {}
    pls_dict_hawa = {}
    # Load fold data
    for key, featu in featuresd.items():
        feat = featu.dropna()
        common_ind = list(set(feat.index).intersection(set(demo.index), set(targs.index)))
        feature = feat.loc[common_ind]
        demo_table = demo.loc[common_ind]
        # Extract train and test indices
        train_idx = list(set(common_ind).intersection(set(train_index)))
        test_idx = list(set(common_ind).intersection(set(test_index)))
        print(f"Fold {fold_num} - Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        eth1 = demo_table[demo_table['race_ethnicity'] == 1].index
        eth2 = demo_table[demo_table['race_ethnicity'] == 2].index
        # hispanic added
        eth3 = demo_table[demo_table['race_ethnicity'] == 3].index
        print(f"Fold {fold_num} - eth1 size: {len(eth1)}, eth2 size: {len(eth2)}, eth3 size: {len(eth3)}")
        #print(eth1)
        train_WAidx = list(set(eth1).intersection(set(train_idx)))
        train_AAidx = list(set(eth2).intersection(set(train_idx)))
        train_HAidx = list(set(eth3).intersection(set(train_idx)))
        print(len(train_WAidx), len(train_AAidx), len(train_HAidx))
        # Balance eth1 to match eth2
        if len(train_WAidx) > len(train_AAidx):
            train_WAidx1 = np.random.choice(train_WAidx, size=len(train_AAidx), replace=False).tolist()
        elif len(train_AAidx) > len(train_WAidx):
            train_AAidx = np.random.choice(train_AAidx, size=len(train_WAidx), replace=False).tolist()
        train_AAWAidx = train_AAidx + train_WAidx1           
        test_WAidx = list(set(eth1).intersection(set(test_idx)))
        test_AAidx = list(set(eth2).intersection(set(test_idx)))    
        test_HAidx = list(set(eth3).intersection(set(test_idx)))    
        # Balance eth1 to match eth3
        print(len(train_WAidx), len(train_AAidx), len(train_HAidx))
        if len(train_WAidx) > len(train_HAidx):
            train_WAidx2 = np.random.choice(train_WAidx, size=len(train_HAidx), replace=False).tolist()
        elif len(train_HAidx) > len(train_WAidx):
            train_HAidx = np.random.choice(train_HAidx, size=len(train_WAidx), replace=False).tolist()
        train_HAWAidx = train_HAidx + train_WAidx2           
        print(len(train_WAidx), len(train_AAidx), len(train_HAidx))

        
        #print(train_idx)
        # Split features and target
        x_tr = feature.loc[train_idx]
        x_te = feature.loc[test_idx]
        y_tr = pd.DataFrame(targs.loc[train_idx][targ].values, index=x_tr.index, columns=[target_name])
        y_te = pd.DataFrame(targs.loc[test_idx][targ].values, index=x_te.index, columns=[target_name])
        if not x_te.index.equals(y_te.index):
            print(targ, key, '4')



        print(f"Fold {fold_num} - Train wa size: {len(train_WAidx)}, Train aa size: {len(train_AAidx)}, Train ha size: {len(train_HAidx)}")
        print(f"Fold {fold_num} - test wa size: {len(test_WAidx)}, test aa size: {len(test_AAidx)},  test ha size: {len(test_HAidx)}")
        if not set(test_WAidx).issubset(y_te.index):
            print('error')
        # Run PLS All
        run_pls(pls_dir, targ, pls_dict, x_tr, x_te, y_tr, y_te)
        
        x_tr_wa = feature.loc[train_WAidx1]
        y_tr_wa = pd.DataFrame(targs.loc[train_WAidx1][targ].values, index=x_tr_wa.index, columns=[target_name])    
        x_tr_aa = feature.loc[train_AAidx]
        y_tr_aa = pd.DataFrame(targs.loc[train_AAidx][targ].values, index=x_tr_aa.index, columns=[target_name])
        x_tr_aawa = feature.loc[train_AAWAidx]
        y_tr_aawa = pd.DataFrame(targs.loc[train_AAWAidx][targ].values, index=x_tr_aawa.index, columns=[target_name])         
        # Run PLS WA
        run_pls(pls_dir, targ, pls_dict_wa, x_tr_wa, x_te, y_tr_wa, y_te)        
        # Run PLS AA
        run_pls(pls_dir, targ, pls_dict_aa, x_tr_aa, x_te, y_tr_aa, y_te)   
        # Run PLS WAAA
        run_pls(pls_dir, targ, pls_dict_aawa, x_tr_aawa, x_te, y_tr_aawa, y_te)   
        
        x_tr_wa = feature.loc[train_WAidx2]
        y_tr_wa = pd.DataFrame(targs.loc[train_WAidx2][targ].values, index=x_tr_wa.index, columns=[target_name])    
        x_tr_ha = feature.loc[train_HAidx]
        y_tr_ha = pd.DataFrame(targs.loc[train_HAidx][targ].values, index=x_tr_ha.index, columns=[target_name])
        x_tr_hawa = feature.loc[train_HAWAidx]
        y_tr_hawa = pd.DataFrame(targs.loc[train_HAWAidx][targ].values, index=x_tr_hawa.index, columns=[target_name])              
        # Run PLS HA
        run_pls(pls_dir, targ, pls_dict_ha, x_tr_ha, x_te, y_tr_ha, y_te)   
        # Run PLS WAHA
        run_pls(pls_dir, targ, pls_dict_hawa, x_tr_hawa, x_te, y_tr_hawa, y_te)  

        # save dict
        joblib.dump(pls_dict, pls_dir + target_name + features_name + '_pls_output_std_All.joblib', compress=0)
        joblib.dump(pls_dict_wa, pls_dir + target_name + features_name + '_pls_output_std_wa.joblib', compress=0)
        joblib.dump(pls_dict_aa, pls_dir + target_name + features_name + '_pls_output_std_aa.joblib', compress=0)
        joblib.dump(pls_dict_aawa, pls_dir + target_name + features_name + '_pls_output_std_waaa.joblib', compress=0) 
        joblib.dump(pls_dict_ha, pls_dir + target_name + features_name + '_pls_output_std_ha.joblib', compress=0) 
        joblib.dump(pls_dict_hawa, pls_dir + target_name + features_name + '_pls_output_std_haaa.joblib', compress=0)  
        gc.collect()    



