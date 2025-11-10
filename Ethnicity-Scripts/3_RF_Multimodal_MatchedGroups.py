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
from sklearn.ensemble import RandomForestRegressor
import shap

# %%
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 120
# fold_num = 0
# Paths
abcd_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/' #/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/ML_Tables/Ethnicity/' 
tables_path = root_dir
fold_base_path = root_dir + 'folds/'
fold_dir = fold_base_path  + 'Fold_' + str(fold_num) +'/'
rf2_dir = fold_dir + 'rf2/'
pls_dir = fold_dir + 'pls/'

if not os.path.isdir(rf2_dir):
    os.mkdir(rf2_dir) 

# %%
def run_RF2(path_out, target_name, rf2_dict, x_tr, x_te, y_tr, y_te):
    try:

        perf = {'nmse':[], 'r2':[], 'pr':[], 'mae':[]}
        rf_hyper_param = {
            'n_estimators': [1000],  # Number of trees in the forest
            'max_depth': list(range(1,11)),  # Maximum depth of the tree
            'max_features': [None, 'sqrt', 'log2'] 
        }
        reg = RandomForestRegressor()        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        search = GridSearchCV(reg, rf_hyper_param, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        search.fit(x_tr.values, y_tr.values)
        print(set_name, ' grd search done')
        best_model = search.best_estimator_
        if best_model is not None:
        #     joblib.dump(search.cv_results_, path_out + 'enet-l1_cv_results_'+ target_name + str(key)+'.pkl')
        #     joblib.dump(best_model, path_out + 'enet-l1_best_model_'+ target_name +str(key)+'.pkl')
            print('set:' , set_name , ' score:', search.best_score_)
            cv_results = search.cv_results_
            df_heatmap = pd.DataFrame(cv_results)


            # predict train y
            tr_y_p = best_model.predict(x_tr.values)
            tr_y_rf2 = pd.DataFrame(tr_y_p, index=y_tr.index)

            # add train performance 
            perf['nmse'].append(mean_squared_error(y_tr, tr_y_p))
            perf['r2'].append(r2_score(y_tr, tr_y_p))
            pearson_corr, _ = pearsonr(y_tr.values.flatten(), tr_y_p.flatten())
            perf['pr'].append(pearson_corr)
            perf['mae'].append(mean_absolute_error(y_tr, tr_y_p))

            # predict test y
            te_y_p = best_model.predict(x_te.values)
            te_y_rf2 = pd.DataFrame(te_y_p, index=y_te.index)
            print(set_name, ' fit done')
            # add test performance 
            perf['nmse'].append(mean_squared_error(y_te, te_y_p))
            perf['r2'].append(r2_score(y_te, te_y_p))
            p_corr, _ = pearsonr(y_te.values.flatten(), te_y_p.flatten())
            perf['pr'].append(p_corr)
            perf['mae'].append(mean_absolute_error(y_te, te_y_p))

            # Calculate SHAP values
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(x_tr.values)
            print(set_name, ' shap done')
            #save to disk
            performance = pd.DataFrame(perf)
            # fill output dict
            rf2_dict[set_name]['data'].update({'xtrain': x_tr, 'xtest': x_tr,  'yptrain': tr_y_rf2, 'yptest': te_y_rf2, 'yttrain': y_tr, 'yttest': y_te}) 
            rf2_dict[set_name]['model'].update({'perf': performance,  'shap': shap_values}) 
            rf2_model[set_name]['model'].update({'best_model': best_model,'cv_results': search.cv_results_})
        else:
            print('best_model is none')
            # plt.close('all')
        return best_model
    except Exception as e:
        logging.error(traceback.format_exc())


# %%
### contrasts list:
contrasts_mid= [    
    'Reward-Neutral',
    'Loss-Neutral',
    'LgReward-Neutral',
    'SmallReward-Neutral',
    'LgLoss-Neutral',
    'SmallLoss-Neutral',
    'LgLoss-SmallLoss',
    'LgReward-SmallReward',
    'RewardHit-RewardMiss',
    'LossHit-LossMiss']
contrasts_sst= [    
    'CorrectGo',
    'IncorrectGo',
    'CorrectStop',
    'IncorrectStop',
    'CorrectStop-CorrectGo',
    'IncorrectStop-CorrectGo',
    'Stop-CorrectGo',
    'CorrectStop-IncorrectStop',
    'IncorrectGo-CorrectGo',
    'IncorrectGo-IncorrectStop']
contrasts_wm= [    
    'place',
    'face',
    'emotionface',
    'face-place',
    'PosFace-NeutFace',
    'NegFace-NeutFace',
    'emotionface-NeutFace',
    'twobk',
    'zerobk',
    'twobk-zerobk']
contrasts3_wm= [    
    'twobk',
    'zerobk',
    'twobk-zerobk']

abcd_contrasts = ['antiLargeRewVsSmallRew_ROI_mid', 'feedPunPosVsNeg_ROI_mid', 'incorrectgovsincorrectstop_ROI_sst', 'anystopvscorrectgo_ROI_sst', 'incorrectstopvscorrectgo_ROI_sst', 'emotionvsneutface_ROI_nbk', 'incorrectgovscorrectgo_ROI_sst', 'correctgovsfixation_ROI_sst', 'antiRewVsNeu_ROI_mid', 'X2back_ROI_nbk', 'facevsplace_ROI_nbk', 'antiSmallLossVsNeu_ROI_mid', 'negfacevsneutface_ROI_nbk', 'antiLargeLossVsNeu_ROI_mid', 'emotion_ROI_nbk', 'correctstopvsincorrectstop_ROI_sst', 'antiSmallRewVsNeu_ROI_mid', 'antiLargeRewVsNeu_ROI_mid', 'antiLosVsNeu_ROI_mid', 'correctstopvscorrectgo_ROI_sst', 'posfacevsneutface_ROI_nbk', 'X0back_ROI_nbk', 'place_ROI_nbk', 'antiLargeLossVsSmallLoss_ROI_mid', 'X2backvs0back_ROI_nbk', 'feedRewPosVsNeg_ROI_mid']
sst_list = [
    'incorrectgovsincorrectstop_ROI_sst',
    'anystopvscorrectgo_ROI_sst',
    'incorrectstopvscorrectgo_ROI_sst',
    'incorrectgovscorrectgo_ROI_sst',
    'correctgovsfixation_ROI_sst',
    'correctstopvsincorrectstop_ROI_sst',
    'correctstopvscorrectgo_ROI_sst'
]
mid_list = [
    'antiLargeRewVsSmallRew_ROI_mid',
    'feedPunPosVsNeg_ROI_mid',
    'antiRewVsNeu_ROI_mid',
    'antiSmallLossVsNeu_ROI_mid',
    'antiLargeLossVsNeu_ROI_mid',
    'antiSmallRewVsNeu_ROI_mid',
    'antiLargeRewVsNeu_ROI_mid',
    'antiLosVsNeu_ROI_mid',
    'antiLargeLossVsSmallLoss_ROI_mid',
    'feedRewPosVsNeg_ROI_mid'
]
nbk_list = [
    'emotionvsneutface_ROI_nbk',
    'X2back_ROI_nbk',
    'facevsplace_ROI_nbk',
    'negfacevsneutface_ROI_nbk',
    'emotion_ROI_nbk',
    'posfacevsneutface_ROI_nbk',
    'X0back_ROI_nbk',
    'place_ROI_nbk',
    'X2backvs0back_ROI_nbk'
]

abcd_rsmri = ['Avg_T1_ASEG_Vol_', 'Dest_Area_', 'Dest_Thick_', 'Avg_T2_ASEG_', 'rsmri_within_avg_data', 'Dest_Vol_', 'rsmri_gordon_aseg_data']
# %%
moda_stack_dict = dict(

                        nAll1_abcd = ['DTI', 'cort', 'surf', 'subc', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ','conn_rest', 
                                'conn_wm', 'conn_sst', 'conn_mid', 'subnet_rest', 'gfc', 'tfc'] 
                                 + contrasts_mid + contrasts_sst + contrasts_wm  + abcd_contrasts + abcd_rsmri, 
                                 
                        #nGD_WMCntr = contrasts_wm + nbk_list,
                        #nGD_MidCntr = contrasts_mid + mid_list,
                        #GD_SstCntr = contrasts_sst + sst_list,
                        #nGD_TaskCntr = contrasts_wm + nbk_list + contrasts_mid + mid_list + contrasts_sst + sst_list,
                        #AllFCs =  ['conn_rest', 'conn_wm', 'conn_sst', 'conn_mid', 'gfc', 'tfc'],
                        #AllRest = ['rsmri_within_avg_data', 'rsmri_gordon_aseg_data', 'conn_rest', 'subnet_rest'],
                        #AllSmri = ['Avg_T1_ASEG_Vol_', 'Dest_Area_', 'Dest_Thick_', 'Avg_T2_ASEG_', 'Dest_Vol_', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ'],
                        #NonTask = abcd_rsmri + ['DTI', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ','conn_rest', 'subnet_rest'],
                        #nGD_TaskAll = contrasts_wm + nbk_list + contrasts_mid + mid_list + contrasts_sst + sst_list + ['conn_wm', 'conn_sst', 'conn_mid'],
                        #nGD_WMAll = contrasts_wm + nbk_list + ['conn_wm'],
                        #nGD_MidAll = contrasts_mid + mid_list+ ['conn_mid'],
                        #nGD_SstAll = contrasts_sst + sst_list+ ['conn_sst'],
                        #D_TaskCntr = nbk_list + mid_list + sst_list,
                        #D_TaskAll = nbk_list + mid_list + sst_list + ['conn_wm', 'conn_sst', 'conn_mid'],
                        #nG_WmAll = contrasts_wm + ['conn_wm'],
                        #nG_MidAll = contrasts_mid + ['conn_mid'],
                        #nG_TaskCntr = contrasts_wm + contrasts_mid + contrasts_sst,
                        #nD_TaskCntr = nbk_list + mid_list + sst_list,
                        
                        
                        )
# moda_stack_dict = dict(
#                         Str = ['cort', 'subc', 'surf', 'rest', 'VolBrain'],
#                         WmCntr = contrasts_wm
#                         )
#, 'part_rest','part_wm','part_mid', 'part_sst', 'tang_rest','tang_wm','tang_mid', 'tang_sst'

# %%

print('started to calculate the Fold #', fold_num, '\n')
demo = pd.read_csv(abcd_dir + 'demo_nesi.csv', index_col=0).dropna()
targ_list = ['total_'] #,'cryst_','fluid_'
for targ in targ_list:
    target = joblib.load(abcd_dir + targ + 'std_targets.joblib')

    abccsmri = joblib.load(pls_dir + targ + 'abccSmri_pls_output_std_All.joblib')
    abcccntr = joblib.load(pls_dir + targ + 'abccCntr_pls_output_std_All.joblib')
    abccconn = joblib.load(pls_dir + targ + 'abccConn_pls_output_std_All.joblib')
    abccgtfc = joblib.load(pls_dir + targ + 'abccGtfc_pls_output_std_All.joblib')
    abcdcntr = joblib.load(pls_dir + targ + 'abccCntr_pls_output_std_All.joblib')
    abcdrsmri = joblib.load(pls_dir + targ + 'abccRsmri_pls_output_std_All.joblib')
    
    All_enet1 = {**abccsmri, **abcccntr, **abccconn, **abccgtfc, **abcdcntr, **abcdrsmri} # 
    # enet output dict for each target
    rf2_dict = {}
    rf2_model = {}
    rf2_dict_wa = {}
    rf2_model_wa = {}
    rf2_dict_aa = {}
    rf2_model_aa = {}
    rf2_dict_waaa = {}
    rf2_model_waaa = {}
    for set_name, names in moda_stack_dict.items():
        set_dict = {key: All_enet1[key] for key in All_enet1.keys() if any(substring in key for substring in names)}
        
        # Extract the indices from the DataFrames
        train_ind = set()
        test_ind =  set()
        # for moda in set_dict.values():
        #     train_ind.update(moda['yptrain'].index)
        #     test_ind.update(moda['yptest'].index)
        # Perform a full outer join on the indices
        set_df_tr = None
        set_df_te = None
        for moda_n , moda in set_dict.items():
            moda['data']['yptrain'].columns = [f"{moda_n}" for col in moda['data']['yptrain'].columns]
            moda['data']['yptest'].columns = [f"{moda_n}" for col in moda['data']['yptrain'].columns]
            if set_df_tr is None and set_df_te is None:
                set_df_tr = moda['data']['yptrain']
                set_df_te = moda['data']['yptest']
            else:
                set_df_tr = set_df_tr.join(moda['data']['yptrain'], how='outer', sort=True)
                set_df_te = set_df_te.join(moda['data']['yptest'], how='outer', sort=True)
        # # If there are duplicate columns due to the join, rename them
        # set_df_tr.columns = pd.io.parsers.ParserBase({'names':set_df_tr.columns})._maybe_dedup_names(set_df_tr.columns)
        # set_df_te.columns = pd.io.parsers.ParserBase({'names':set_df_te.columns})._maybe_dedup_names(set_df_te.columns)
        # set_df_tr.to_csv(enet2_dir + targ + set_name + '_df.csv')
        # Reorder columns
        if set_df_tr is not None and set_df_te is not None:
            
            set_df_tr = set_df_tr[sorted(set_df_tr.columns)]
            set_df_te = set_df_te[sorted(set_df_te.columns)]
            
            #replace nan with extremes
            #tr_l = set_df_tr.fillna(1000)
            #tr_s = set_df_tr.fillna(-1000)
            #set_df_tr = tr_l.join(tr_s, lsuffix='_l', rsuffix='_s')
            #te_l = set_df_te.fillna(1000)
            #te_s = set_df_te.fillna(-1000)
            #set_df_te = te_l.join(te_s, lsuffix='_l', rsuffix='_s')

            # select shared indices
            common_ind = list(set(demo.index).intersection(set(targs.index)))
            # feature = feat.loc[common_ind]
            demo_table = demo.loc[common_ind]

            train_idx = set_df_tr.index
            test_idx = set_df_te.index


            # Select ethnicity
            eth1 = demo_table[demo_table['race_ethnicity'] == 1].index
            eth2 = demo_table[demo_table['race_ethnicity'] == 2].index
            # Train and test split
            train_WAidx = list(set(eth1).intersection(set(train_idx)))
            train_AAidx = list(set(eth2).intersection(set(train_idx)))
            test_WAidx = list(set(eth1).intersection(set(test_idx)))
            test_AAidx = list(set(eth2).intersection(set(test_idx)))
            print('train aa size',  len(test_AAidx) , 'train wa size', len(test_WAidx))
            # RandWAonly selection
            if len(train_WAidx) > len(train_AAidx):
                train_WAidx1 = np.random.choice(train_WAidx, size=len(train_AAidx), replace=False).tolist()
            elif len(train_AAidx) > len(train_WAidx):
                train_AAidx = np.random.choice(train_AAidx, size=len(train_WAidx), replace=False).tolist()
            # RandWA + AA:
            train_AAWAidx = train_AAidx + train_WAidx1           
            test_WAidx = list(set(eth1).intersection(set(test_idx)))
            test_AAidx = list(set(eth2).intersection(set(test_idx)))    
            # Split features and target
            x_tr = set_df_tr.loc[train_idx]
            x_te = set_df_te.loc[test_idx]
            y_tr = pd.DataFrame(target.loc[train_idx][targ].values, index=x_tr.index, columns=[targ])
            y_te = pd.DataFrame(target.loc[test_idx][targ].values, index=x_te.index, columns=[targ])

            if not set(test_WAidx).issubset(y_te.index):
                print('error')
            # Run rf All
            run_RF2(rf2_dir, targ, rf2_dict, x_tr, x_te, y_tr, y_te)
            
            x_tr_wa = set_df_tr.loc[train_WAidx1]
            y_tr_wa = pd.DataFrame(target.loc[train_WAidx1][targ].values, index=x_tr_wa.index, columns=[targ])    
            x_tr_aa = set_df_tr.loc[train_AAidx]
            y_tr_aa = pd.DataFrame(target.loc[train_AAidx][targ].values, index=x_tr_aa.index, columns=[targ])
            x_tr_aawa = set_df_tr.loc[train_AAWAidx]
            y_tr_aawa = pd.DataFrame(target.loc[train_AAWAidx][targ].values, index=x_tr_aawa.index, columns=[targ])         
            # Run PLS WA
            run_RF2(rf2_dir, targ, rf2_dict_wa, x_tr_wa, x_te, y_tr_wa, y_te)        
            # Run PLS AA
            run_RF2(rf2_dir, targ, rf2_dict_aa, x_tr_aa, x_te, y_tr_aa, y_te)   
            # Run PLS WAAA
            run_RF2(rf2_dir, targ, rf2_dict_waaa, x_tr_aawa, x_te, y_tr_aawa, y_te)   

    # save dict
    joblib.dump(rf2_dict, rf2_dir + targ + 'rf2-1_output_std_All.joblib', compress=0)
    joblib.dump(rf2_model, rf2_dir + targ + 'rf2-1_model_std_All.joblib', compress=0) 
    joblib.dump(rf2_dict_wa, rf2_dir + targ + 'rf2-1_output_std_wa.joblib', compress=0)
    joblib.dump(rf2_model_wa, rf2_dir + targ + 'rf2-1_model_std_wa.joblib', compress=0) 
    joblib.dump(rf2_dict_aa, rf2_dir + targ + 'rf2-1_output_std_aa.joblib', compress=0)
    joblib.dump(rf2_model_aa, rf2_dir + targ + 'rf2-1_model_std_aa.joblib', compress=0) 
    joblib.dump(rf2_dict_waaa, rf2_dir + targ + 'rf2-1_output_std_waaa.joblib', compress=0)
    joblib.dump(rf2_model_waaa, rf2_dir + targ + 'rf2-1_model_std_waaa.joblib', compress=0) 
    # Save rf2_dict using pickle
    # with open(rf2_dir + targ + 'rf2-1_output_std.pkl', 'wb') as f:
    #     pickle.dump(rf2_dict, f)
    # with open(rf2_dir + targ + 'rf2-1_model_std.pkl', 'wb') as f:
    #     pickle.dump(rf2_model, f)
   



