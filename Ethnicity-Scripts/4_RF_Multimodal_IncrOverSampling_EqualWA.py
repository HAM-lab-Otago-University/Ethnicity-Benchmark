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
    fold_num = int(sys.argv[1]) % 4
else:
    fold_num = 0
print(fold_num)
# Paths
abcd_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/' #/media/hcs-sci-psy-narun/ABCC/fmriresults01/derivatives/ML_Tables/Ethnicity/' 
tables_path = root_dir
fold_base_path = os.path.join(root_dir, 'folds/tsp/')
fold_dir = os.path.join(fold_base_path, f'Fold_{fold_num}/')
rf2_dir = fold_dir + 'rf2/'
pls_dir = fold_dir + 'pls/'

if not os.path.isdir(rf2_dir):
    os.mkdir(rf2_dir) 

demo = pd.read_csv(abcd_dir + 'demo_nesi.csv', index_col=0).dropna()
targs = pd.read_csv(abcd_dir + 'cog_all.csv', index_col=0, low_memory=False).dropna()
targs.index = targs.index.str.replace('_', '')
# %%
def run_RF2(rf2_dict, x_tr, x_te, y_tr, y_te,
    wa_best_mae, best_mae, best_round, wa_best_round, sim_round,
    test_AAidx, test_WAidx, key, tolerance, train_aa_len, train_wa_len
):
    try:
        scalerx = StandardScaler()
        x_tr_scaled = scalerx.fit_transform(x_tr.values)
        x_te_scaled = scalerx.transform(x_te.values)
        scalery = StandardScaler()
        y_tr_scaled = scalery.fit_transform(y_tr.values)
        y_te_scaled = scalery.transform(y_te.values)
        y_tr_scaleddf = pd.DataFrame(y_tr_scaled, index=y_tr.index)
        y_te_scaleddf = pd.DataFrame(y_te_scaled, index=y_te.index)

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
        search.fit(x_tr_scaled, y_tr_scaled.ravel())
        print(set_name, ' grd search done')
        best_model = search.best_estimator_
        if best_model is not None:
        #     joblib.dump(search.cv_results_, path_out + 'enet-l1_cv_results_'+ target_name + str(key)+'.pkl')
        #     joblib.dump(best_model, path_out + 'enet-l1_best_model_'+ target_name +str(key)+'.pkl')
            print('set:' , set_name , ' score:', search.best_score_)
            # cv_results = search.cv_results_
            # df_heatmap = pd.DataFrame(cv_results)
            # Extract relevant information
            # params = cv_results['params']
            # mean_test_scores = cv_results['mean_test_score'].reshape(len(rf_hyper_param['max_depth']), len(rf_hyper_param['max_features']))
            # max_d_values = np.unique([param['max_depth'] for param in params])
            # max_f_values = np.unique([param['max_features'] for param in params])

            # # Create a DataFrame for the heatmap
            # df_heatmap = pd.DataFrame(mean_test_scores, index=max_d_values, columns=max_f_values)
        
            #print('alpha:', len(alpha_values))
            #print('l1_ratio' , len(l1_ratio_values))
            #print('mtest_score' , len(mean_test_scores))
            # df_heatmap.to_csv(path_out + 'enet-l1_' + target_name + str(key)+'.csv')
            # Plot the heatmap
            # plt.figure(figsize=(12, 8))
            # sns.heatmap(df_heatmap, annot=False, cmap='viridis', cbar_kws={'label': 'Mean Test Score'})
            # plt.title('Hyperparameter Heatmap')
            # plt.xlabel('Alpha')
            # plt.ylabel('L1 Ratio')
            # plt.savefig(path_out + '/enet-l1_' + target_name + str(key) + '.png')
            # plt.close('all')

        tr_y_p = best_model.predict(x_tr_scaled)
        te_y_p = best_model.predict(x_te_scaled)

        perf['nmse'].append(mean_squared_error(y_tr_scaled, tr_y_p))
        perf['r2'].append(r2_score(y_tr_scaled, tr_y_p))
        perf['pr'].append(pearsonr(y_tr_scaled.flatten(), tr_y_p.flatten())[0])
        perf['mae'].append(mean_absolute_error(y_tr_scaled, tr_y_p))

        perf['nmse'].append(mean_squared_error(y_te_scaled, te_y_p))
        perf['r2'].append(r2_score(y_te_scaled, te_y_p))
        perf['pr'].append(pearsonr(y_te_scaled.flatten(), te_y_p.flatten())[0])
        perf['mae'].append(mean_absolute_error(y_te_scaled, te_y_p))

        y_te_scaled_aa = y_te_scaleddf.loc[test_AAidx].values
        te_y_p_aa = pd.DataFrame(te_y_p, index=y_te.index).loc[test_AAidx].values
        perf['nmse'].append(mean_squared_error(y_te_scaled_aa, te_y_p_aa))
        perf['r2'].append(r2_score(y_te_scaled_aa, te_y_p_aa))
        perf['pr'].append(pearsonr(y_te_scaled_aa.flatten(), te_y_p_aa.flatten())[0])
        perf['mae'].append(mean_absolute_error(y_te_scaled_aa, te_y_p_aa))

        y_te_scaled_wa = y_te_scaleddf.loc[test_WAidx].values
        te_y_p_wa = pd.DataFrame(te_y_p, index=y_te.index).loc[test_WAidx].values
        perf['nmse'].append(mean_squared_error(y_te_scaled_wa, te_y_p_wa))
        perf['r2'].append(r2_score(y_te_scaled_wa, te_y_p_wa))
        perf['pr'].append(pearsonr(y_te_scaled_wa.flatten(), te_y_p_wa.flatten())[0])
        perf['mae'].append(mean_absolute_error(y_te_scaled_wa, te_y_p_wa))

        performance = pd.DataFrame(perf, index=['train', 'test', 'test_aa', 'test_wa'])
        test_aa_mae = performance.loc['test_aa', 'mae']
        test_wa_mae = performance.loc['test_wa', 'mae']
        
        print('test_aa_mae', test_aa_mae,'test_wa_mae', test_wa_mae)
	
        if test_wa_mae < wa_best_mae - tolerance:
            wa_best_mae = test_wa_mae
            wa_best_round = sim_round

        # recent_maes.append(test_aa_mae)
        if test_aa_mae < best_mae - tolerance:
            best_mae = test_aa_mae
            best_round = sim_round
        # if len(recent_maes) > window_size:
        #     recent_maes = recent_maes[-window_size:]
        # worse_in_window = all(mae > best_mae + tolerance for mae in recent_maes)
        # is_better_than_wa = test_aa_mae <= test_wa_mae
        # sim_limit = sim_round > max_sim

        # if is_better_than_wa or sim_limit or (sim_round - best_round > patience and worse_in_window):
        #     sim_key = False
        # print('after check:', sim_key)
        # if not sim_key:
        #     print('best aa mae:', best_mae, 'best aa round:', best_round ,'current aa mae:', test_aa_mae)
        sim_key_str = str(sim_round)
        # if sim_key_str not in pls_dict[key]:
        #     pls_dict[key][sim_key_str] = {'model': {}}

        #pls_dict[key][sim_key_str]['model'] = {'perf': performance}
        rf2_dict[key][sim_key_str] = {
                'model': {'perf': performance},
                'data': {
                    # 'Xtrain': pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index),
                    # 'Xtest1': pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index),
                    'yptrain': pd.DataFrame(best_model.predict(x_tr_scaled), index=y_tr.index),
                    'yptest': pd.DataFrame(te_y_p, index=y_te.index),
                    'yttrain': y_tr_scaleddf,
                    'yttest': y_te_scaleddf
                }
            }

        if sim_round == best_round:
            rf2_dict[key]['best_aa_round'] = {
                'model': {'perf': performance, 'round_num': best_round, 'best_model': best_model},
                'data': {
                    # 'Xtrain': pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index),
                    # 'Xtest1': pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index),
                    'yptrain': pd.DataFrame(best_model.predict(x_tr_scaled), index=y_tr.index),
                    'yptest': pd.DataFrame(te_y_p, index=y_te.index),
                    'yttrain': y_tr_scaleddf,
                    'yttest': y_te_scaleddf
                }
            }

        if sim_round == wa_best_round:
            rf2_dict[key]['best_wa_round'] = {
                'model': {'perf': performance, 'round_num': wa_best_round, 'best_model': best_model},
                'data': {
                    # 'Xtrain': pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index),
                    # 'Xtest1': pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index),
                    'yptrain': pd.DataFrame(best_model.predict(x_tr_scaled), index=y_tr.index),
                    'yptest': pd.DataFrame(te_y_p, index=y_te.index),
                    'yttrain': y_tr_scaleddf,
                    'yttest': y_te_scaleddf
                },
                'train_lens': {'len_aa': train_aa_len, 'len_wa': train_wa_len}
            }

        gc.collect()
    except Exception as e:
        logging.error(traceback.format_exc())

    return wa_best_mae, best_mae, best_round, wa_best_round


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

                        # nAll1_abcd = ['DTI', 'cort', 'surf', 'subc', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ','conn_rest', 
                        #         'conn_wm', 'conn_sst', 'conn_mid', 'subnet_rest', 'gfc', 'tfc'] 
                        #          + contrasts_mid + contrasts_sst + contrasts_wm  + abcd_contrasts + abcd_rsmri, 
                                 
                        #nGD_WMCntr = contrasts_wm + nbk_list,
                        #nGD_MidCntr = contrasts_mid + mid_list,
                        #GD_SstCntr = contrasts_sst + sst_list,
                        nGD_TaskCntr = contrasts_wm + nbk_list + contrasts_mid + mid_list + contrasts_sst + sst_list,
                        # AllFCs =  ['conn_rest', 'conn_wm', 'conn_sst', 'conn_mid', 'gfc', 'tfc'],
                        # AllRest = ['rsmri_within_avg_data', 'rsmri_gordon_aseg_data', 'conn_rest', 'subnet_rest'],
                        # TaskConn = ['conn_wm', 'conn_sst', 'conn_mid'],
                        # AllSmri = ['Avg_T1_ASEG_Vol_', 'Dest_Area_', 'Dest_Thick_', 'Avg_T2_ASEG_', 'Dest_Vol_', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ'],
                        # NonTask = abcd_rsmri + ['DTI', 'VolBrain', 'T1_white', 'T1_gray', 'T2_white', 'T2_gray', 'Sulcal_Depth', 'T1_norm', 'T2_norm', 'T1_summ','T2_summ','conn_rest', 'subnet_rest'],
                        # nGD_TaskAll = contrasts_wm + nbk_list + contrasts_mid + mid_list + contrasts_sst + sst_list + ['conn_wm', 'conn_sst', 'conn_mid'],
                        # nGD_WMAll = contrasts_wm + nbk_list + ['conn_wm'],
                        # nGD_MidAll = contrasts_mid + mid_list+ ['conn_mid'],
                        # nGD_SstAll = contrasts_sst + sst_list+ ['conn_sst'],
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

print(f"\nStarting Fold {fold_num}...")
Fold = joblib.load(root_dir + 'abcc_2split.joblib')
train_index = Fold[fold_num][0]
test_index = Fold[fold_num][1]
targ_list = ['nihtbx_totalcomp_uncorrected']
for targ in targ_list:
    target_name = targ.split('_')[1][0:5] + '_'
    #target = joblib.load(main_fold + targ + 'std_targets.joblib')
    print('loading dics')
    # abccsmri = joblib.load(pls_dir + target_name + 'abccSmri_pls_output_std_Ewaaa_sim.joblib')
    print('smri loaded')
    abcccntr = joblib.load(pls_dir + target_name + 'abccCntr_pls_output_std_Ewaaa_sim.joblib')
    
    # abcctfc = joblib.load(pls_dir + target_name + 'tfc_pls_output_std_Ewaaa_sim.joblib')
    # abccgfc = joblib.load(pls_dir + target_name + 'gfc_pls_output_std_Ewaaa_sim.joblib')
    # abccconnRoth = joblib.load(pls_dir + target_name + 'avg_rest_pls_output_std_Ewaaa_sim.joblib')
    # abccconn = joblib.load(pls_dir + target_name + 'subnet_rest_pls_output_std_Ewaaa_sim.joblib')

    # abccconnmid = joblib.load(pls_dir + target_name + 'conn_mid_pls_output_std_Ewaaa_sim.joblib')
    # abccconnsst = joblib.load(pls_dir + target_name + 'conn_sst_pls_output_std_Ewaaa_sim.joblib')
    # abccconnwm = joblib.load(pls_dir + target_name + 'conn_wm_pls_output_std_Ewaaa_sim.joblib')
    # abccconnrest = joblib.load(pls_dir + target_name + 'conn_rest_pls_output_std_Ewaaa_sim.joblib')

    abcdcntr = joblib.load(pls_dir + target_name + 'abcdCntr_pls_output_std_Ewaaa_sim.joblib')
    # abcdrsmri = joblib.load(pls_dir + target_name + 'abcdRsmri_pls_output_std_Ewaaa_sim.joblib')
    print('rsmri loaded')
    All_enet1 = {**abcccntr,  **abcdcntr} # **abccconnRoth, **abccconnrest,
    # enet output dict for each target
    print('smri combined')
    # rf2_model = {}
    for set_name, names in moda_stack_dict.items():
        print(set_name)
        rf2_dict = {}
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
            equal_sim = str(moda['best_wa_round']['train_lens']['len_aa'])
            moda[equal_sim]['data']['yptrain'].columns = [f"{moda_n}" for col in moda[equal_sim]['data']['yptrain'].columns]
            moda[equal_sim]['data']['yptest'].columns = [f"{moda_n}" for col in moda[equal_sim]['data']['yptrain'].columns]
            if set_df_tr is None and set_df_te is None:
                set_df_tr = moda[equal_sim]['data']['yptrain']
                set_df_te = moda[equal_sim]['data']['yptest']
            else:
                set_df_tr = set_df_tr.join(moda[equal_sim]['data']['yptrain'], how='outer', sort=True)
                set_df_te = set_df_te.join(moda[equal_sim]['data']['yptest'], how='outer', sort=True)
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

            common_ind = list(set(demo.index).intersection(set(targs.index)))
            # feature = feat.loc[common_ind]
            demo_table = demo.loc[common_ind]

            train_idx = set_df_tr.index
            test_idx = set_df_te.index

            eth1 = demo_table[demo_table['race_ethnicity'] == 1].index
            eth2 = demo_table[demo_table['race_ethnicity'] == 2].index

            train_WAidx = list(set(eth1).intersection(set(train_idx)))
            train_AAidx = list(set(eth2).intersection(set(train_idx)))
            test_WAidx = list(set(eth1).intersection(set(test_idx)))
            test_AAidx = list(set(eth2).intersection(set(test_idx)))
            print('train aa size',  len(test_AAidx) , 'train wa size', len(test_WAidx))

            sim_key = True
            sim_round = 0

            if set_name not in rf2_dict:
                rf2_dict[set_name] = {}

            # recent_maes = []
            best_mae = np.inf
            wa_best_mae = np.inf
            best_round = 0
            wa_best_round = 0
            # patience = 100
            tolerance = 0.001
            max_sim = 3*len(train_AAidx)
            seed = 42 + sim_round
            np.random.seed(seed)
            train_WAidx_sampled = np.random.choice(train_WAidx, size=len(train_AAidx), replace=False).tolist()
            print('train wa sampled', len(train_WAidx_sampled))

            while sim_key:                
                print('sim round', sim_round)    

                seed = 42 + sim_round
                np.random.seed(seed)
                train_AAWAidx = train_WAidx_sampled.copy()
                print('WA train size:', len(train_AAWAidx))
                train_AAidx_sampled = np.random.choice(train_AAidx, size=min(sim_round, len(train_AAidx)), replace=False).tolist()
                if sim_round > len(train_AAidx):
                    extra_samples = np.random.choice(train_AAidx, size=sim_round - len(train_AAidx), replace=True).tolist()
                    train_AAidx_sampled += extra_samples
                train_AAWAidx += train_AAidx_sampled
                print('wa+aa train size:', len(train_AAWAidx))
                x_tr = set_df_tr.loc[train_AAWAidx]
                x_te = set_df_te.loc[test_idx]
                y_tr = pd.DataFrame(targs.loc[train_AAWAidx][targ].values, index=x_tr.index, columns=[target_name])
                y_te = pd.DataFrame(targs.loc[test_idx][targ].values, index=x_te.index, columns=[target_name])

                if not x_te.index.equals(y_te.index):
                    print(targ, set_name, 'index mismatch')

                print(sim_key, sim_round, x_tr.shape, y_tr.shape)           

                wa_best_mae, best_mae, best_round, wa_best_round = run_RF2(
                    rf2_dict, x_tr, x_te, y_tr, y_te,
                    wa_best_mae, best_mae, best_round, wa_best_round, sim_round,
                    test_AAidx, test_WAidx, set_name, tolerance, len(train_AAidx), len(train_WAidx)
                )
                if sim_round >= max_sim:
                    sim_key = False
                    print(sim_key, sim_round, best_mae, best_round, wa_best_mae, wa_best_round)
                sim_round += 1
            # save dict
        joblib.dump(rf2_dict, rf2_dir + target_name + set_name + '_rf2_output_std_sim_Ewaaa.joblib', compress=0)
        gc.collect()
    # joblib.dump(rf2_model, rf2_dir + targ + 'rf2-1_model_std.joblib', compress=0) 
    # # Save rf2_dict using pickle
    # with open(rf2_dir + targ + 'rf2-1_output_std.pkl', 'wb') as f:
    #     pickle.dump(rf2_dict, f)
    # with open(rf2_dir + targ + 'rf2-1_model_std.pkl', 'wb') as f:
    #     pickle.dump(rf2_model, f)
   




