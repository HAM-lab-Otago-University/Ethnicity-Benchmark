# %%
# All necessary imports
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import pickle
import gc
import traceback
import logging
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, LeavePGroupsOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr

# %%
# Command-line fold number (for HPC batch jobs)
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 4
else:
    fold_num = 0
print(fold_num)

# %%
# Paths
abcd_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/'
tables_path = root_dir
fold_base_path = os.path.join(root_dir, 'folds/tsp/')
fold_dir = os.path.join(fold_base_path, f'Fold_{fold_num}/')
if not os.path.isdir(fold_dir):
    os.makedirs(fold_dir)
pls_dir = os.path.join(fold_dir, 'pls/')
if not os.path.isdir(pls_dir):
    os.makedirs(pls_dir)

# %%
# Load features and targets
features_name = 'abccCntr'
featuresd = joblib.load(abcd_dir + features_name + '.joblib')
demo = pd.read_csv(abcd_dir + 'demo_nesi.csv', index_col=0).dropna()
targs = pd.read_csv(abcd_dir + 'cog_all.csv', index_col=0, low_memory=False).dropna()
targs.index = targs.index.str.replace('_', '')

# %%
# Function to run PLS (modular and returns updated stopping params)
def run_pls(
    pls_dict, x_tr, x_te, y_tr, y_te,
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

        perf = {'nmse': [], 'r2': [], 'pr': [], 'mae': []}
        nc = list(range(1, x_tr.shape[1] + 1)) if x_tr.shape[1] < 15 else list(range(1, 16))
        param_grid = {'n_components': nc, 'scale': [False]}

        grid_search = GridSearchCV(PLSRegression(), param_grid, scoring='neg_mean_squared_error',
                                   cv=KFold(5, shuffle=True, random_state=42))
        grid_search.fit(x_tr_scaled, y_tr_scaled)
        best_model = grid_search.best_estimator_

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

        print('test aa mae:', test_aa_mae, 'test wa mae:', test_wa_mae)

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
        pls_dict[key][sim_key_str] = {
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
            pls_dict[key]['best_aa_round'] = {
                'model': {'perf': performance, 'round_num': best_round, 'best_model': best_model},
                'data': {
                    'Xtrain': pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index),
                    'Xtest1': pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index),
                    'yptrain': pd.DataFrame(best_model.predict(x_tr_scaled), index=y_tr.index),
                    'yptest': pd.DataFrame(te_y_p, index=y_te.index),
                    'yttrain': y_tr_scaleddf,
                    'yttest': y_te_scaleddf
                }
            }

        if sim_round == wa_best_round:
            pls_dict[key]['best_wa_round'] = {
                'model': {'perf': performance, 'round_num': wa_best_round, 'best_model': best_model},
                'data': {
                    'Xtrain': pd.DataFrame(best_model.transform(x_tr_scaled), index=x_tr.index),
                    'Xtest1': pd.DataFrame(best_model.transform(x_te_scaled), index=x_te.index),
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
# Main simulation script
print(f"\nStarting Fold {fold_num}...")
Fold = joblib.load(root_dir + 'abcc_2split.joblib')
train_index = Fold[fold_num][0]
test_index = Fold[fold_num][1]

targ_list = ['nihtbx_totalcomp_uncorrected']
for targ in targ_list:
    target_name = targ.split('_')[1][0:5] + '_'
    pls_dict = {}
    for key, featu in featuresd.items():
        print(targ, key)
        feat = featu.dropna()
        common_ind = list(set(feat.index).intersection(set(demo.index), set(targs.index)))
        feature = feat.loc[common_ind]
        demo_table = demo.loc[common_ind]

        train_idx = list(set(common_ind).intersection(set(train_index)))
        test_idx = list(set(common_ind).intersection(set(test_index)))

        eth1 = demo_table[demo_table['race_ethnicity'] == 1].index
        eth2 = demo_table[demo_table['race_ethnicity'] == 2].index

        train_WAidx = list(set(eth1).intersection(set(train_idx)))
        train_AAidx = list(set(eth2).intersection(set(train_idx)))
        test_WAidx = list(set(eth1).intersection(set(test_idx)))
        test_AAidx = list(set(eth2).intersection(set(test_idx)))

        # max_sim = len(train_WAidx) * 2
        sim_key = True
        sim_round = 0

        if key not in pls_dict:
            pls_dict[key] = {}

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
        while sim_key:

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
            x_tr = feature.loc[train_AAWAidx]
            x_te = feature.loc[test_idx]
            y_tr = pd.DataFrame(targs.loc[train_AAWAidx][targ].values, index=x_tr.index, columns=[target_name])
            y_te = pd.DataFrame(targs.loc[test_idx][targ].values, index=x_te.index, columns=[target_name])

            if not x_te.index.equals(y_te.index):
                print(targ, key, 'index mismatch')

            print(sim_key, sim_round, x_tr.shape)

            wa_best_mae, best_mae, best_round, wa_best_round = run_pls(
                pls_dict, x_tr, x_te, y_tr, y_te,
                wa_best_mae, best_mae, best_round, wa_best_round, sim_round,
                test_AAidx, test_WAidx, key, tolerance, len(train_AAidx), len(train_WAidx)
            )

            # wa_aa_ratio = len(train_AAidx)/(len(train_AAidx) + sim_round)
            
            if sim_round >= max_sim:
                sim_key = False
                print(sim_key, sim_round, best_mae, best_round, wa_best_mae, wa_best_round)
            sim_round += 1
        gc.collect()
    joblib.dump(pls_dict, pls_dir + target_name + features_name + '_pls_output_std_Ewaaa_sim.joblib', compress=0)
        

