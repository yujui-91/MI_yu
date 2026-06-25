from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from natsort import natsorted
from tqdm import tqdm
import gc
import sys
sys.path.append(r'V:\dunwei\miSKNA\program\tools')
from tools import METRICS
gc.collect()
torch.cuda.empty_cache()

def data_train_test_split(all_data, train_id_df, test_id_df):
    train_data_dict = {}
    for train_id in train_id_df['research_id']:
        train_data = all_data[train_id].copy()
        train_data_dict[train_id] = train_data

    test_data_dict = {}
    for test_id in test_id_df['research_id']:
        test_data = all_data[test_id].copy()
        test_data_dict[test_id] = test_data

    return train_data_dict, test_data_dict

def make_fold_id(train_id_df, n_splits=10, random_state=42):
    
    ALL_TRAIN_IDS = train_id_df['research_id'].astype(str).str.strip().to_numpy()
    ALL_TRAIN_LABELS = (train_id_df['research_id'].astype(str).str.strip().str.startswith('MACE').astype(int).to_numpy())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_id_splits = []
    for train_index, valid_index in skf.split(ALL_TRAIN_IDS, ALL_TRAIN_LABELS):
        train_group_ids, valid_group_ids = ALL_TRAIN_IDS[train_index], ALL_TRAIN_IDS[valid_index]
        fold_id_splits.append((train_group_ids, valid_group_ids))

    return fold_id_splits

def combine_mantis_each_fold_by_layer(each_fold_label_dict, each_fold_proba_dict, y_label, yhat_probability, fold, layer):
    y_label = y_label.copy()
    y_label['fold'] = fold+1
    y_label['layer'] = f'layer_{layer + 1}'
    y_label['sample_idx'] = y_label.groupby(['research_id', 'fold', 'layer']).cumcount()
    each_fold_label_dict.setdefault(f'fold_{fold+1}', []).append(y_label)

    yhat_probability = yhat_probability.copy()
    yhat_probability['fold'] = fold+1
    yhat_probability['layer'] = f'layer_{layer + 1}'
    yhat_probability['sample_idx'] = yhat_probability.groupby(['research_id', 'fold', 'layer']).cumcount()
    each_fold_proba_dict.setdefault(f'fold_{fold+1}', []).append(yhat_probability)

    return each_fold_label_dict, each_fold_proba_dict

def avg_mantis_each_by_layer(result_label_dict, result_proba_dict, all_fold_label_dict, all_fold_proba_dict, dataset, save_avg_layer_path):
    os.makedirs(save_avg_layer_path, exist_ok=True)
    for fold in all_fold_label_dict.keys():
        label_data_concat = pd.concat(all_fold_label_dict[fold], ignore_index=True)
        label_first = label_data_concat.groupby(['research_id', 'sample_idx'])[f'y_{dataset}'].first().reset_index()

        proba_data_concat = pd.concat(all_fold_proba_dict[fold], ignore_index=True)
        proba_mean = proba_data_concat.groupby(['research_id', 'sample_idx'])[f'yhat_{dataset}_proba'].mean().reset_index()

        merged_data_df = pd.merge(label_first, proba_mean, on=['research_id', 'sample_idx'])
        merged_data_df[f'yhat_{dataset}'] = (merged_data_df[f'yhat_{dataset}_proba'].values > 0.5).astype(int)

        label_data_merged = merged_data_df[['research_id', f'y_{dataset}', f'yhat_{dataset}']]
        proba_data_merged = merged_data_df[['research_id', f'yhat_{dataset}_proba']]

        fold_num = fold.split('_')[-1]
        label_data_merged.to_csv(os.path.join(save_avg_layer_path, f'y_{dataset}_label_{fold_num}.csv'), index=False)
        proba_data_merged.to_csv(os.path.join(save_avg_layer_path, f'yhat_{dataset}_probability_{fold_num}.csv'), index=False)

        result_label_dict[fold] = label_data_merged 
        result_proba_dict[fold] = proba_data_merged

    return result_label_dict, result_proba_dict


def mantis_xgb(mantis_dataset_path, mantis_save_run_path, train_id_df, test_id_df, fold_id_splits, hyperparams_dict):

    train_each_fold_label_dict, valid_each_fold_label_dict, test_each_fold_label_dict = {}, {}, {}
    train_each_fold_proba_dict, valid_each_fold_proba_dict, test_each_fold_proba_dict = {}, {}, {}
    train_avg_layer_label_dict, valid_avg_layer_label_dict, test_avg_layer_label_dict = {}, {}, {}
    train_avg_layer_proba_dict, valid_avg_layer_proba_dict, test_avg_layer_proba_dict = {}, {}, {}

    for layer in range(6):
        print(f'========== Processing Layer {layer + 1} / 6 ==========')
        save_layer_dir = os.path.join(mantis_save_run_path, f'layer_{layer+1}/')
        save_layer_combine_dir = save_layer_dir + "combine/"
        os.makedirs(save_layer_dir, exist_ok=True)
        os.makedirs(save_layer_combine_dir, exist_ok=True)

        pca_features_num = []

        with np.load(os.path.join(mantis_dataset_path, f'mantis_all_data_dict_layer{layer+1}.npz'), allow_pickle=True) as mantis_all_data_dict:
            train_data_dict, test_data_dict = data_train_test_split(mantis_all_data_dict, train_id_df, test_id_df)

            test_list, test_ids = [], []
            for test_id in test_id_df['research_id']:
                test_data = test_data_dict[test_id]
                test_list.append(test_data)
                test_ids.extend([str(test_id)] * test_data.shape[0])

            test = np.vstack(test_list)
            X_test, y_test = test[:, 1:], test[:, 0]

            for fold, (train_group_ids, valid_group_ids) in enumerate(fold_id_splits):
                print(f'--- Fold {fold + 1} / {len(fold_id_splits)} ---')

                train_list, train_ids = [], []
                valid_list, valid_ids = [], []

                for train_id in train_group_ids:
                    train_data = train_data_dict[train_id]
                    train_list.append(train_data)
                    train_ids.extend([str(train_id)] * train_data.shape[0])

                for valid_id in valid_group_ids:
                    valid_data = train_data_dict[valid_id]
                    valid_list.append(valid_data)
                    valid_ids.extend([str(valid_id)] * valid_data.shape[0])

                train = np.vstack(train_list)
                valid = np.vstack(valid_list)

                X_train, y_train = train[:, 1:], train[:, 0]
                X_valid, y_valid = valid[:, 1:], valid[:, 0]
                print(f'Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}')

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_valid)
                X_test_scaled = scaler.transform(X_test)

                pca = PCA(n_components=0.95, random_state=42)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_valid_pca = pca.transform(X_valid_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                print(f'PCA X_Train shape : {X_train_pca.shape}, PCA X_Valid shape : {X_valid_pca.shape}, PCA X_Test shape : {X_test_pca.shape}')
                pca_features_num.append({'num_features': X_train_pca.shape[1]})

                del X_train, X_valid, X_train_scaled, X_valid_scaled, X_test_scaled, train_list, valid_list, train, valid
                gc.collect()
                torch.cuda.empty_cache()


                model = XGBClassifier(**hyperparams_dict)
                model.fit(X_train_pca, y_train)

                yhat_train_proba = model.predict_proba(X_train_pca)[:, 1]
                yhat_train = (yhat_train_proba > 0.5).astype(int)

                yhat_valid_proba = model.predict_proba(X_valid_pca)[:, 1]
                yhat_valid = (yhat_valid_proba > 0.5).astype(int)

                yhat_test_proba = model.predict_proba(X_test_pca)[:, 1]
                yhat_test = (yhat_test_proba > 0.5).astype(int)

                y_train_label = pd.DataFrame({'research_id': train_ids, 'y_train': y_train, 'yhat_train': yhat_train})
                yhat_train_probability = pd.DataFrame({'research_id': train_ids, 'yhat_train_proba': yhat_train_proba})
                y_valid_label = pd.DataFrame({'research_id': valid_ids, 'y_valid': y_valid, 'yhat_valid': yhat_valid})
                yhat_valid_probability = pd.DataFrame({'research_id': valid_ids, 'yhat_valid_proba': yhat_valid_proba})
                y_test_label = pd.DataFrame({'research_id': test_ids, 'y_test': y_test, 'yhat_test': yhat_test})
                yhat_test_probability = pd.DataFrame({'research_id': test_ids, 'yhat_test_proba': yhat_test_proba})

                y_train_label.to_csv(os.path.join(save_layer_dir, f'y_train_label_{fold+1}.csv'), index=False)
                yhat_train_probability.to_csv(os.path.join(save_layer_dir, f'yhat_train_probability_{fold+1}.csv'), index=False)
                y_valid_label.to_csv(os.path.join(save_layer_dir, f'y_valid_label_{fold+1}.csv'), index=False)
                yhat_valid_probability.to_csv(os.path.join(save_layer_dir, f'yhat_valid_probability_{fold+1}.csv'), index=False)
                y_test_label.to_csv(os.path.join(save_layer_dir, f'y_test_label_{fold+1}.csv'), index=False)
                yhat_test_probability.to_csv(os.path.join(save_layer_dir, f'yhat_test_probability_{fold+1}.csv'), index=False)


                train_all_fold_label_dict, train_all_fold_proba_dict = combine_mantis_each_fold_by_layer(train_each_fold_label_dict, train_each_fold_proba_dict, y_train_label, yhat_train_probability, fold, layer)
                valid_all_fold_label_dict, valid_all_fold_proba_dict = combine_mantis_each_fold_by_layer(valid_each_fold_label_dict, valid_each_fold_proba_dict, y_valid_label, yhat_valid_probability, fold, layer)
                test_all_fold_label_dict, test_all_fold_proba_dict = combine_mantis_each_fold_by_layer(test_each_fold_label_dict, test_each_fold_proba_dict, y_test_label, yhat_test_probability, fold, layer)

                del model
                del scaler, pca
                del X_train_pca, X_valid_pca, X_test_pca
                del y_train, y_valid
                del yhat_train, yhat_valid, yhat_test
                del yhat_train_proba, yhat_valid_proba, yhat_test_proba
                del y_train_label, y_valid_label, y_test_label
                del yhat_train_probability, yhat_valid_probability, yhat_test_probability
                del train_ids, valid_ids

                gc.collect()
                torch.cuda.empty_cache()

            if pca_features_num:
                pca_features_df = pd.DataFrame(pca_features_num)
                pca_features_df.to_csv(os.path.join(save_layer_combine_dir, 'pca_features_num.csv'), index=False)
                del pca_features_df
        
            del test_list, test, X_test, y_test, test_ids
            del pca_features_num
            gc.collect()
            torch.cuda.empty_cache()

    save_avg_layer_path = os.path.join(mantis_save_run_path, 'layer_avg/')
    train_avg_layer_label_dict, train_avg_layer_proba_dict = avg_mantis_each_by_layer(train_avg_layer_label_dict, train_avg_layer_proba_dict, train_all_fold_label_dict, train_all_fold_proba_dict, dataset='train', save_avg_layer_path=save_avg_layer_path)
    valid_avg_layer_label_dict, valid_avg_layer_proba_dict = avg_mantis_each_by_layer(valid_avg_layer_label_dict, valid_avg_layer_proba_dict, valid_all_fold_label_dict, valid_all_fold_proba_dict, dataset='valid', save_avg_layer_path=save_avg_layer_path)
    test_avg_layer_label_dict, test_avg_layer_proba_dict = avg_mantis_each_by_layer(test_avg_layer_label_dict, test_avg_layer_proba_dict, test_all_fold_label_dict, test_all_fold_proba_dict, dataset='test', save_avg_layer_path=save_avg_layer_path)

    return train_avg_layer_label_dict, train_avg_layer_proba_dict, valid_avg_layer_label_dict, valid_avg_layer_proba_dict, test_avg_layer_label_dict, test_avg_layer_proba_dict

def ecgfounder_xgb(ecgfounder_dataset_path, ecgfounder_save_run_path, train_id_df, test_id_df, fold_id_splits, hyperparams_dict):
    save_combine_dir = ecgfounder_save_run_path + "combine/"
    os.makedirs(save_combine_dir, exist_ok=True)
    ecgfounder_all_data_dict = np.load(os.path.join(ecgfounder_dataset_path, f'ecg_all_data_dict.npz'), allow_pickle=True)

    train_ecgfounder_label_dict, valid_ecgfounder_label_dict, test_ecgfounder_label_dict = {}, {}, {}
    train_ecgfounder_proba_dict, valid_ecgfounder_proba_dict, test_ecgfounder_proba_dict = {}, {}, {}
    
    train_data_dict, test_data_dict = data_train_test_split(ecgfounder_all_data_dict, train_id_df, test_id_df)
    
    # pca_features_num = []

    test_list, test_ids = [], []
    for test_id in test_id_df['research_id']:
        test_data = test_data_dict[test_id]
        test_list.append(test_data)
        test_ids.extend([str(test_id)] * test_data.shape[0])
        
    test = np.vstack(test_list)
    X_test, y_test = test[:, 1:], test[:, 0]
    
    for fold, (train_group_ids, valid_group_ids) in enumerate(fold_id_splits):
        print(f'--- Fold {fold + 1} / {len(fold_id_splits)} ---')
        train_list, train_ids = [], []
        valid_list, valid_ids = [], []

        for train_id in train_group_ids:
            train_data = train_data_dict[train_id]
            train_list.append(train_data)
            train_ids.extend([str(train_id)] * train_data.shape[0])

        for valid_id in valid_group_ids:
            valid_data = train_data_dict[valid_id]
            valid_list.append(valid_data)
            valid_ids.extend([str(valid_id)] * valid_data.shape[0])

        train = np.vstack(train_list)
        valid = np.vstack(valid_list)

        X_train, y_train = train[:, 1:], train[:, 0]
        X_valid, y_valid = valid[:, 1:], valid[:, 0]
        print(f'Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}')


        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_valid_scaled = scaler.transform(X_valid)
        # X_test_scaled = scaler.transform(X_test)

        # pca = PCA(n_components=0.95, random_state=42)
        # X_train_pca = pca.fit_transform(X_train_scaled)
        # X_valid_pca = pca.transform(X_valid_scaled)
        # X_test_pca = pca.transform(X_test_scaled)
        # print(f'PCA X_Train shape : {X_train_pca.shape}, PCA X_Valid shape : {X_valid_pca.shape}, PCA X_Test shape : {X_test_pca.shape}')
        # pca_features_num.append({'num_features': X_train_pca.shape[1]})

        del train_list, valid_list, train, valid
        gc.collect()
        torch.cuda.empty_cache()

        model = XGBClassifier(**hyperparams_dict)
        model.fit(X_train, y_train)

        yhat_train_proba = model.predict_proba(X_train)[:, 1]
        yhat_train = (yhat_train_proba > 0.5).astype(int)

        yhat_valid_proba = model.predict_proba(X_valid)[:, 1]
        yhat_valid = (yhat_valid_proba > 0.5).astype(int)

        yhat_test_proba = model.predict_proba(X_test)[:, 1]
        yhat_test = (yhat_test_proba > 0.5).astype(int)

        y_train_label = pd.DataFrame({'research_id': train_ids, 'y_train': y_train, 'yhat_train': yhat_train})
        yhat_train_probability = pd.DataFrame({'research_id': train_ids, 'yhat_train_proba': yhat_train_proba})
        y_valid_label = pd.DataFrame({'research_id': valid_ids, 'y_valid': y_valid, 'yhat_valid': yhat_valid})
        yhat_valid_probability = pd.DataFrame({'research_id': valid_ids, 'yhat_valid_proba': yhat_valid_proba})
        y_test_label = pd.DataFrame({'research_id': test_ids, 'y_test': y_test, 'yhat_test': yhat_test})
        yhat_test_probability = pd.DataFrame({'research_id': test_ids, 'yhat_test_proba': yhat_test_proba})

        y_train_label.to_csv(os.path.join(ecgfounder_save_run_path, f'y_train_label_fold{fold+1}.csv'), index=False)
        yhat_train_probability.to_csv(os.path.join(ecgfounder_save_run_path, f'yhat_train_probability_fold{fold+1}.csv'), index=False)
        y_valid_label.to_csv(os.path.join(ecgfounder_save_run_path, f'y_valid_label_fold{fold+1}.csv'), index=False)
        yhat_valid_probability.to_csv(os.path.join(ecgfounder_save_run_path, f'yhat_valid_probability_fold{fold+1}.csv'), index=False)
        y_test_label.to_csv(os.path.join(ecgfounder_save_run_path, f'y_test_label_fold{fold+1}.csv'), index=False)
        yhat_test_probability.to_csv(os.path.join(ecgfounder_save_run_path, f'yhat_test_probability_fold{fold+1}.csv'), index=False)

        train_ecgfounder_label_dict[f'fold_{fold+1}'] = y_train_label
        train_ecgfounder_proba_dict[f'fold_{fold+1}'] = yhat_train_probability
        valid_ecgfounder_label_dict[f'fold_{fold+1}'] = y_valid_label
        valid_ecgfounder_proba_dict[f'fold_{fold+1}'] = yhat_valid_probability
        test_ecgfounder_label_dict[f'fold_{fold+1}'] = y_test_label
        test_ecgfounder_proba_dict[f'fold_{fold+1}'] = yhat_test_probability

        del model
        del X_train, X_valid
        del y_train, y_valid
        # del X_train_pca, X_valid_pca, X_test_pca
        # del scaler, pca
        # del X_train_scaled, X_valid_scaled, X_test_scaled
        del yhat_train, yhat_valid, yhat_test
        del yhat_train_proba, yhat_valid_proba, yhat_test_proba
        del y_train_label, y_valid_label, y_test_label
        del yhat_train_probability, yhat_valid_probability, yhat_test_probability
        del train_ids, valid_ids
        gc.collect()
        torch.cuda.empty_cache()


        # if pca_features_num:
        #     pca_features_df = pd.DataFrame(pca_features_num)
        #     pca_features_df.to_csv(os.path.join(save_combine_dir, 'pca_features_num.csv'), index=False)
        #     del pca_features_df

    del test_list, test, X_test, y_test, test_ids

    ecgfounder_all_data_dict.close()
    del ecgfounder_all_data_dict
    gc.collect()
    torch.cuda.empty_cache()

    return train_ecgfounder_label_dict, train_ecgfounder_proba_dict, valid_ecgfounder_label_dict, valid_ecgfounder_proba_dict, test_ecgfounder_label_dict, test_ecgfounder_proba_dict