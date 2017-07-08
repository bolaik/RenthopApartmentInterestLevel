import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from collections import defaultdict

def get_stats(train_df, test_df, target_col, group_col = 'manager_id'):
    '''
    Obtain statistics (count, min, max, std, mean, median) of numerical features.
    :param train_df:
    :param test_df:
    :param target_col: numeric columns to group with (e.g. price, bedrooms, bathrooms)
    :param group_col: categorical columns to group on (e.g. manager_id, building_id)
    :return: dataframe of encoded statistics for train_df and test_df, with same row-indexing
    as original train_df and test_df
    '''

    train_df['row_id'], test_df['row_id'] = train_df.index, test_df.index
    train_df['is_train'], test_df['is_train'] = 1, 0
    all_df = train_df[['row_id', 'is_train', target_col, group_col]].append(test_df[['row_id','is_train', target_col, group_col]])

    grouped = all_df[[target_col, group_col]].groupby(group_col)
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_col, '%s_size_%s' % (target_col, group_col)]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_col, '%s_mean_%s' % (target_col, group_col)]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_col, '%s_std_%s' % (target_col, group_col)]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_col, '%s_median_%s' % (target_col, group_col)]
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)

    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_col, '%s_max_%s' % (target_col, group_col)]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_col, '%s_min_%s' % (target_col, group_col)]

    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)

    all_df = pd.merge(all_df, the_stats)

    selected_train = all_df[all_df['is_train'] == 1].copy()
    selected_test = all_df[all_df['is_train'] == 0].copy()
    selected_train.index = selected_train['row_id']
    selected_test.index = selected_test['row_id']
    selected_train.drop([target_col, group_col, 'row_id', 'is_train'], axis=1, inplace=True)
    selected_test.drop([target_col, group_col, 'row_id', 'is_train'], axis=1, inplace=True)

    return selected_train, selected_test


# calculate lambda_n as a sigmoid function
def sgm(n, k=1, f=1):
    return 1 / (1 + np.exp( - (n - k) / f))

def get_label_encoder(train_df, test_df, label_col = 'interest_level', group_col = 'manager_id', kfold = 5, seed = 816):
    '''
    Apply cross-validation to get statistics of output labels encoded on certain categorical features.
    Statistics for test_df is encoded with train_df
    :param train_df:
    :param test_df:
    :param label_col: column of predicted labels
    :param group_col: categorial column for the label column to group on (e.g. manager_id, latlng_grid)
    :return: dataframe of encoded statistics for train and test set
    '''

    # find overall rating for manager_id's with only one entry
    # applied to estimate manager_id in validation and test set, which is missing in train
    grouped = train_df[[group_col, label_col]].groupby(group_col)
    the_size = pd.DataFrame(grouped.count()).reset_index()
    record1_id = the_size.loc[the_size[label_col]==1, group_col]
    record1 = train_df[train_df[group_col].isin(record1_id)][label_col].value_counts()
    record1 = record1 / sum(record1)

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    x_train, y_train = train_df.loc[:, train_df.columns != label_col], train_df[label_col]

    train_low = [np.nan]*train_df.shape[0]
    train_medium = [np.nan]*train_df.shape[0]
    train_high = [np.nan]*train_df.shape[0]
    for train_ind, val_ind in skf.split(x_train, y_train):
        bld_level = defaultdict(dict)
        for it in train_ind:
            id = x_train.loc[it, group_col]
            label = y_train.iloc[it]
            bld_level[id][label] = bld_level[id].get(label, 0) + 1
        for id in bld_level.keys():
            for label in bld_level[id].keys():
                bld_level[id][label] /= sum(bld_level[id].values())
        for iv in val_ind:
            id = x_train.loc[iv, group_col]
            try:
                train_low[iv] = bld_level[id][0]
                train_medium[iv] = bld_level[id][1]
                train_high[iv] = bld_level[id][2]
            except:
                train_low[iv] = record1[0]
                train_medium[iv] = record1[1]
                train_high[iv] = record1[2]
    train_df[''.join([group_col, '_low'])] = train_low
    train_df[''.join([group_col, '_medium'])] = train_medium
    train_df[''.join([group_col, '_high'])] = train_high

    test_low = [np.nan] * test_df.shape[0]
    test_medium = [np.nan] * test_df.shape[0]
    test_high = [np.nan] * test_df.shape[0]
    bld_level = defaultdict(dict)
    for id, label in zip(x_train[group_col], y_train):
        bld_level[id][label] = bld_level[id].get(label, 0) + 1
    for id in bld_level.keys():
        for label in bld_level[id].keys():
            bld_level[id][label] /= sum(bld_level[id].values())
    for i, id in enumerate(test_df[group_col]):
        try:
            test_low[i] = bld_level[id][0]
            test_medium[i] = bld_level[id][1]
            test_high[i] = bld_level[id][2]
        except:
            test_low[i] = record1[0]
            test_medium[i] = record1[1]
            test_high[i] = record1[2]
    test_df[''.join([group_col, '_low'])] = test_low
    test_df[''.join([group_col, '_medium'])] = test_medium
    test_df[''.join([group_col, '_high'])] = test_high

    features_to_use = ['listing_id', ''.join([group_col, '_low']), ''.join([group_col, '_medium']), ''.join([group_col, '_high'])]
    return train_df[features_to_use], test_df[features_to_use]


def get_label_inter_stats(train_df, test_df, target_col, group_col='manager_id', label_col='interest_level', kfold=5, seed=816):
    '''
    within each level in label_col, do the group_col stats summary of the target_col feature
    :param train_df:
    :param test_df:
    :param target_col: feature stats on
    :param group_col:  feature encoded on
    :param label_col: prediction level
    :param kfold: for cross-validation
    :param seed: set random seed
    :return: train, test set after stats encoding
    '''

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    x_train, y_train = train_df.loc[:, train_df.columns != label_col], train_df[label_col]
    labels = sorted(y_train.unique())
    v0, v1, v2 = ['%s_%s_mean_pred_%s' % (target_col, group_col, str(label)) for label in labels]

    train_low = [np.nan]*train_df.shape[0]
    train_medium = [np.nan]*train_df.shape[0]
    train_high = [np.nan]*train_df.shape[0]
    for k, (train_ind, val_ind) in enumerate(skf.split(x_train, y_train)):
        print('Start cv fold {0}'.format(k))
        train = train_df.loc[train_ind, [group_col, target_col, label_col]].copy()
        train = train.join(pd.get_dummies(train[label_col], prefix='pred').astype(int))
        for label in labels:
            train[''.join([target_col, '_', str(label)])] = train[target_col] * train[''.join(['pred_', str(label)])]
            train = train.drop([''.join(['pred_', str(label)])], axis=1)
        train = train.drop([target_col, label_col], axis=1)
        g = train.groupby(group_col)
        gmean = pd.DataFrame(g.mean()).reset_index()
        gmean.columns = [group_col, v0, v1, v2]
        stats_df = gmean

        for iv in val_ind:
            id = train_df.loc[iv, group_col]
            try:
                train_low[iv] = float(stats_df.loc[stats_df[group_col]==id, v0])
                train_medium[iv] = float(stats_df.loc[stats_df[group_col]==id, v1])
                train_high[iv] = float(stats_df.loc[stats_df[group_col]==id, v2])
            except:
                train_low[iv] = -1
                train_medium[iv] = -1
                train_high[iv] = -1
    train_df[v0] = train_low
    train_df[v1] = train_medium
    train_df[v2] = train_high

    print('Start prediction on test data')
    test_low = [np.nan]*test_df.shape[0]
    test_medium = [np.nan]*test_df.shape[0]
    test_high = [np.nan]*test_df.shape[0]
    train = train_df[[group_col, target_col, label_col]].copy()
    train = train.join(pd.get_dummies(train[label_col], prefix='pred').astype(int))
    for label in labels:
        train[''.join([target_col, '_', str(label)])] = train[target_col] * train[''.join(['pred_', str(label)])]
        train = train.drop([''.join(['pred_', str(label)])], axis=1)
    train = train.drop([target_col, label_col], axis=1)
    g = train.groupby(group_col)
    gmean = pd.DataFrame(g.mean()).reset_index()
    gmean.columns = [group_col, v0, v1, v2]
    stats_df = gmean

    for i, id in enumerate(test_df[group_col]):
        try:
            test_low[i] = float(stats_df.loc[stats_df[group_col]==id, v0])
            test_medium[i] = float(stats_df.loc[stats_df[group_col]==id, v1])
            test_high[i] = float(stats_df.loc[stats_df[group_col]==id, v2])
        except:
            test_low[i] = -1
            test_medium[i] = -1
            test_high[i] = -1
    test_df[v0] = test_low
    test_df[v1] = test_medium
    test_df[v2] = test_high

    feature_to_use = ['listing_id', v0, v1, v2]
    return train_df[feature_to_use], test_df[feature_to_use]
