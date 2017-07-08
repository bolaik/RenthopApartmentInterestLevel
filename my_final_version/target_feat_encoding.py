import pandas as pd
import numpy as np

def get_stats(train_df, test_df, target_column, group_column = 'manager_id'):
    '''
    target_column: numeric columns to group with (e.g. price, bedrooms, bathrooms)
    group_column: categorical columns to group on (e.g. manager_id, building_id)
    '''

    train_df['row_id'] = train_df.index
    test_df['row_id'] = test_df.index
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    all_df = train_df[['row_id', 'is_train', target_column, group_column]].append(test_df[['row_id','is_train', target_column, group_column]])

    grouped = all_df[[target_column, group_column]].groupby(group_column)
    the_size = pd.DataFrame(grouped.size()).reset_index()
    the_size.columns = [group_column, '%s_size_%s' % (target_column, group_column)]
    the_mean = pd.DataFrame(grouped.mean()).reset_index()
    the_mean.columns = [group_column, '%s_mean_%s' % (target_column, group_column)]
    the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
    the_std.columns = [group_column, '%s_std_%s' % (target_column, group_column)]
    the_median = pd.DataFrame(grouped.median()).reset_index()
    the_median.columns = [group_column, '%s_median_%s' % (target_column, group_column)]
    the_stats = pd.merge(the_size, the_mean)
    the_stats = pd.merge(the_stats, the_std)
    the_stats = pd.merge(the_stats, the_median)

    the_max = pd.DataFrame(grouped.max()).reset_index()
    the_max.columns = [group_column, '%s_max_%s' % (target_column, group_column)]
    the_min = pd.DataFrame(grouped.min()).reset_index()
    the_min.columns = [group_column, '%s_min_%s' % (target_column, group_column)]

    the_stats = pd.merge(the_stats, the_max)
    the_stats = pd.merge(the_stats, the_min)

    all_df = pd.merge(all_df, the_stats)

    selected_train = all_df[all_df['is_train'] == 1].copy()
    selected_test = all_df[all_df['is_train'] == 0].copy()
    selected_train.index = selected_train['row_id']
    selected_test.index = selected_test['row_id']
    selected_train.drop([target_column, group_column, 'row_id', 'is_train'], axis=1, inplace=True)
    selected_test.drop([target_column, group_column, 'row_id', 'is_train'], axis=1, inplace=True)

    return selected_train, selected_test

basic_feat = pd.read_csv('feat_input/basic_feat.csv')
train_df = basic_feat[basic_feat.interest_level != -1].copy()
test_df = basic_feat[basic_feat.interest_level == -1].copy()

train_stack_list = train_df[['listing_id','manager_id']].copy()
test_stack_list = test_df[['listing_id', 'manager_id']].copy()

'''
selected_manager_id_proj = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'listing_id', 'feature_num',
                            'photo_num', 'desc_wordcount', 'distance_city', 'day_of_year', 'created_month', 'created_day',
                            'created_hour', 'day_of_week', 'price_bed_rt', 'price_bath_rt', 'price_room_rt', 'bed_bath_rt',
                            'bed_bath_dif', 'bed_bath_sum', 'bed_bath_sum', 'bed_room_rt', 'time_stamp', 'img_sizes_mean']
'''

selected_manager_id_proj = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'distance_city']

for target_col in selected_manager_id_proj:
    print('encoding: {0} on manager_id'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col)
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)


selected_bedrooms_proj = ['price', 'bathrooms']

for target_col in selected_bedrooms_proj:
    print('encoding: {0} on bedrooms'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col, group_column='bedrooms')
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)

selected_bathrooms_proj = ['price']
for target_col in selected_bathrooms_proj:
    print('encoding: {0} on bathrooms'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col, group_column='bathrooms')
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)

'''
selected_latlng_grid_label_proj = ['price']
for target_col in selected_latlng_grid_label_proj:
    print('encoding: {0} on bathrooms'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col, group_column='latlng_grid_label')
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)
'''

all_stack_list = train_stack_list.append(test_stack_list)
all_stack_list = all_stack_list.drop('manager_id', axis=1)
all_stack_list.to_csv('feat_input/feat_encoding.csv', index=False)

print(all_stack_list.info())