from encoder import get_label_encoder
from encoder import get_label_inter_stats
from encoder import get_stats
import pandas as pd

basic_feat = pd.read_json('feat_input/basic_feat.json')
train_df = basic_feat[basic_feat.interest_level != -1].copy()
test_df = basic_feat[basic_feat.interest_level == -1].copy()

train_stack_list = train_df[['listing_id','manager_id']].copy()
test_stack_list = test_df[['listing_id', 'manager_id']].copy()

selected_manager_id_proj = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'distance_city']

for target_col in selected_manager_id_proj:
    print('encoding: {0} on manager_id'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_col=target_col)
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)


selected_bedrooms_proj = ['price', 'bathrooms']

for target_col in selected_bedrooms_proj:
    print('encoding: {0} on bedrooms'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_col=target_col, group_col='bedrooms')
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)

selected_bathrooms_proj = ['price']
for target_col in selected_bathrooms_proj:
    print('encoding: {0} on bathrooms'.format(target_col))
    tmp_train, tmp_test = get_stats(train_df, test_df, target_col=target_col, group_col='bathrooms')
    train_stack_list = pd.concat([train_stack_list, tmp_train], axis=1)
    test_stack_list = pd.concat([test_stack_list, tmp_test], axis=1)

for group_col in ['manager_id']:
    print('encoding {0} on {1}'.format('interest_level', group_col))
    tmp_train, tmp_test = get_label_encoder(train_df, test_df, group_col=group_col)
    train_stack_list = train_stack_list.merge(tmp_train, on='listing_id')
    test_stack_list = test_stack_list.merge(tmp_test, on='listing_id')

for target_col in ['price']:
    for group_col in ['manager_id']:
        print('encoding {0} on {1}, conditioned on {2}'.format(target_col, group_col, 'interest_level'))
        tmp_train, tmp_test = get_label_inter_stats(train_df, test_df, target_col=target_col, group_col=group_col)
        train_stack_list = train_stack_list.merge(tmp_train, on='listing_id')
        test_stack_list = test_stack_list.merge(tmp_test, on='listing_id')

all_stack_list = train_stack_list.append(test_stack_list)
all_stack_list = all_stack_list.drop('manager_id', axis=1)

print('write to csv file')
all_stack_list.to_csv('feat_input/feat_stats_encoding.csv', index=False)

print(all_stack_list.info())