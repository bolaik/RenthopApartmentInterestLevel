import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.externals import joblib
import re
from nltk.stem.snowball import SnowballStemmer
import lightgbm as lgb

lgb.cv()

def xgb_data_prep():
#    basic_feat = pd.read_json('feat_input/basic_feat.json')
#    basic_feat.replace('', np.nan)
    basic_feat = pd.read_csv('feat_input/basic_feat.csv')
    longtime_feat = pd.read_csv('feat_input/longtime_feat.csv')
    encoded_feat = pd.read_csv('feat_input/feat_stats_encoding.csv')
    print('Loading features finished')

    # apply ordinal encoding to categorical feature

#    basic_feat.display_address = basic_feat.display_address.replace(r'\r$', '', regex=True)
#    basic_feat.street_address = basic_feat.street_address.replace(r'\r$', '', regex=True)

    categorical = ["display_address", "manager_id", "building_id", "street_address"]
    for f in categorical:
        if basic_feat[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(basic_feat[f].values))
            basic_feat[f] = lbl.transform(list(basic_feat[f].values))
    print('Ordinal encoding finished')

    all = basic_feat.merge(longtime_feat, on='listing_id')
    all = all.merge(encoded_feat, on='listing_id')

    train = all[all.interest_level != -1].copy()
    test = all[all.interest_level == -1].copy()
    y_train=train["interest_level"]

    '''
    train_num=train.shape[0]
    stemmer = SnowballStemmer('english')

    all['features'] = all['features'].apply(lambda x: re.findall(r"[\w'|\w\-|\ ]+", x))
    all['features'] = all['features'].apply(lambda x: [stemmer.stem(y) for y in x])
    all['features'] = all['features'].apply(lambda x: ['_'.join(['feature'] + y.split()) for y in x])
    all['features'] = all['features'].apply(lambda x: ' '.join(x))
    dtm = CountVectorizer(max_features=100, token_pattern=r"[\w\-|\w']+")

#    all['features'] = all['features'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))
#    dtm = CountVectorizer(stop_words='english', max_features=100)

    all_sparse = dtm.fit_transform(all["features"].values.astype('U'))
    print(dtm.get_feature_names())

    tr_sparse = all_sparse[:train_num]
    te_sparse = all_sparse[train_num:]
    print("Document-term matrix finished")
    '''

    x_train = train.drop(["interest_level","features"],axis=1)
    x_test = test.drop(["interest_level","features"],axis=1)

#    x_train = sparse.hstack([x_train.astype(float),tr_sparse.astype(float)]).tocsr()
#    x_test = sparse.hstack([x_test.astype(float),te_sparse.astype(float)]).tocsr()

    return x_train, y_train, x_test, test.listing_id

def xgb_cv(dtrain, num_rounds = 50000, early_stop_rounds=250):
    print('Start xgboost cross-validation')
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'gamma': 1,
              'min_child_weight': 1.5,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.03,
              'tree_method': 'exact',
              'seed': 36683,
              'nthread': 12,
              'num_class': 3,
              'silent': 1
              }
    xgb2cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_rounds,
                    nfold=5,
                    stratified=True,
                    verbose_eval=50,
                    early_stopping_rounds=early_stop_rounds)
    return xgb2cv


def xgb_train(dtrain, num_rounds = 10000):
    print('Start xgboost training')
    params = {'booster': 'gbtree',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'gamma': 1,
              'min_child_weight': 1.5,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.03,
              'tree_method': 'exact',
              'seed': 36683,
              'nthread': 12,
              'num_class': 3,
              'silent': 1
              }
    watchlist = [(dtrain, 'train')]
    xgb2train = xgb.train(params=params,
                          dtrain=dtrain,
                          evals=watchlist,
                          verbose_eval=50,
                          num_boost_round=num_rounds)
    return xgb2train


def main():
    X_train, y_train, X_test, listing_id = xgb_data_prep()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # cross-validation and training
    xgb_cv_hist = xgb_cv(dtrain)
    xgb_clf = xgb_train(dtrain, xgb_cv_hist.shape[0])

    # model prediction on test data
    print('Start model prediction')
    preds = xgb_clf.predict(dtest)
    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = listing_id.values
    out_df.to_csv("output/xgb_cv5.csv", index=False)

    # save model to file
    print('Start dump model to file')
    joblib.dump(xgb_clf, "output/xgb_cv5_joblib.dat")


if __name__=="__main__":
    main()
