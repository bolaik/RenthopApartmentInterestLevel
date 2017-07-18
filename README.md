# RenthopApartmentInterestLevel
Solution posts to Kaggle competition [Two sigma connect Renthop for rental listing inquries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)

## features selected

The basic features include features from [Branden Murray's post](https://www.kaggle.com/brandenkmurray/it-is-lit) and some further revisions.

## advanced features

- encoded features: This is based on the fact that some categorical turns out to have pretty high impact on prediction.
    - encode some numerical features, such as `bedrooms`, `distance_city` on `manager_id`, by taking the mean of features grouped by the categorical feature.
    - inspired by this, one can also encode important numerical features, such as `price`, on other features.
    - encode prediction (i.e. `interest_level`) on `manager_id`, requires cross validation.
    - encode (group mean, other statistics could also be defined) other numerical features (e.g. `price`) on categorical features (e.g. `manager_id`) conditioned on `interest_level`, also request cross validation. 
- geological features: local price fluctuation, from `plantsgo`
- images features: also see [magic feature](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31870)

## classifiers

- Trained different machine learning models, including xgboost, lightgbm, nn, adaboost, gb, rf, et, lsvc, lr, knn 
- check out <https://github.com/bolaik/RenthopApartmentInterestLevel/tree/master/my_final_version/classifiers_ipynb>

## Ensemble

- Train level-2 models, including xgb, nn, knn, lr, lightgbm, where xgb prediction is submitted because of best cv score.

## many helpful links for reference

- [Check out top-1 solution and github repo for many insightful ideas about feature engineering](https://github.com/plantsgo/Rental-Listing-Inquiries)
- [List of solution posts](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32119)
- [My best solution only using manager_id](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32114)
- [Unsupervised and supervised neighborhood encoding](https://www.kaggle.com/arnaldcat/unsupervised-and-supervised-neighborhood-encoding)
- [Price compared to neighborhood median](https://www.kaggle.com/luisblanche/price-compared-to-neighborhood-median)
- [A proxy for $/sqft and the interest on 1/2-baths](https://www.kaggle.com/arnaldcat/a-proxy-for-sqft-and-the-interest-on-1-2-baths)
