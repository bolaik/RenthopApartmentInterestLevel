import numpy as np
import pandas as pd
import geocoder
import time
import json
from sklearn.cluster import KMeans

t1 = pd.read_csv('../input/train.csv')
s1 = pd.read_csv('../input/test.csv')

#t1 = pd.read_json('../input/train.json')
#s1 = pd.read_json('../input/test.json')

# replace prediction object with (0,1,2)
interest_level_map = {'low':0, 'medium':1, 'high':2}
t1['interest_level'] = t1['interest_level'].apply(lambda x: interest_level_map[x])
s1['interest_level'] = -1

# combine train and test data
ts1 = t1.append(s1)

# number of photos, features, description words
ts1['feature_num'] = [len(x) for x in ts1['features']]
ts1['photo_num'] = [len(x) for x in ts1['photos']]
ts1['desc_wordcount'] = ts1['description'].apply(lambda x: len(str(x).split(' ')))

# outliers of geographical data
outlier_addrs = ts1[(ts1.longitude==0) | (ts1.latitude==0)]
outlier_ny = [''.join([street_addr, ', new york']) for street_addr in outlier_addrs.street_address]
coords = [geocoder.google(street_addr).latlng for street_addr in outlier_ny]
ts1.loc[(ts1.longitude==0) | (ts1.latitude==0), ['latitude', 'longitude']] = coords

# distance to city center
ny_center = geocoder.google('new york').latlng
ts1['distance_city'] = list(map(lambda lng, lat: np.sqrt((lat - ny_center[0])**2 + (lng - ny_center[1])**2), ts1['longitude'], ts1['latitude']))

# created time
ts1['day_of_year'] = [time.strptime(x, '%Y-%m-%d %H:%M:%S').tm_yday for x in ts1['created']]
ts1['created'] = pd.to_datetime(ts1['created'])
ts1['created_month'] = ts1['created'].dt.month
ts1['created_day'] = ts1['created'].dt.day
ts1['created_hour'] = ts1['created'].dt.hour
ts1['day_of_week'] = ts1['created'].dt.weekday

# bedroom, bathroom engineered features
ts1['price_bed_rt'] = [price/bed if bed != 0 else -1 for (price, bed) in zip(ts1['price'], ts1['bedrooms'])]
ts1['price_bath_rt'] = [price/bath if bath != 0 else -1 for (price, bath) in zip(ts1['price'], ts1['bathrooms'])]
ts1['price_room_rt'] = [price/(bed+bath) if (bed+bath) != 0 else -1 for (price, bed, bath) in zip(ts1['price'], ts1['bedrooms'], ts1['bathrooms'])]
ts1['bed_bath_rt'] = [bed/bath if bath != 0 else -1 for (bed, bath) in zip(ts1['bedrooms'], ts1['bathrooms'])]
ts1['bed_bath_dif'] = ts1['bedrooms'] - ts1['bathrooms']
ts1['bed_bath_sum'] = ts1['bedrooms'] + ts1['bathrooms']
ts1['bed_room_rt'] = [bed/(bed+bath) if (bed+bath) != 0 else -1 for (bed, bath) in zip(ts1['bedrooms'], ts1['bathrooms'])]

# magic features for photos
time_stamp = pd.read_csv('../input/listing_image_time.csv')
ts1 = ts1.merge(time_stamp, left_on='listing_id', right_on='Listing_Id').drop('Listing_Id', axis=1)
# mean photo size in each listing
with open('../best-plantsgo/input_plantsgo/jpgs.json') as json_data:
    img_sizes = json.load(json_data)
img_size_dic = {}
for key in img_sizes.keys():
    img_list = img_sizes[key]
    if img_list:
        mean_size = np.mean([np.prod(img) for img in img_list])
    else:
        mean_size = 0
    img_size_dic[key] = mean_size
ts1['img_sizes_mean'] = [img_size_dic.get(str(key), 0) for key in ts1['listing_id']]

# add features from descriptions
'''
descrpt = ['quiet', 'comfort', 'service', 'care', 'supermarket', 'shopping', 'subway', 'bus', 'health', 'parking', 'central park', 'large window']
for item in descrpt:
    name = '_'.join(['descrpt'] + item.split())
    ts1[name] = [s.lower().count(item) if s is not np.nan else 0 for s in ts1.description]
'''

# k-means to make grids of latlng
'''
kmeans = KMeans(n_clusters=40, random_state=2017, n_jobs=-1)
latlng = ts1[['latitude', 'longitude']].copy()
latlng_col_mean, latlng_col_std = latlng.mean(axis=0), latlng.std(axis=0)
latlng = (latlng - latlng_col_mean) / latlng_col_std
kmeans.fit(latlng)
ts1['latlng_grid_label'] = kmeans.labels_
'''


ts1 = ts1.drop(['photos', 'created', 'id', 'description'], axis=1)
ts1.to_csv('feat_input/basic_feat.csv', index=False)

#ts1.to_json('feat_input/basic_feat.json')

print(ts1.info())


