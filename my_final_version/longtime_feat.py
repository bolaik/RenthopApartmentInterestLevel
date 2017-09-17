import math
import pandas as pd

print('load basic_feat dataset')
ts1 = pd.read_json('feat_input/basic_feat.json')

# for same-type (bath, bed) apartment in the same neighborhood, find num of apartments with lower price
print('find num of apartments of the same type with lower price')
ts1["jwd_type_low_than_num"] = list(map(lambda lo, la, ba, be, p:
                                        ts1[(ts1.latitude > la - 0.01) & (ts1.latitude < la + 0.01) & (ts1.longitude > lo - 0.01) & (ts1.longitude < lo + 0.01) &
                                            (ts1.bathrooms == ba) & (ts1.bedrooms == be) & (ts1.price <= p)].shape[0],
                                        ts1["longitude"], ts1["latitude"], ts1["bathrooms"], ts1["bedrooms"], ts1["price"]))

# same-type (bath, bed) apartment number in the same neighbourhood
print('find total num of apartments of the same type in the neighborhood')
ts1["jwd_type_ts1"] = list(map(lambda lo, la, ba, be:
                               ts1[(ts1.latitude > la - 0.01) & (ts1.latitude < la + 0.01) & (ts1.longitude > lo - 0.01) & (ts1.longitude < lo + 0.01) &
                                   (ts1.bathrooms == ba) & (ts1.bedrooms == be)].shape[0],
                               ts1["longitude"], ts1["latitude"], ts1["bathrooms"], ts1["bedrooms"]))

# percentage of apartments with lower price in the neighbourhood
print('percentage of the same-type apartment with lower price in the neighborhood')
ts1["jwd_type_rt"] = ts1["jwd_type_low_than_num"] / ts1["jwd_type_ts1"]

# (lat, lng) of buildings with zero building_id
building_zeros_la = list(ts1[ts1.building_id.astype("str") == "0"].latitude)
building_zeros_lo = list(ts1[ts1.building_id.astype("str") == "0"].longitude)
building_zeros = list(zip(building_zeros_la, building_zeros_lo))

print("num of buildings with building_id=0 in one km range")
def building_zero_num(la, lo, n):
    num = 0
    for s in building_zeros:
        slo = float(s[1])
        sla = float(s[0])
        dis = math.sqrt((la - sla) ** 2 + (lo - slo) ** 2) * 111
        if dis <= n:
            num += 1
    return num
ts1["building_zero_num"] = list(map(lambda la, lo: building_zero_num(la, lo, 1), ts1["latitude"], ts1["longitude"]))

ts1 = ts1[["jwd_type_low_than_num", "jwd_type_ts1", "jwd_type_rt", "building_zero_num", "listing_id"]]
ts1.to_csv("feat_input/longtime_feat.csv", index=False)
