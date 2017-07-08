## fork from Brandon's script
## two features added: number of photos and number of description characters
library(data.table)
library(jsonlite)
library(h2o)
library(lubridate)
h2o.init(nthreads = -1)

# Load data
t1 <- fromJSON("../input/train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
t2 <- data.table(bathrooms=unlist(t1$bathrooms)
                 ,bedrooms=unlist(t1$bedrooms)
                 ,building_id=as.factor(unlist(t1$building_id))
                 ,created=as.POSIXct(unlist(t1$created))
                 ,n_photos = as.numeric(sapply(t1$photos, length))
                 ,n_description = as.numeric(sapply(t1$description, nchar))
                 # ,description=unlist(t1$description) # parse errors
                 # ,display_address=unlist(t1$display_address) # parse errors
                 ,latitude=unlist(t1$latitude)
                 ,longitude=unlist(t1$longitude)
                 ,listing_id=unlist(t1$listing_id)
                 ,manager_id=as.factor(unlist(t1$manager_id))
                 ,price=unlist(t1$price)
                 ,interest_level=as.factor(unlist(t1$interest_level))
                 # ,street_adress=unlist(t1$street_address) # parse errors
                 )
t2[,":="(yday=yday(created)
      ,month=month(created)
      ,mday=mday(created)
      ,wday=wday(created)
      ,hour=hour(created))]

train <- as.h2o(t2[,-"created"], destination_frame = "train.hex")

varnames <- setdiff(colnames(train), "interest_level")
gbm1 <- h2o.gbm(x = varnames
                ,y = "interest_level"
                ,training_frame = train
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 500
                ,learn_rate = 0.01
                ,max_depth = 7
                ,min_rows = 20
                ,sample_rate = 0.8
                ,col_sample_rate = 0.7
                ,stopping_rounds = 5
                ,stopping_metric = "logloss"
                ,stopping_tolerance = 0
                ,seed=321
                )

# Load data
s1 <- fromJSON("../input/test.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
s2 <- data.table(bathrooms=unlist(s1$bathrooms)
                 ,bedrooms=unlist(s1$bedrooms)
                 ,building_id=as.factor(unlist(s1$building_id))
                 ,created=as.factor(unlist(s1$created))
                 ,n_photos = as.numeric(sapply(s1$photos, length))
                 ,n_description = as.numeric(sapply(s1$description, nchar))
                 # ,description=unlist(s1$description) # parse errors
                 # ,display_address=unlist(s1$display_address) # parse errors
                 ,latitude=unlist(s1$latitude)
                 ,longitude=unlist(s1$longitude)
                 ,listing_id=unlist(s1$listing_id)
                 ,manager_id=as.factor(unlist(s1$manager_id))
                 ,price=unlist(s1$price)
                 # ,street_adress=unlist(s1$street_address) # parse errors
)
s2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]
test <- as.h2o(s2[,-"created"], destination_frame = "test.hex")

preds <- as.data.table(h2o.predict(gbm1, test))

testPreds <- data.table(listing_id = unlist(s1$listing_id), preds[,.(high, medium, low)])
fwrite(testPreds, "submission.csv")

