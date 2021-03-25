library(xgboost)
library(data.table)
library(mlr)
library(dplyr)
library(caret)
dt1 <- study17_top20
sample <- sample.int(n = nrow(dt1), size = floor(.8*nrow(dt1)), replace = F)
traindt <- dt1[sample,]
traindt1 <- traindt
traindt1 <- traindt1 %>% mutate_each(funs(as.factor), c(Labels))
## or
as.factor(traindt1$Labels)
testdt <- dt1[-sample,]
testdt1 <- testdt
testdt1 <- testdt1 %>% mutate_each(funs(as.factor), c(Labels))
## or
as.factor(testdt1$Labels)

dtrain <- xgb.DMatrix(as.matrix(sapply(traindt[-c(2)], as.numeric)), label = traindt$Labels)
dtest <- xgb.DMatrix(as.matrix(sapply(testdt[-c(2)], as.numeric)), label=testdt$Labels)

watchlist <- list(train = dtrain, test=dtest)

## basic model
xgb_model_1 <- xgb.train(data = dtrain,
                         nrounds = 3,
                         booster = "gbtree",
                         objective = "binary:logistic",
                         eta=0.1, gamma=0, lambda=0.5, max_depth=6, min_child_weight=1,
                         subsample=1, colsample_bytree=1,
                         watchlist = watchlist,
                         maximize = F, eval_metric = list("error","auc"))
pred1 <- predict(xgb_model_1, dtest)
pred1 <- ifelse(pred1 < 0.5,1,0)
cm <- confusionMatrix(as.factor(pred1), as.factor(testdt$Labels), positive = "1")
##hyperparameter tuning
library(mlr)
traintask <- makeClassifTask(data = traindt1[-c(1)], target = "Labels", positive = "1")
testtask <- makeClassifTask(data = testdt1[-c(1)], target = "Labels")
xgb_learner <- makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "error",
    nrounds = 100,
    eta = 0.1
  )
)
xgb_model_learner <- train(xgb_learner, task = traintask)
#set hyperparams to tune
xgb_params <- makeParamSet(
  makeDiscreteParam("booster", values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth", lower=3L, upper=10L),
  makeNumericParam("lambda", lower = 0.0001, upper = 10),
  makeNumericParam("alpha", lower = 0.0001, upper = 10),
  makeNumericParam("gamma", lower = 0, upper = 10),
  makeNumericParam("colsample_bytree", lower = .1, upper = 1),
  makeNumericParam("min_child_weight", lower = 1L, upper = 10L),
  makeNumericParam("subsample", lower = 0.5, upper =1)
)
control <- makeTuneControlRandom(maxit=50L)
resample_desc <- makeResampleDesc("CV", stratify = T, iters = 5L)
##hyperparameter tuning - performs random grid search
tuned_params <- tuneParams(
  learner = xgb_learner,
  task = traintask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control,
  measures = acc,
  show.info = T
)
tuned_params
##using tuned params for model
params <- list(tuned_params$besttunedparams)
xgb_model_2 <- xgb.train(data = dtrain,
                         nrounds = 5,
                         objective = "binary:logistic",
                         params=tuned_params,
                         watchlist = watchlist,
                         eta = 0.1,
                         print_every_n = 10, early_stopping_rounds = 10,
                         maximize = F, eval_metric = list("error","auc"))
##tune iteration rounds
xgbcv <- xgb.cv(params = tuned_params,
                data = dtrain,
                nrounds = 100,
                nfold = 5,
                showsd = T,
                stratified = T,
                print_every_n = 10, early_stopping_rounds = 20, maximize = F,
                eval_metric = "auc")
##best iteration rounds =
##then, feed the rounds back into xgb_model_2
# 5. Feature importance
importance_matrix <- xgb.importance(model = xgb_model_1)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)
