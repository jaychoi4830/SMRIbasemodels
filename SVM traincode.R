require('e1071')
require(caret)
require(kernlab)
library(e1071)
library(caret)
library(kernlab)
library(pROC)

##SVM Linear Kernal tuning:
set.seed(00100)
linear.tune <- tune.svm(Labels~.,
                        data = traindt1,
                        kernel = "linear",
                        cost = c(0.001,0.01,0.1,1,5,10))
summary(linear.tune)
#list of model performance based on 10-fold CV on different cost levels

best.linearsvm <- linear.tune$best.model
lineartune.test = predict(best.linearsvm, newdata=testdt)
table(lineartune.test, testdt$Labels)
##then caculate accuracy by = (TN+TP)/(TN+TP+FN+FP)

##SVM Radial Kernal tuning:
set.seed(00101)
radial.tune <- tune.svm(Labels~., data = traindt1, kernal = "radial",
                        gamma = c(0.1, 0.3, 0.5, 1, 2, 3, 4, 5),
                        cost = c(0.001, 0.01,0.1, 0.5, 1, 10, 30, 50, 70, 100))
summary(radial.tune)
best.radialsvm <- radial.tune$best.model
radialtune.test = predict(best.radialsvm, newdata = testdt)
table(radialtune.test, testdt$Labels)
## calculate accuracy

##SVM Polynomial Kernal tuning:
set.seed(00110)
poly.tune <- tune.svm(Labels~., data = traindt1,
                      kernal = "polynomial",
                      degree = c(3,4,5),
                      coef0 = c(0.1,0.5,1,2,3,4,5))
summary(poly.tune)
best.polysvm <- poly.tune$best.model
polytune.test <- predict(best.polysvm, newdata=testdt)
table(polytune.test, testdt$Labels)
## calculate accuracy
set.seed(01110)
sig.tune <- tune.svm(Labels~., data=traindt1, kernal ="sigmoid",
                     gamma = c(0.1,0.3,0.5,1,2,3,4,5),
                     coef0 = c(0.1,0.5,1,2,3,4,5))
summary(sig.tune)
best.sigsvm <- sig.tune$best.model
sigtune.test <- predict(best.sigsvm, newdata = testdt)
table(sigtune.test, testdt$Labels)
## calculate accuracy
## Confusion Matrix
confusionMatrix(sigtune.test, test$labels, positive = "1") #change test

## K-Fold Cross-validation
folds <- createFolds(traindt1$Labels, k =5)
svm.cv = lapply(folds, function(x){
  training_fold = traindt[-x,]
  test_fold = traindt[x,]
  classifier = best.radialsvm
  cv_pred = predict(classifier, newdata = test_fold[-1])
  cm = table(test_fold[, 1], cv_pred)
  accuracy = (cm[1,1] + cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return(accuracy)
})
#mean of accuracy of k-fold cv
accuracy_cv = mean(as.numeric(svm.cv))
accuracy_cv

