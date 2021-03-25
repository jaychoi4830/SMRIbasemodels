library(caret)
##do not split data into test and train for caret
set.seed(10000)
train.control <- trainControl(method = "cv", number = 10)
#10 fold CV
glm_model <- train(Labels ~.,
                   data=traindt[-c(1)],
                   trControl = train.control,
                   method = "glm",
                   family = binomial()
                   )
#glm_model created
summary(glm_model)
glm_model$results

