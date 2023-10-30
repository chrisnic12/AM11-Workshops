###########################################################
##################Support Vector Machines##################
###########################################################
########################Job Attrition######################
rm(list=ls())
graphics.off()
# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome graphics
library(rsample)   # for data splittingg
library(modeldata) #package that includes couple of useful datasets

# Modeling packages
library(caret)    # for classification and regression training
library(kernlab)  # for fitting SVMs

data("attrition")
# Load attrition data
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)
head(df)

# Create training (80%) and test (20%) sets
set.seed(123)  # for reproducibility
attrition_split <- initial_split(df, prop = 0.8, strata = "Attrition")
#If we want to explicitly control the sampling so that our training and test 
#sets have similar y distributions, we can use stratified sampling
attrition_train <- training(attrition_split)
attrition_test  <- testing(attrition_split)

#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
attrition_svm <- train(
  Attrition ~ ., 
  data = attrition_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Print results
print(attrition_svm$results)
attrition_svm

#Plotting the results, we see that smaller values of the cost parameter
#( C≈ 2–8) provide better cross-validated accuracy scores for these 
#training data:
ggplot(attrition_svm) + theme_light()

#In caret and kernlab we can set different costs for missclassification, 
#this is accomplished via the class.weights argument, which is just a named 
#vector of weights for the different classes. In the employee attrition example, 
#for instance, we might specify
class.weights = c("No" = 1, "Yes" = 10)
#in the call to caret::train() to make false negatives
#(i.e., predicting “Yes” when the truth is “No”) ten times more costly than 
#false positives (i.e., predicting “No” when the truth is “Yes”). 

# Class probabilities rather than only classify:
# Control params for SVM
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  # for reproducibility
attrition_svm_auc <- train(
  Attrition ~ ., 
  data = attrition_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)

# Print results
attrition_svm_auc$results
attrition_svm_auc
#Similar to before, we see that smaller values of the cost parameter  
#C≈2−4  provide better cross-validated AUC scores on the training data
# (column Sens) refers to the proportion of Nos correctly predicted as No 
#and specificity (column Spec) refers to the proportion of Yess correctly predicted as Yes

confusionMatrix(attrition_svm_auc)

# Cross-Validated (10 fold) Confusion Matrix 
# 
# (entries are percentual average cell counts across resamples)
# 
# Reference
# Prediction   No  Yes
# No           82.4 10.9
# Yes          1.4  5.3
# 
# Accuracy (average) : 0.8768

#Model validation on the test set
#In this case it is clear that we do a 
#far better job at predicting the Nos.
test_validation = predict(attrition_svm_auc, attrition_test) 
confusionMatrix(data = test_validation, attrition_test$Attrition)

