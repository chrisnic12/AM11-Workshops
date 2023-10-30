##############bank direct marketing example#################
rm(list=ls())
graphics.off()
# Helper packages
library(dplyr)    # for data wrangling. A must for a data scientist :)
library(ggplot2)  # for awesome graphics
library(rsample)  # for efficient data splitting

# Modeling packages
library(caret)    # for classification and regression training 
library(kernlab)  # for fitting SVMs
library(e1071)    # for fitting SVMss


################load the data###############
ds_bank <- read.csv("/Users/christosnicolaides/Dropbox (Personal)/LBS/Teaching January 2023/SVM/SVM Workshop 01102023/SVM Workshop Material 01102023/bank_small.csv",
                    stringsAsFactors = TRUE) #******* you need to covert all strings to factors!

# Citation Request:
# This dataset is public available for research. The details are described 
# in [Moro et al., 2011]. Please include this citation if you plan to use this database:
#   
# [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank 
# Direct Marketing: An Application of the CRISP-DM Methodology. 
# In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling 
# Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
# 
# Available at: [pdf] http://hdl.handle.net/1822/14838
# [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt
# 
# 1. Title: Bank Marketing
# 
# 2. Sources
# Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012
# 
# 3. Past Usage:
# 
# The full dataset was described and analyzed in:
# 
# S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
# In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, 
# Portugal, October, 2011. EUROSIS.
# 
# 4. Relevant Information:
#   
# The data is related with direct marketing campaigns of a Portuguese banking institution. 
# The marketing campaigns were based on phone calls. Often, more than one contact to the 
# same client was required, in order to access if the product (bank term deposit) 
# would be (or not) subscribed. 
# 
# There are two datasets: 
# 1) bank-full.csv with all examples, ordered by date (from May 2008 to November 2010).
# 2) bank.csv with 10% of the examples (4521), randomly selected from bank-full.csv.
# The smallest dataset is provided to test more computationally demanding machine 
# learning algorithms (e.g. SVM).
# 
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# 5. Number of Instances: 45211 for bank-full.csv (4521 for bank.csv)
# 
# 6. Number of Attributes: 16 + output attribute (y).
# 
# 7. Attribute information:
#   
#   For more information, read [Moro et al., 2011].
# 
# Input variables:
# # bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid",
# "entrepreneur","student","blue-collar","self-employed","retired","technician","services") 
# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
# 4 - education (categorical: "unknown","secondary","primary","tertiary")
# 5 - default: has credit in default? (binary: "yes","no")
# 6 - balance: average yearly balance, in euros (numeric) 
# 7 - housing: has housing loan? (binary: "yes","no")
# 8 - loan: has personal loan? (binary: "yes","no")
#
# # related with the last contact of the current campaign:
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
# 10 - day: last contact day of the month (numeric)
# 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 12 - duration: last contact duration, in seconds (numeric)
#
# # other attributes:
# 13 - campaign: number of contacts performed during this campaign and for this client
# (numeric, includes last contact)
# 14 - pdays: number of days that passed by after the client was last contacted from 
# a previous campaign (numeric, -1 means client was not previously contacted)
# 15 - previous: number of contacts performed before this campaign and for this client (numeric)
# 16 - poutcome: outcome of the previous marketing campaign
# (categorical: "unknown","other","failure","success")
# 
# Output variable (desired target):
# 17 - y - has the client subscribed a term deposit? (binary: "yes","no")

nrow(ds_bank)
summary(ds_bank)

#split our dataset into training and test set using the outcome variable as strata
# Create training (80%) and test (20%) sets
set.seed(123)  # for reproducibility
bank_split <- initial_split(ds_bank, prop = 0.8, strata = "y")
bank_train <- training(bank_split)
bank_test  <- testing(bank_split)


#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation#
#set.seed(123)  # for reproducibility
bank_svm_tune <- train(
  y ~ ., 
  data = bank_train,
  method = "svmRadial", # Radial kernel      
  preProcess = c("center", "scale"),  # center & scale the data
  trControl = trainControl(method = "cv", number = 10), #cross-validation (10-fold) 
  tuneLength = 10 #use 10 default values for the main parameter
)

#dispay and plot results of cross validation
ggplot(bank_svm_tune) + theme_light()
print(bank_svm_tune)
bank_svm_tune
#*note:  The column labeled “Accuracy” is the overall agreement 
#rate averaged over cross-validation iterations
#it looks like sigma = 0.01758396 and C = 4 give the best accuracy


#let's re-define the grid and make a grid-search
set.seed(1292)
# Use the expand.grid to specify the search space	
grid <- expand.grid(sigma = c(0.016, 0.017, 0.018, 0.019, 0.020),
                    C = c(1,2,3,4,5,6,7,8,9,10))
bank_svm_tune <- train(
  y ~ ., 
  data = bank_train,
  method = "svmRadial",         # Radial kernel      
  preProcess = c("center", "scale"),  # center & scale the data
  trControl = trainControl(method = "cv", number = 10), #cross-validation (10-fold) 
  tuneGrid = grid
)

ggplot(bank_svm_tune) + theme_light()
print(bank_svm_tune)
#it looks like C=7, \sigma = 0.016 give the best accuracy
#dispay and plot results of cross validation

confusionMatrix(bank_svm_tune)


#Model validation on the test set
test_validation = predict(bank_svm_tune, bank_test) 
confusionMatrix(data = test_validation, as.factor(bank_test$y))



########Cross validation and model with e1071######
set.seed(1400)

fit.tune <- tune.svm(y ~ ., data = bank_train, gamma = 10^(-5:-1), cost = 10^(-3:1))
summary(fit.tune)
plot(fit.tune)

#it looks like \gamma=0.01-0.1 and C=1-10 does better job
fit.tune <- tune.svm(y ~ ., data = bank_train, gamma = seq(0.01,0.12,0.01), cost = (1:10))
summary(fit.tune)
fit.tune
#gamma=0.11 / C = 2
# model specification
fit_e1071 <- svm(y ~ ., data = bank_train, cost=2, gamma = 0.11)
test_validation = predict(fit_e1071, bank_test) 
confusionMatrix(data = test_validation, bank_test$y)

##########Model in kernlab######
rbf <- rbfdot(sigma=0.11)
fit_kl <- ksvm(y ~ . , data = bank_train, type="C-svc", kernel = rbf, C = 2)
test_validation = predict(fit_kl, bank_test) 
confusionMatrix(data = test_validation, bank_test$y)