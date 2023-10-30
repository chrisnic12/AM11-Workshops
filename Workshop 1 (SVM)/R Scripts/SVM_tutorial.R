#####################################################################
##################Support Vector Machines Tutorial###################
#####################################################################
rm(list=ls())
#graphics.off()
  setwd("/Users/christosnicolaides/Dropbox (Personal)/LBS/Teaching January 2023/SVM/SVM Workshop 01102023/")
#We will use "e1071" and "kernlab" packages
library(e1071)
library(ggplot2) #for visualizations

#####################################################################
################ Linear Separable Data ##############################

#first apply to a small dataset for better visual illustrations
set.seed(10111)
#create the data!
x = matrix(rnorm(40), 20, 2) #random random points (x1, x2)
y = rep(c(-1, 1), c(10, 10)) #create the y vector
x[y == 1,] = x[y == 1,] + 1  #try somehow to separate the two classes
plot(x, col = y + 3, pch = 19)


#put the data in the right form
dat = data.frame(x, y = as.factor(y)) #get the data into a dataframe
#svmfit from e1071 package for more info: 
#https://www.rdocumentation.org/packages/NormalizeMets/versions/0.25/topics/SvmFit

#make a call to svm on this dataframe, using y as the response variable 
#and other variables as the predictors. You tell SVM that the kernel is linear, 
#the tune-in parameter cost is 10, and scale equals false. In this example, 
#you ask it not to standardize the variables.
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit) #print info about the model

# plot function for SVM that shows the decision boundary
plot(svmfit, dat)
#hmmm not flexible visualization tool

##create a better visualizer####
#split the x space into 100 x 100 lattice frid
make.grid = function(x, n = 100) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}
xgrid = make.grid(x)
head(xgrid)

#for each dot at the 100 x 100 grid gets the prediction of the model 
ygrid = predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y + 3, pch = 19)
#plots and ephasize the support vectors
points(x[svmfit$index,], pch = 5, cex = 2)

# the svm function is not too friendly, in that you have to do some work to 
# get back the linear coefficients. The reason is probably that this only 
# makes sense for linear kernels, and the function is more general. So let's 
# use a formula to extract the coefficients more efficiently. You extract beta 
# and beta0, which are the linear coefficients of the hyperplane: b0+b1x1+b2x2=0
beta = drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 = svmfit$rho

#####Make a try to plot the margin!
#plot the 100 x 100 grid and color the points according to the predictions
plot(xgrid, col = c("red", "blue")[as.numeric(ygrid)], pch = 20, cex = .2)
#plot the data
points(x, col = y + 3, pch = 19)
#Emphsize support vectors
points(x[svmfit$index,], pch = 5, cex = 2)
#plot the hyperplne
abline(beta0 / beta[2], -beta[1] / beta[2])
#plot the upper margin
abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
#plot the lower margin
abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)


################Non Linear-Separable data################
rm(list=ls())
#example from the textbook Elements of Statistical Learning, which has a canonical
#example in 2 dimensions where the decision boundary is non-linear -- data in "ESL.mixture.rda"
load(file = "ESL.mixture.rda")
plot(ESL.mixture$x, col = ESL.mixture$y + 1)
# data seems to overlap quite a bit, but you can see that there's something special
#in its structure

#Now, let's make a data frame with the response y, and turn that into a factor. 
# After that, you can fit an SVM with radial kernel and cost as 5.
dat = data.frame(y = factor(ESL.mixture$y), ESL.mixture$x)
fit = svm(factor(y) ~ ., data = dat, scale = TRUE, kernel = "radial", cost = 10)
print(fit) 

# It's time to create a grid and make your predictions. These data actually came supplied 
# with grid points. If you look down on the summary on the names that were on the list, there 
# are 2 variables px1 and px2, which are the grid of values for each of those variables. You 
# can use expand.grid to create the grid of values. Then you predict the classification at 
# each of the values on the grid.
xgrid = expand.grid(X1 = ESL.mixture$px1, X2 = ESL.mixture$px2)
ygrid = predict(fit, xgrid) #predict the value of y at each grid point using the model fit
plot(xgrid, col = as.numeric(ygrid), pch = 25, cex = .5)
points(ESL.mixture$x, col = ESL.mixture$y + 1, pch = 19)

#get the real values of the model rather than classification
func = predict(fit, xgrid, decision.values = TRUE) 
func = attributes(func)$decision
contour(ESL.mixture$px1, ESL.mixture$px2, matrix(func, 69, 99), level = 0, add = TRUE)
#note that px1, px2 are dimensions 69 and 99 respectively.


#########Another Case for Non Linear-Separable data################
rm(list=ls())
#construct the data
x <- matrix(rnorm(200*2), ncol = 2)
x[1:100,] <- x[1:100,] + 2.5
x[101:150,] <- x[101:150,] - 2.5
y <- c(rep(1,150), rep(2,50))
dat <- data.frame(x,y=as.factor(y))

# Plot data
ggplot(data = dat, aes(x = X2, y = X1, color = y, shape = y)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme(legend.position = "none")

# set pseudorandom number generator for reproducibility
set.seed(123)
svmfit <- svm(y~., data = dat, kernel = "radial", gamma = 1, cost = 1)
#radial basis function!!!

# plot classifier
plot(svmfit, dat)

#make 100x100 grid for  better visualization
make.grid = function(x, n = 100) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}
xgrid = make.grid(x)
#for each dot at the 100 x 100 grid gets the prediction of the model 
ygrid = predict(svmfit, xgrid)
plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)
points(x, col = y, pch = 19)
#plots and ephasize the support vectors
points(x[svmfit$index,], pch = 5, cex = 2)
