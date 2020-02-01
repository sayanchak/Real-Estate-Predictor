# Sayan Chakraborti

library(ISLR)

# saving my data in an online google doc and importing it into my R session
myurl <- "https://docs.google.com/spreadsheets/d/e/2PACX-1vRfbdE0TtF6kWyvAEQVcRzhCf5H5A0lw7KDdjyYAaTiVVONcS357weVP-iYotmTnEnYXrJdGU3BMTqJ/pub?gid=1515730767&single=true&output=csv"
boston <- read.csv(url(myurl))
#plotting with median home value as (y), and all other variables as (x)
# add in lowess lines as well
plot(boston$crim, boston$medv, main="Crime and Home Prices", xlab = "per capita crime rate by town.", ylab = "home price")
lines(lowess(boston$crim,boston$medv), col = "red")
plot(boston$zn, boston$medv, main= "Residential Land Zoned and Home Prices", xlab = "proportion of residential land zoned for lots over 25,000 sq.ft.", ylab = "home price")
lines(lowess(boston$zn,boston$medv), col = "red")
plot(boston$indus, boston$medv, main="Industry and Home Prices", xlab = "proportion of non-retail business acres per town.", ylab = "home price")
lines(lowess(boston$indus,boston$medv), col = "red")
plot(boston$chas, boston$medv, main="Charles River Dummy and Home Prices", xlab = "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).", ylab = "home price")
lines(lowess(boston$chas,boston$medv), col = "red")
plot(boston$nox, boston$medv, main="Nitrogen Oxdes and Home Prices", xlab = "nitrogen oxides concentration (parts per 10 million).", ylab = "home price")
lines(lowess(boston$nox,boston$medv), col = "red")
plot(boston$rm, boston$medv, main="Rooms/dwelling and Home Prices", xlab = "average number of rooms per dwelling.", ylab = "home price")
lines(lowess(boston$rm,boston$medv), col = "red")
plot(boston$age, boston$medv, main="Older Houses and Home Prices", xlab = "proportion of owner-occupied units built prior to 1940.", ylab = "home price")
lines(lowess(boston$age,boston$medv), col = "red")
plot(boston$dis, boston$medv, main="Employment centres and Home Prices", xlab = "weighted mean of distances to five Boston employment centres.", ylab = "home price")
lines(lowess(boston$dis,boston$medv), col = "red")
plot(boston$rad, boston$medv, main="Highways and Home Prices", xlab = "index of accessibility to radial highways.", ylab = "home price")
lines(lowess(boston$rad,boston$medv), col = "red")
plot(boston$tax, boston$medv, main="Taxes and Home Prices", xlab = "full-value property-tax rate per $10,000.", ylab = "home price")
lines(lowess(boston$tax,boston$medv), col = "red")
plot(boston$ptratio, boston$medv, main="PTRatio and Home Prices", xlab = "pupil-teacher ratio by town.", ylab = "home price")
lines(lowess(boston$ptratio,boston$medv), col = "red")
plot(boston$black, boston$medv, main="Black and Home Prices", xlab = "proportion of blacks by town.", ylab = "home price")
lines(lowess(boston$black,boston$medv), col = "red")
plot(boston$lstat, boston$medv, main="Lower Status and Home Prices", xlab = "lower status of the population (percent).", ylab = "home price")
lines(lowess(boston$lstat,boston$medv), col = "red")
# here I bootstrap a train and test data set
v1<-sort(sample(10000,6000))
boston.train<-boston[v1,]
boston.test<-boston[-v1,]
library(regclass)
# first I try simple multiple linear regression on the data set
medv.simple_linear.train <- glm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=boston.train)
# look at summary of model and check for multicollinearity
summary(medv.simple_linear.train)
VIF(medv.simple_linear.train)
# assess the accuracy of this specific model
medv.simple_linear.predict<-predict(medv.simple_linear.train, boston.test)
plot(medv.simple_linear.predict, boston.test$medv)
cor(medv.simple_linear.predict, boston.test$medv)
# now I try a multiple polynomial regression on the data set
medv.poly_2.train <- glm(medv~I(crim^2)+I(zn^2)+I(indus^2)
    +I(chas^2)+I(nox^2)+I(rm^2)+I(age^2)+I(dis^2)+I(rad^2)
    +I(tax^2)+I(ptratio^2)+I(black^2)+I(lstat^2), data=boston.train)
# look at summary of model and check for multicollinearity
summary(medv.poly_2.train)
VIF(medv.poly_2.train)
# assess the accuracy of this specific model
medv.poly_2.predict<-predict(medv.poly_2.train, boston.test)
plot(medv.poly_2.predict, boston.test$medv)
cor(medv.poly_2.predict, boston.test$medv)


# Bayesian multiple first order linear regression
library(tidyverse)
library(rstanarm)
medv.bayes_linear<- stan_glm(
  medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat,
  data = boston.train
)
# Gives us the trace plots, a 95% prediction interval and a posterior predictive check
plot(medv.bayes_linear, plotfun = "trace")
plot(medv.bayes_linear, plotfun = "dens")
posterior_interval(medv.bayes_linear)
pp_check(medv.bayes_linear)
summary(medv.bayes_linear)



# Hierarchical Bayesian Linear Model
medv.HIERARCHICALbayes_linear <- stan_glmer(
  medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat
  + (1 | lstat),
  control = list(adapt_delta = 0.99),
  data = boston.train
)
posterior_interval(medv.HIERARCHICALbayes_linear)
pp_check(medv.HIERARCHICALbayes_linear)
summary(medv.HIERARCHICALbayes_linear)



library(tree)
library(rpart)
library(randomForest)
# useful plots to understand my decision tree steps and see if
# random forest classifier converges or not
medv.tree.train <- tree(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=boston.train)
medv.rpart.train <- rpart(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=boston.train)
medv.forest.train <- randomForest(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat, data=boston.train, na.action = na.roughfix)
par(mfrow=c(1,1)) 
plot(medv.tree.train)
text(medv.tree.train)
plot(medv.rpart.train)
text(medv.rpart.train)
plot(medv.forest.train)
medv.tree.predict<-predict(medv.tree.train, boston.test)
medv.rpart.predict<-predict(medv.rpart.train,boston.test) 
medv.forest.predict<-predict(medv.forest.train,boston.test)
# plots for the accuracy of my tree, rpart and random forest models
plot(medv.tree.predict, boston.test$medv)
plot(medv.rpart.predict, boston.test$medv)
plot(medv.forest.predict, boston.test$medv)
# get the correlations of my tree, rpart and random forest models
cor(medv.tree.predict, boston.test$medv)
cor(medv.rpart.predict, boston.test$medv)
cor(medv.forest.predict, boston.test$medv)