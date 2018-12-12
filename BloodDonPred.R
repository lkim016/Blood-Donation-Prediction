## Lori Kim
### Project 5

#setwd("C:/Users/lkim016/Desktop")

install.packages("caret")
library(class)
library(caret)
library(readxl)
library(dplyr)
library(tidyr)
library(boot)

bld.tr = read_excel("blood_traindata.xlsx")
bld.tst = read_excel("blood_testdata.xlsx")
attach(bld.tr) # attachs the dataset to R so I can just use the names
#attach(bld.tst)
#detach(bld.tr)
#detach(bld.tst)

# four predictor variables:
# "ID, Months since Last Donation, Number of Donations,
# Total Volume Donated, Months since First Donation, Made Donation this month"
# clean data (get rid of ID) / rename col for regression
colnames(bld.tr) = c("id","trMoLast","trNumDon","trTotalVal","trMoFirst","trDonMade")
colnames(bld.tst) = c("id","tstMoLast","tstNumDon","tstTotalVal","tstMoFirst","tstDonMade","tstProb")
bld.tr = bld.tr %>% select(-one_of(c("id","trTotalVal")))
bld.tst = bld.tst %>% select(-one_of(c("id","tstTotalVal")))
bld.tr$trDonMade = factor(bld.tr$trDonMade)
bld.tst$tstDonMade = factor(bld.tst$tstDonMade)

hist(bld.tr$trNumDon)
hist(log(bld.tr$trNumDon))

# look at the dimensions
#dim(bld.tr)
#dim(bld.tst)

# list types for each attribute
#sapply(bld.tr, class)
#sapply(bld.tst, class)

# take a look at the data
#head(bld.tr)
#head(bld.tst)

#* LOWER AIC AND BIC IS BETTER HIGHER R-SQUARED IS BETTER

# use logistic model to predict the probability of those people to donate their blood
logi.fit = glm(trDonMade~ . , data = bld.tr, family = "binomial")
summary(logi.fit)

logi.fit3 = glm(trDonMade~ trMoLast + trMoFirst , data = bld.tr, family = "binomial")
summary(logi.fit3)

logi.fit4 = glm(trDonMade~ trMoFirst+trNumDon , data = bld.tr, family = "binomial") # best predictor variables
summary(logi.fit4)

logi.fit5 = glm(trDonMade~ trMoLast , data = bld.tr, family = "binomial")
summary(logi.fit5)

logi.fit6 = glm(trDonMade~ trNumDon , data = bld.tr, family = "binomial")
summary(logi.fit6)

logi.fit7 = glm(trDonMade~ trMoLast + trNumDon , data = bld.tr, family = "binomial")
summary(logi.fit7)

plot(bld.tr$trNumDon, bld.tr$trDonMade, xlab = "Number of Donations", ylab = "Donations made this month")

### reference: iris file 10 Fold Cross-Validation
# We use the dataset to create a partition (80% training 20% validating)
  ## train & validation set for bld.tr
trainset = createDataPartition(bld.tr$trDonMade, p=0.80, list=FALSE)
train = bld.tr[trainset,]
valid = bld.tr[-trainset,]

#
# 10-fold Cross-Validation to train the models
#
control = trainControl(method="cv", number=10)
metric = "Accuracy"

# Linear algorithms
# Linear Discriminant Analysis (LDA)
#set.seed(7)
fit.lda <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="lda", metric=metric, trControl=control)

# Nonlinear algorithms
# Classfication and Regression Trees (CART)
#set.seed(7)
fit.cart <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="rpart", metric=metric, trControl=control)

# k-Nearest Neighbors (KNN)
#set.seed(7)
fit.knn <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="knn", metric=metric, trControl=control)

# Support Vector Machines (SVM)
#set.seed(7)
fit.svm <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="svmRadial", metric=metric, trControl=control)

# Random Forest
#set.seed(7)
fit.rf <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="rf", metric=metric, trControl=control)

# Logistic Regression
#set.seed(7)
fit.glm <- train(trDonMade~ trMoFirst+trNumDon, data=train, method="glm", metric=metric, trControl=control)

# Select Best Model
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, glm=fit.glm))
summary(results)

dotplot(results)

# Kappa Statistics (0<=k<=1)
# Adjust accuracy by accounting for the possibility of a correct prediction by chance alone.
# Very good: 0.8<k<1; good: 0.6<k<0.8; moderate: 0.4<k<0.6; fair: 0.2<k<0.4; poor: 0<k<0.2

# Summarize the Best Model
print(fit.lda)

# Make Predictions: knn / test accuracy of knn model against the validation set
predict1 = predict(fit.lda, newdata = valid)
confusionMatrix(predict1, valid$trDonMade)
table(predict1,valid$trDonMade)

# Make a prediction for bld.tst
# analyze the prediction for DonMade
predict.prob = predict(fit.lda, bld.tst, type = "prob")

bld.tst$tstProb = predict.prob[1:nrow(bld.tst),2]

predDon=rep(0, nrow(bld.tst))
predDon[predict.prob[2]>.5]="1"
bld.tst$tstDonMade = predDon[1:nrow(bld.tst)]

# analyze the prediction for DonMade with logistic regression
logi.probs=predict(logi.fit4, bld.tst,type="response")
logi.pred=rep(0, nrow(bld.tst)) # the 400 is the length of the data set
logi.pred[logi.probs>.5]="1"
bld.tst$tstDonMade = logi.pred[1:nrow(bld.tst)]
