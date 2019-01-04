# Set the working directory.
setwd("C:/Users/Jiseong Yang/git_projects/DSC3006")
getwd()
rm(list = ls())

# install.packages("ROCR")
# install.packages("caTools")
library(ROCR)
library(caTools)

# Read in and split the dataset.
bank = read.csv("bank_dataset.csv", header = F)
train_test = sample.split(bank, SplitRatio = 0.8)
train = subset(bank, train_test == TRUE)
test = subset(bank, train_test == FALSE) 

# Modeling
bank_model<-glm(V10~., data = train, family="binomial")
summary(bank_model)

# ROC Curve
y_pred <- predict(bank_model, newdata = test, type = "response") 
prediction <- prediction(y_pred, test$V10)
bank_roc <- performance(prediction, measure = "tpr", x.measure = "fpr") #perf
plot(bank_roc, main = "ROC curve")
abline(a=0, b= 1, col="gray")
str(bank_roc)

# Get AUC
auc <- performance(prediction, measure = "auc")
auc <- auc@y.values[[1]]
auc_plot <- paste(c("AUC  = "),auc,sep="")
legend(0.5,0.4,auc_plot,cex=1,box.col = "white")

# Get cutoff value
cost.bank_roc = performance(prediction, "cost", cost.fp = 2, cost.fn = 1)
threshold<-prediction@cutoffs[[1]][which.min(cost.bank_roc@y.values[[1]])]
threshold
threshold_plot <- paste(c("threshold  = "),round(threshold,2),sep="")
legend(0.15,0.8,threshold_plot,cex=0.6,box.col = "White")
