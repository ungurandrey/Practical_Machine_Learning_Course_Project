#install.packages('caret', dependencies = TRUE)
library(caret)
#library(ggplot2)
library(lattice)
library(gridExtra)
#library(randomForest)
library(rattle)
library(rpart)
library(tidyr)
#Download files from the data source, and read the two csv files into two data frames.  
TrainDataUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
TrainFile <- "./data/pml-training.csv"
TestFile  <- "./data/pml-testing.csv"

if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(TrainFile)) {
  download.file(TrainDataUrl, destfile=TrainFile, method="curl")
}
if (!file.exists(TestFile)) {
  download.file(TestDataUrl, destfile=TestFile, method="curl")
}
#Make sure we will get NA values for '', 'NA', '#Div/0!'
TrainData <- read.csv("./data/pml-training.csv",header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c("NA","#DIV/0!",""))
ValidationData <- read.csv("./data/pml-testing.csv",header=TRUE, as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c("NA","#DIV/0!",""))
#After investigating all the variables of the sets, we see that there are a lot of NA values.Remove the variables that contains missing values.
#Remove the first seven variables as they have little impact on the outcome classe
TrainDataClean <- TrainData[,colSums(is.na(TrainData))==0]
TrainDataClean <- TrainDataClean[,-c(1:7)]
dim(TrainDataClean)
ValidationDataClean <- ValidationData[,colSums(is.na(ValidationData))==0]
ValidationDataClean <- ValidationDataClean[,-c(1:7)]
dim(ValidationDataClean)
#Preparing the datasets for prediction Preparing the data for prediction by splitting the training data into 70% as train data and 30% as test data. 
set.seed(127) 
inTrain <- createDataPartition(TrainDataClean$classe, p = 0.7, list = FALSE)
TrainDataClean <- TrainDataClean[inTrain, ]
TestData <- TrainDataClean[-inTrain, ]
dim(TrainDataClean)
dim(TestData)

#Prediction with Decision Trees
decisonTree <- rpart(classe ~ ., data=TrainDataClean, method="class")
fancyRpartPlot(decisonTree)

predictionsDecTree <- predict(decisonTree, TestData, type = "class")
cmtree <- confusionMatrix(table(predictionsDecTree, TestData$classe))
cmtree
cmtree$table
cmtree$overall[1]

# Train with random forests Random Search
trControl <- trainControl(method = "cv", number = 3)
model_RF <- train(classe~., data=TrainDataClean, method="rf", trControl=trControl, verbose=FALSE)
print(model_RF)

plot(model_RF,main="Accuracy of Random forest model by number of predictors")
trainpred <- predict(model_RF,newdata=TestData)

confMatRF <- confusionMatrix(table(trainpred,TestData$classe))

# display confusion matrix and model accuracy
confMatRF$table
confMatRF$overall[1]
names(model_RF$finalModel)
model_RF$finalModel$classes
plot(model_RF$finalModel,main="Model error of Random forest model by number of trees")

model_GBM <- train(classe~., data=TrainDataClean, method="gbm", trControl=trControl, verbose=FALSE)
print(model_GBM)