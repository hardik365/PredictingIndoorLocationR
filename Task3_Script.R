# Required
library(doParallel)
library(caret)
library(readr)
library(C50)
library(forcats)
library(gbm)
library(dbplyr)
library(tidyverse)
library(e1071)
set.seed(123)



# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6
# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(24)
# Register Cluster
registerDoParallel(cl)
# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 24 

#Reading the file and combining the columns
trainingData <-read.csv("trainingData.csv")
trainingData <-cbind(trainingData, paste(sep="_",trainingData$BUILDINGID, trainingData$FLOOR, trainingData$SPACEID, trainingData$RELATIVEPOSITION))
colnames(trainingData)[537] <- "LOCO"

validationData <-read.csv("validationData2.csv")
validationData <-cbind(validationData, paste(sep="_",validationData$BUILDINGID, validationData$FLOOR, validationData$SPACEID, validationData$RELATIVEPOSITION))
colnames(validationData)[525] <- "LOCO"

write.csv(trainingData, file = "trainingDataEdit.csv" , row.names=FALSE) 
write.csv(validationData, file = "validationDataEdit.csv" , row.names=FALSE) 


##I physically moved the place of loco to the front
trainingData <-read.csv("trainingDataEdit.csv")

trainingData$FLOOR<-as.factor(trainingData$FLOOR)
trainingData$BUILDINGID<-as.factor(trainingData$BUILDINGID)
trainingData$SPACEID<-as.factor(trainingData$SPACEID)
trainingData$RELATIVEPOSITION<-as.factor(trainingData$RELATIVEPOSITION)
#trainingData$LOCO<-as.factor(trainingData$LOCO)

#save our work so far in a csv file
write.csv(trainingData, file = "trainingDataEdit.csv" , row.names=FALSE) 
trainingData <-read.csv("trainingDataEdit.csv")

str(trainingData)
attributes(trainingData)


trainingDataTemp <- trainingData[, -c(1,2,3,5,6,8,9,10)]
ValidData <- validationData[, -c(1,2,3,4)]

#trainingDataTemp$LOCO<-as.character(trainingDataTemp$LOCO)


#with one building
oneBuildingData <- filter(trainingDataTemp, BUILDINGID == "0")
oneBuildingData <- oneBuildingData[, -c(1)]

oneBuildingData$LOCO<-as.factor(oneBuildingData$LOCO)
ValidData$LOCO<-as.factor(ValidData$LOCO)


#partition our data for one building data
inTrainingBD <- createDataPartition(oneBuildingData$LOCO, p = .75, list = FALSE)
trainingBD <- oneBuildingData[inTrainingBD,]
testingBD <- oneBuildingData[-inTrainingBD,]


##lets setup and try random forest
rfitControlRF <- trainControl(method = "repeatedcv", number = 10, repeats = 1, search = 'random')
#train Random Forest Regression model
rfFit <- train(LOCO~.,
                data = trainingBD, 
                method = "rf",
                trControl = rfitControlRF)
rfFit   
#mtry  Accuracy   Kappa    
#178   0.7629240  0.7618935
#273   0.7573840  0.7563317
#376   0.7556664  0.7546020
plot((rfFit), main = "Random Forest Accuracy")



testRFBD <- predict(rfFit, testingBD)
resampleRF <- postResample(testRFBD, testingBD$LOCO)
resampleRF
#Accuracy     Kappa 
#0.7597765 0.7587959 



##Let's try with c5
x <-  trainingBD[, -c(1)]
y <- trainingBD$LOCO

fitControlC5 <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10, returnResamp="all")

gridC5 <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )

c5Model<- train(x=x,y=y,tuneGrid=gridC5,trControl=fitControlC5,method="C5.0",verbose=FALSE)
c5Model
#winnow  trials  Accuracy   Kappa    
#FALSE    1      0.5800102  0.5782250
#FALSE    5      0.6600387  0.6585807
#FALSE   10      0.6887959  0.6874598
#FALSE   15      0.6995935  0.6983034
#FALSE   20      0.7056465  0.7043807
#TRUE    1      0.5807620  0.5789819
#TRUE    5      0.6623214  0.6608739
#TRUE   10      0.6902914  0.6889628
#TRUE   15      0.6997948  0.6985052
#TRUE   20      0.7046703  0.7034019

plot((c5Model), main = "C5 Accuracy")

testc5BD <- predict(c5Model, testingBD)
resamplec5 <- postResample(testc5BD, testingBD$LOCO)
resamplec5
#Accuracy     Kappa 
#0.6959298 0.6946928 

NBmodel = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
NBmodel
#usekernel  Accuracy   Kappa    
#FALSE            NaN        NaN
#TRUE      0.1737173  0.1688376

plot((NBmodel), main = "Naive Bayes Accuracy")

testNBD <- predict(NBmodel, testingBD)
resampleNB<-postResample(testNBD, testingBD$LOCO)
resampleNB
#Accuracy     Kappa 
#0.1835595 0.1786828 

resampleData <- data.frame(Model = c("Random Forest", "C5", "Naives Bayes"),
                        Kappa = c(0.7587959,0.6946928,0.1786828),
                        Accuracy = c(0.7597765, 0.6959298,0.1835595 ))

AccPlot<-ggplot(data=resampleData, aes(x=Model, y=Accuracy)) +
  geom_bar(stat="identity", fill="steelblue")+
  ggtitle("Resampled Accuracy")

AccPlot

kappaPlot<-ggplot(data=resampleData, aes(x=Model, y=Kappa)) +
  geom_bar(stat="identity", fill="red")+
  ggtitle("Resampled Kappa")
  
kappaPlot




summary(Modelresamples)




# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)



unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
