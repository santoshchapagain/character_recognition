##Load, and if required install, the libraries
 if(!require(ggplot2))install.packages("ggplot2", dep = TRUE)
 if(!require(gridExtra))install.packages("gridExtra")
 if(!require(e1071))install.packages("e1071")
 if(!require(caret))install.packages("caret",
                                      repos = "http://cran.r-project.org", 
                                     dependencies = c("Depends", "Imports", "Suggests"))
  if(!require(kernlab))install.packages("kernlab")
  if(!require(RColorBrewer))install.packages("RColorBrewer")
  if(!require(tree))install.packages("tree")
  if(!require(rpart))install.packages("rpart")
  if(!require(maptree))install.packages("maptree")
  #if(!require(RWeka))install.packages("RWeka")
  #if(!require(FNN))install.packages("FNN")
 
##Read the training dataset
letters.labeled <- read.csv("letters_labeled.csv", header = T)

#Observe the part of the dataset
head(letters.labeled,10)[,1:15]

#check if there are any missing values in the data set
sum(is.na(letters.labeled))

#Check if there are identical rows
sum(duplicated(letters.labeled))

#Summary of the labels of the dataset
summary(letters.labeled$Letter)

#Split the data into training (70%) and test(30%) data.
set.seed(3)
ind = sample(dim(letters.labeled)[1],dim(letters.labeled)[1]*0.7)
train.letters.labeled <- letters.labeled[ind,]
test.letters.labeled <- letters.labeled[-ind,]

#Observe the Summary 
summary(train.letters.labeled[,2:150]) #Noticeable that for some columns all the pixel values are 0
summary(train.letters.labeled[,151:300]) 
summary(train.letters.labeled$Letter) #Summarize the labels of the training dataset

#Observe the distribution of the labels in the original data, the train data and test data
plot1<- ggplot(letters.labeled, aes(x = Letter, y = (..count..)/sum(..count..))) + 
        geom_bar(fill="red") + 
        theme_bw() +
        labs(y = "Relative frequency", title = "Labeled dataset") + 
        scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
        geom_text(stat = "count", aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot2<- ggplot(train.letters.labeled, aes(x = Letter, y = (..count..)/sum(..count..))) + 
        geom_bar(fill="navyblue") +
        theme_bw() +
        labs(y = "Relative frequency", title = "Train dataset") + 
        scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
        geom_text(stat = "count", aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

plot3<- ggplot(test.letters.labeled, aes(x = Letter, y = (..count..)/sum(..count..))) + 
        geom_bar(fill="purple") + 
        theme_bw() +
        labs(y = "Relative frequency", title = "Test dataset") + 
        scale_y_continuous(labels=scales::percent, limits = c(0 , 0.15)) +
        geom_text(stat = "count", aes(label = scales:: percent((..count..)/sum(..count..)), vjust = -1))

grid.arrange(plot1, plot2, plot3, nrow=3)


################MODEL BUILDING################

###TREE###
model1_rpart <- rpart(Letter~., method="class",data = train.letters.labeled)
printcp(model1_rpart)
draw.tree(model1_rpart, cex=0.5, nodeinfo = T, col = gray(0:8/8))
pred.rpart <- predict(model1_rpart, newdata=test.letters.labeled, type="class")
table(`Actual`=test.letters.labeled$Letter, `Predicted`=pred.rpart)
error.rate.rpart <- sum(test.letters.labeled$Letter!=pred.rpart)/nrow(test.letters.labeled)
cat("Accuracy for Tree model is", 1-error.rate.rpart)
#Accuracy obtained is 31% only


##SVM with Linear Kernel
model1_linear <- ksvm(Letter~., 
                      data=train.letters.labeled,
                      kernel="vanilladot",
                      scaled=F,
                      C=0.1)

summary(model1_linear)

pred.model1.linear <- predict(model1_linear, newdata = test.letters.labeled, type="response")
cm1 <- confusionMatrix(pred.model1.linear, test.letters.labeled$Letter)
print(cm1$overall)
print(cm1)

#Print Confusion Matrix in GGPLOT
cmf <- as.data.frame(cm1$table)
cm.gg <- ggplot(cmf)+
          geom_tile(aes(x=Prediction,y=Reference, fill=Freq))+
          scale_x_discrete(name="Reference (Actual)")+
          scale_y_discrete(name="Predicted", limits=rev(levels(cmf$Prediction)))+
          geom_text(aes(x=Prediction,y=Reference,label=Freq),color="white",size=4)
print(cm.gg)
#Overall Accuracy obtained is 63.33%

##SVM with Radial Kernel
model1_radial <- ksvm(Letter~.,
                      data=train.letters.labeled,
                      scaled=F,
                      kernel="rbfdot",
                      C=1,
                      kpar="automatic")

print(model1_radial)

pred.model1.radial <- predict(model1_radial, newdata = test.letters.labeled, type="response")
cm2 <- confusionMatrix(pred.model1.radial, test.letters.labeled$Letter)
print(cm2$overall)
#Overall Accuracy obtained is 61.33%


########KNN#########
model2_knn <- IBk(Letter~., data=train.letters.labeled)
summary(model2_knn)
pred.knn <- predict(model2_knn, newdata=test.letters.labeled, type="class")
table(`Actual`=test.letters.labeled$Letter, `Predicted`=pred.knn)
error.rate.knn <- sum(test.letters.labeled$Letter!=pred.knn)/nrow(test.letters.labeled)
Accuracy.knn <- 1-error.rate.knn
Accuracy.knn
##Accuracy from KNN is 49.66%


