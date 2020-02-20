##Clear the R environment, if required
rm(list = ls())


#####################################################################################
#####################Install the libraries, if required##############################
#####################################################################################

if(!require(ggplot2))install.packages("ggplot2")
if(!require(gridExtra))install.packages("gridExtra")
if(!require(ggfortify))install.packages("ggfortify")
if(!require(caret))install.packages("caret")
if(!require(plyr))install.packages("plyr")


###########################################################################################
###################Read the datasets and split into training and test sets#################
###########################################################################################

##Read the labeled dataset
labeledletters<-read.csv("letters_labeled.csv",
                         header=TRUE,
                         sep=",")

#Omit the rows with missing values
labeledletters <- na.omit(labeledletters)

#Convert the letter into lower cases if it is otherwise, for the sake of consistency
labeledletters$Letter <- tolower(labeledletters$Letter)

#Convert the class of the lable to factor
labeledletters$Letter <- as.factor(labeledletters$Letter)

###Splitting from train
set.seed(123)

##create a sample size
smp_size <- floor(0.70 * nrow(labeledletters))

#Create the train indices
train_ind <- sample(seq_len(nrow(labeledletters)), 
                    size = smp_size)

#Create the train and test datasets
train <- (labeledletters)[train_ind, ]
test <- (labeledletters)[-train_ind, ]

#Observe the distribution of the labels in the original data, the train data and test data
plot1<- ggplot(labeledletters, 
               aes(x = Letter, 
                   y = (..count..)/sum(..count..))) + 
  geom_bar(fill="red") + 
  theme_bw() +
  labs(y = "Relative frequency", 
       title = "Unlabeled dataset") + 
  scale_y_continuous(labels=scales::percent, 
                     limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), 
                vjust = -1), 
            size=3)+
  theme(axis.text=element_text(size=12),
        axis.title = element_text(size=12, face="bold"),
        plot.title = element_text(hjust = 0.5, size = 14, face="bold"))

plot2<- ggplot(train, 
               aes(x = Letter, 
                   y = (..count..)/sum(..count..))) + 
  geom_bar(fill="navyblue") +
  theme_bw() +
  labs(y = "Relative frequency", 
       title = "Train dataset for unlabeled data") + 
  scale_y_continuous(labels=scales::percent, 
                     limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), 
                vjust = -1), 
            size=3)+
  theme(axis.text=element_text(size=12),
        axis.title = element_text(size=12, face="bold"),
        plot.title = element_text(hjust = 0.5, size = 14, face="bold"))

plot3<- ggplot(test, 
               aes(x = Letter, 
                   y = (..count..)/sum(..count..))) + 
  geom_bar(fill="purple") + 
  theme_bw() +
  labs(y = "Relative frequency", 
       title = "Test dataset for unlabeled data") + 
  scale_y_continuous(labels=scales::percent, 
                     limits = c(0 , 0.15)) +
  geom_text(stat = "count", 
            aes(label = scales:: percent((..count..)/sum(..count..)), 
                vjust = -1), size=3)+
  theme(axis.text=element_text(size=12),
        axis.title = element_text(size=12, face="bold"),
        plot.title = element_text(hjust = 0.5, size = 14, face="bold"))


#Plot the above 3 plots together
grid.arrange(plot1, plot2, plot3, nrow=3)


####################################################################################
##############################Dimension Reduction::PCA##############################
####################################################################################


valdata <- labeledletters
system.time(
  val_PCA_matrix <- as.matrix(valdata) %*% pcaX$rotation[,1:80]
)
#Convert the matrix to a dataframe
val_PCA <- data.frame(val_PCA_matrix)
final_prediction <- predict(svm_Linear, newdata = val_PCA)
final_prediction
write.csv(final_prediction,"C:\\Data\\Updated_final_predictions.csv")


# conduct PCA on training dataset
X <- train[,-1]
Y <- train[,1]

#Reducing Train using PCA
train_reduced <- X
pcaX <- prcomp(train_reduced)

#Plot the first two Principal Components, and see if they make any sense
##Library ggfortify does the job. Plots the first two PCs
autoplot(pcaX, data=train, colour='Letter', title="Principal component Score Plot")  

# Creating a datatable to store and plot the No of Principal Components vs Cumulative Variance Explained
vexplained <- as.data.frame(pcaX$sdev^2/sum(pcaX$sdev^2))
vexplained <- cbind(c(1:3136),vexplained,cumsum(vexplained[,1]))
colnames(vexplained) <- c("No_of_Principal_Components",
                          "Individual_Variance_Explained",
                          "Cumulative_Variance_Explained")

#Plotting the curve using the datatable obtained
plot(vexplained$No_of_Principal_Components,
     vexplained$Cumulative_Variance_Explained*100, 
     xlim = c(0,3000),type='b',pch=16,
     xlab = "Principal Components",
     ylab = "Cumulative Variance Explained (%)",
     main = 'Principal Components vs Cumulative Variance Explained')
abline(h=vexplained$Cumulative_Variance_Explained[80]*100, 
       v = vexplained$No_of_Principal_Components[80],
       col='red',lwd=2,lty=5)


#define where to stop plotting PCs
cut <- 2000
#Plot the Cum. variance explained by the 'cut' number of PCs
plot(vexplained$No_of_Principal_Components[1:cut],
     vexplained$Cumulative_Variance_Explained[1:cut], 
     xlim = c(0,cut),type='b',pch=16,
     xlab = "Principal Componets",
     ylab = "Cumulative Variance Explained",
     main = 'Principal Components vs Cumulative Variance Explained')
abline(h=vexplained$Cumulative_Variance_Explained[80], 
       v = vexplained$No_of_Principal_Components[80],
       col='red',lwd=2,lty=5)

#Datatable to store the summary of the datatable obtained
vexplainedsummary <- vexplained[seq(0,100,10),]
vexplainedsummary <- vexplainedsummary[-2] #Drop the second column

##Create a PCA matrix for the train data
train_PCA_matrix <- as.matrix(train_reduced) %*% pcaX$rotation[,1:80]
##convert the matrix to dataframe
train_PCA <- data.frame(train_PCA_matrix)
##Append the classes to the trainPCA data
train_PCA$Letter <- train[,1]

#Remove the classes from the test data
test_reduced <- test[,-1]
#Create a PCA matrix for the test data
test_PCA_matrix <- as.matrix(test_reduced) %*% pcaX$rotation[,1:80]
#Convert the matrix to a dataframe
test_PCA <- data.frame(test_PCA_matrix)
#Append the classes to the testPCA data
test_PCA$Letter <- test[,1]


#####################################################################################################
##################################### MODEL DEVELOPMENT #############################################
#####################################################################################################


#####################################################################################################
##################################### MODEL 1: SVM with Linear Kernel################################
#####################################################################################################


## Support Vector Machines with Linear Basis Function Kernel
#Let's call the number of classes N. The samples of your data-set is called M.
###NC35
# Say you have 35 classes. The OVO ensemble will be composed of 595 (= 35 * (34) / 2) binary classifiers. 
#The first will discriminante A from B, the second A from C, and the third B from C.. etc. 
#Now, if x is to be classified, x is presented to each binary classifier of the ensemble to create a vector of individual classifications, e.g. (A, B, B). 
#The final step to assign a label to x is the majority voting. In this example, x would belong to class B.


#The cost was optimized to 0.05. Optimization is performed below.
Linear_grid <- expand.grid(C = c(0.05))

#10-fold Cross-validation
trctrl <- trainControl(method = "cv", number = 10)

#SVM with linear kernel
svm_Linear <- train(Letter ~., 
                    data = train_PCA, 
                    method = "svmLinear",
                    trControl=trctrl,
                    tuneGrid = Linear_grid)


#SVM with linear kernel
svm_Linear <- train(Letter ~., 
                    data = train, 
                    method = "svmLinear",
                    trControl=trctrl,
                    tuneGrid = Linear_grid,
                    preProcess=c('pca','center','scale'))


#Linear grid to optimize the Cost
# Linear_grid_test <- expand.grid(C = c(0.01,0.025,0.05,0.75,1))



#Optimize the SVM with Linear Kernel
# svm_Linear_test <- train(Letter ~., 
#                          data = train_PCA, 
#                          method = "svmLinear",
#                          trControl=trctrl,
#                          tuneGrid = Linear_grid_test)


#plot(svm_Linear_test)


#Predict the test data using SVM model with Linear Kernel
test_pred_Linear <- predict(svm_Linear, newdata = test_PCA)

#Build a Confusion Matrix for the test result
cf_matrix_linear <- confusionMatrix(test_pred_Linear,test_PCA$Letter)

#Build a Confusion Matrix for the Validation result
cf_matrix_linear2 <- confusionMatrix(test_pred_Linear2,validationset_PCA$Letter)


#Function to Plot Confusion Matrix in GGPLOT
plotConfusionMatrix <- function(cf_matrix){
  cmf <- as.data.frame(cf_matrix$table)
  colnames(cmf)<- c("Reference","Prediction","Freq")
  cm.gg <- ggplot(cmf)+
    geom_tile(aes(x=Prediction,y=Reference, fill=Freq))+
    scale_x_discrete(name="Reference (Actual)", position="top")+
    scale_y_discrete(name="Predicted", limits=rev(levels(cmf$Prediction)))+
    geom_text(aes(x=Prediction,y=Reference,label=Freq),color="white",size=4.5)
  print(cm.gg)+
    ggtitle("Confusion Matrix")+
    theme(plot.title = element_text(size=18, face="bold", hjust = 0.5),
          axis.text = element_text(size=16),
          axis.title = element_text(size=16))
}

#Plot the confusion matrix for test data set predicted with SVM Linear Kernel
plotConfusionMatrix(cf_matrix_linear)
###Overall accuracy obtained is ________%

##Overall Accuracy for the test dataset
cf_matrix_linear$overall


#Plot the confusion matrix for Validation data set predicted with SVM Linear Kernel
plotConfusionMatrix(cf_matrix_linear2)
###Overall accuracy obtained is ________%

##Overall Accuracy for the validation dataset
cf_matrix_linear2$overall


#Function to find out the class based accuracy
class_accuracy <- function(cf_object){
  class_acc<- vector("list",35)
  for( i in 1:35){
    
    class_acc[[i]] <- round(cf_object$table[i,i]/sum(cf_object$table[,i])*100,2)
    
    #print(class_acc)
    
  }
  class_acc <- data.frame(as.matrix(class_acc))
  class_acc <- cbind(unique(as.data.frame(cf_object$table)$Prediction),class_acc)
  colnames(class_acc) <- c("Class","Accuracy")
  rownames(class_acc) <- NULL
  class_acc
}


#Find the class-based accuracy for the test data predicted using SVM Linear Kernel
class_accuracy(cf_matrix_linear)



#####################################################################################################
############################## Model 2: Linear Discriminant Analysis #########################################
#####################################################################################################

trctrl <- trainControl(method = "cv", number = 10)
LDA_Model <- train(Letter ~., data = train_PCA, method = "lda",trControl=trctrl)
LDA_Predict <- predict(LDA_Model, newdata = test_PCA)
LDA_Matches <- LDA_Predict == test_PCA$Letter
LDA_AccRate<- (sum(LDA_Matches, na.rm = TRUE) / nrow(test_PCA)) * 100
LDA_AccRate

cf_matrix_lda <- confusionMatrix(LDA_Predict, test_PCA$Letter)

class_accuracy(cf_matrix_lda)

plotConfusionMatrix(cf_matrix_lda)

######################################################################################################
############################################### Model 3: KNN ##########################################
######################################################################################################

trctrl <- trainControl(method = "cv", number = 10)
grid_knn <- expand.grid(k = c(5))
KNN_Model <- train(Letter ~ ., data = train_PCA, method = "knn",tuneGrid = grid_knn,trControl=trctrl)
KNN_Predict <- predict(KNN_Model, newdata = test_PCA)
cf_matrix_knn <- confusionMatrix(KNN_Predict,test_PCA$Letter)

#Plot the confusion matrix for test data set predicted with KNN
plotConfusionMatrix(cf_matrix_knn)
###Overall accuracy obtained is ________%

##Overall Accuracy for the test dataset
cf_matrix_knn$overall

#Find the class-based accuracy for the test data predicted using KNN
class_accuracy(cf_matrix_knn)

