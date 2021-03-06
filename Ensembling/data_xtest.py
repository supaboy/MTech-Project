library(tm)
#library(PyTorchTextTools)
library(readr)
library(e1071)
library(dplyr)
library(caret)
#df<- read.csv(file.choose(), stringsAsFactors = FALSE)
df<- read.csv("datasets/investment reports dataset/*", stringsAsFactors = FALSE)
#set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
df<-df[1:137,]
df$class <- as.factor(df$class)
corpus <- Corpus(VectorSource(df$text))
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
dtm <- DocumentTermMatrix(corpus.clean)
df.train <- df[1:100,]
df.test <- df[101:137,]

dtm.train <- dtm[1:100,]
dtm.test <- dtm[101:137,]

corpus.clean.train <- corpus.clean[1:100]
corpus.clean.test <- corpus.clean[101:137]
dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))

dim(dtm.train.nb)
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  #y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
#--------------------------------------Naive Bayes Algorithm-----------------------------------------
system.time( NB_classifier <- naiveBayes(trainNB, df.train$class, laplace = 1) )
system.time( NB_pred <- predict(NB_classifier, newdata=testNB) )
#table("Predictions"= NB_pred,  "Actual" = df.test$class )
NB_conf.mat <- confusionMatrix(NB_pred, df.test$class)

NB_conf.mat
#conf.mat$byClass
#conf.mat$overall
NB_conf.mat$overall['Accuracy']

#_____________________________Support Vectore machine (SVM)-----------------------------------------
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  #y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
trainNB<- as.data.frame(trainNB)
testNB<- as.data.frame(testNB)
train_SVM<-cbind(class=factor(df.train$class), trainNB)
test_SVM<- cbind(class=factor(df.test$class), testNB)
train_SVM<-as.data.frame(train_SVM)
test_SVM<-as.data.frame(test_SVM)
system.time( SVM_classifier <- svm(class~.,data = train_SVM) )
system.time( SVM_pred <- predict(SVM_classifier, na.omit(test_SVM)) )
SVM_conf.mat <- confusionMatrix(SVM_pred, test_SVM$class,positive = "1")
SVM_conf.mat
#table("Predictions"= NB_pred,  "Actual" = df.test$class )
SVM_conf.mat$overall['Accuracy']
