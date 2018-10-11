#load libraries
library(data.table) 
library(jsonlite)
library(purrr)
library(RecordLinkage) 
library(stringr)
library(tm)
library(tidyverse)
library(dplyr)
library(glmnet)
library(SnowballC)

# Read datasets
blog.ori <- read.csv('train.csv', stringsAsFactors = FALSE)
blog.test <- read.csv('test.csv', stringsAsFactors = FALSE)

# I did not use date as a variable, so exclude that from the datasets
train <- blog.ori[, c('post.id', 'user.id', 'gender', 'topic', 'sign', 'text', 'age')]
test <- blog.test[, c('post.id', 'user.id', 'gender', 'topic', 'sign', 'text')]

# Add age to the test set and bind the train and test data sets
test$age <- 0
tdata <- rbind(train,test)

# tdata$desc_word_count <- str_count(tdata$text,pattern = "\\w+")

# Build the Corpus
text_corpus <- Corpus(VectorSource(tdata$text))
# inspect(text_corpus[1:4])
# print(lapply(text_corpus[1:2], as.character))

# text cleaning with the tm library
text.clean = tm_map(text_corpus, stripWhitespace)                          # remove extra whitespace
text.clean = tm_map(text.clean, removeNumbers)                      # remove numbers
text.clean= tm_map(text.clean, removePunctuation)                  # remove punctuation
text.clean = tm_map(text.clean, content_transformer(tolower))       # ignore case
text.clean = tm_map(text.clean, removeWords, stopwords("english"))  # remove stop words
#text.clean1 = tm_map(text.clean, PlainTextDocument)
text.clean = tm_map(text.clean, stemDocument)    
# print(as.character(text.clean[[1]]))
# text.clean[[1]]$content

# Convert the copus into document matrix
docterm_corpus <- DocumentTermMatrix(text.clean)
dim(docterm_corpus)

# Delete the words that has very low frequency, choose sparse level at 0.95
new_docterm_corpus <- removeSparseTerms(docterm_corpus,sparse = 0.95)
dim(new_docterm_corpus)
# inspect(new_docterm_corpus)

# See the frequencies of the words
colS <- colSums(as.matrix(new_docterm_corpus))
length(colS)
doc_features <- data.table(name = attributes(colS)$names, count = colS)

#most frequent and least frequent words
doc_features[order(-count)][1:10] #top 10 most frequent words
doc_features[order(count)][1:10] #least 10 frequent words

# Create the TF-IDF matrix
data_mining_tf <- as.data.table(as.matrix(weightTfIdf(new_docterm_corpus)))

#create data set for training
# processed_data <- as.data.table(as.matrix(new_docterm_corpus))

#combing the data
data_one <- cbind(data.table(post.id = tdata$post.id, user.id = tdata$user.id, age = tdata$age, gender = tdata$gender, sign = tdata$sign, topic = tdata$topic),data_mining_tf)

#split the data set into train and test
train_one <- data_one[age != 0]
test_one <- data_one[age == 0]
# test_one[,interest_level := NULL]

# Change characters into factors
train_one[,c('sign','topic','gender')] <- lapply(train_one[,c('sign','topic','gender')] , factor)
test_one[,c('sign','topic','gender')] <- lapply(test_one[,c('sign','topic','gender')] , factor)
test_one$age <- NULL

## Simple linear regression
blog.lm1 <- lm(age ~ .-user.id-post.id, data = train_one)
summary(blog.lm1)

pred <- predict(blog.lm1, newdata = test_one)
test_age <- as.data.frame(cbind(test_one$user.id,pred))
names(test_age) <- c('user.id', 'age')

result <- test_age %>% 
  group_by(user.id) %>% 
  summarise(age = mean(age))
  
write.csv(result, file = 'submission1.csv', row.names = FALSE)

## Features selection
blog.lm2 <- lm(age ~ .-user.id-post.id-still-start-someth-second-reason-never-made-long-life-left-hard-happen-face-doesnt-decid-better-best-anyon-also-alreadi-almost, data = train_one)
summary(blog.lm2)

pred <- predict(blog.lm2, newdata = test_one)
test_age <- as.data.frame(cbind(test_one$user.id,pred))
names(test_age) <- c('user.id', 'age')

result <- test_age %>% 
  group_by(user.id) %>% 
  summarise(age = mean(age))

write.csv(result, file = 'submission2.csv', row.names = FALSE)

## Lasso
x1 <- data.matrix(train_one[,-3])
y1 <- train_one$age

lasso.lambda <- cv.glmnet(y=y1, x=x1, nfolds = 7)
coef(lasso.lambda)

#blog.lasso1 <- glmnet(x, y, alpha = 1)

pred <- predict(lasso.lambda, newx = data.matrix(test_one))
# pred <- predict(blog.lasso1, newx = test_one)
test_age <- as.data.frame(cbind(test_one$user.id,pred))
names(test_age) <- c('user.id', 'age')

result <- test_age %>% 
  group_by(user.id) %>% 
  summarise(age = mean(age))

write.csv(result, file = 'submission4.csv', row.names = FALSE)

## Linear Model 3
blog.lm3 <- lm(age ~ .-user.id-post.id-still-start-someth-second-reason-never-made-long-life-left-hard-happen-face-doesnt-decid-better-best-anyon-also-alreadi-almost-hour-word-end, data = train_one)
summary(blog.lm3)

pred <- predict(blog.lm3, newdata = test_one)
test_age <- as.data.frame(cbind(test_one$user.id,pred))
names(test_age) <- c('user.id', 'age')

result <- test_age %>% 
  group_by(user.id) %>% 
  summarise(age = mean(age))

write.csv(result, file = 'submission5.csv', row.names = FALSE)
