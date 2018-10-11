"
SYS 6018 - Kaggle Competition ; Predicting Blogger Age

"

# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Load necessary packages
library(ggplot2)
library(dplyr)
library(mice)
library(caret)
library(onehot)
library(zoo)
library(broom)
library(tidyverse)
library(hrbrthemes)
library(viridis)
library(ggridges)
library(treemap)
library(dplyr)
library(tidytext)
library(stringr)
library(wordcloud)
library(tm)
library(data.table) 

# LOADING DATA ; Initial look

train <- read.csv('train.csv',stringsAsFactors = F)
test <- read.csv('test.csv',stringsAsFactors = F)
blog1 <- bind_rows(train,test) # combine into 1 corpus

#Initial look at data
head(blog1)
names(blog1)
str(blog1)
summary(blog1)
colSums(is.na(blog1)) # no nulls found
unique(blog1$topic) # list all unique topics

# EXPLORATORY DATA ANALYSIS - EDA

train_eda = train
# Distribution of Age by Gender
# Density Plot 
ggplot(train_eda, aes(x=age, color=gender)) + stat_density(fill=NA) + ggtitle("Distribution of Age in the data")
# Conclusion - Doesn't seem like gender has too much to do with age distribution

# Distribution of Age by SunSign 
# Box Plot
ggplot(train_eda, aes(x=sign, y=age, fill=sign)) + geom_boxplot(alpha=0.3) + theme(legend.position="none")
# Conclusion - Nothing stands out when Age is plotted by SunSig. The median remains similar across all sun signs.

# Distribution of Age sample by Topic 
# Treemap
treemap(train_eda,index="topic",vSize="age",type="index")
# Conclusion - Topics with more data - indUnk,Student,Technology

# Distribution of Age by Topic 
# Violin Plot
sample_size = train_eda %>% group_by(topic) %>% summarize(num=n())
train_eda %>%
  left_join(sample_size) %>%
  mutate(myaxis = paste0(topic, "\n")) %>%
  ggplot( aes(x=myaxis, y=age, fill=topic)) +
  geom_violin(width=1.4) +
  geom_boxplot(width=0.1, color="grey", alpha=0.2) +
  scale_fill_viridis(discrete = TRUE) +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("A Violin wrapping a boxplot") +
  xlab("")  + coord_flip()
# Conclusion - Some topics are age-sensitive

# Distribution of specific Topics with Age
# Density Plot
train_student <- train_eda[which(train_eda$topic == "Student" |train_eda$topic == "Science" | 
                                   train_eda$topic == "Marketing" |train_eda$topic == "Military"|train_eda$topic == "Transportation" )
                           ,names(train_eda) %in% c("age","topic")]

ggplot(train_student, aes(x=age, color=topic,fill = topic)) + geom_density(alpha=0.55) + ggtitle("Distribution of Age for specific Topics")
# Conclusion - marketing and science heavily weighed in the mid 20s , military equally distributed 
# and student is concentrated in the below 17 range, transportation is more concentrated in the mid 30s range


# Distribution of Date with Age 
# Scatter Plot
train_eda$date <- gsub(",","-", train_eda$date)
train_eda$date <- as.Date(train_eda$date, "%d-%B-%Y" , tryFormats = "%Y-%m-%d")
ggplot(train_eda, aes(x=date, y=age)) + geom_point()
# Conclusion - Blogs published prior to 2002 have a higher chance to fall in the age category 25-30 years. 
# Maybe helpful to extract just year of publishing

# -----------------------------------------------------------------------------------------------------------------------------

# DATA PRE-PROCESSING and FEATURE EXTRACTION

# Convert date column to date type 
blog1$date <- gsub(",","-", blog1$date)
blog1$date <- as.Date(blog1$date, "%d-%B-%Y" , tryFormats = "%Y-%m-%d")

# One-hot encoding for Gender,Topic
# Not encoding SunSign,Post Ids and User Ids since i don't intend to use them in prediction
dmy <- dummyVars(" ~ gender", data = blog1 ,sep = "_",fullRank = T)
blog2 <- data.frame(predict(dmy, newdata = blog1))
blog2$date <- as.Date(blog1$date)                                  # attach date
blog2[is.na(blog2$date), ]['date'] = median(blog2$date, na.rm = T) # replace missing date values with median
blog2$year =  format(as.Date(blog2$date, format="%Y-%m-%d"),"%Y")  # extract year and store as column
blog2$text <- blog1$text                                           # attach text
blog2$age <- blog1$age                                             # attach age
blog2$postid <- blog1$post.id
blog2$topic <- blog1$topic

# Text Mining
# Split data back
train_text <- blog2[1:442961,] 
test_text <- blog2[442962:681284,]


#---- Exploring the Text Data --------
textcorpus_orig <- bind_rows(train_text,test_text) 

# Print the training corpus
textcorpus_eda <- data_frame(line = textcorpus_orig$postid , topic = textcorpus_orig$topic, text = textcorpus_orig$text)

# remove numbers from text
textcorpus_eda$text <- removeNumbers(textcorpus_eda$text)

# Use unnest() to separate out words from text
raw_text <- textcorpus_eda %>%
  unnest_tokens(word, text)

# Distribution of posts by topic
raw_text %>%
  group_by(topic) %>%
  summarize(messages = n_distinct(line)) %>%
  ggplot(aes(topic, messages)) +
  geom_col() +
  coord_flip()
# We see the max posts for topics like indUnk and Student (observed before as well)

# Remove words and numbers from text
data(stop_words)
custom_stop_words <- bind_rows(data_frame(word = c("miss","urllink","nbsp","â","ã"), 
                                          lexicon = c("custom")), 
                               stop_words)
usenet_words <- raw_text %>%
  anti_join(custom_stop_words)

usenet_words %>%
  count(word, sort = TRUE)

# wordcloud of frequently occuring words
usenet_words %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100))

# Sentiment Analysis by Word
contributions <- usenet_words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>% #using afinn that allots sentiment score from -5 to 5 
  group_by(word) %>%
  summarize(occurences = n(),
            contribution = sum(score))
# words like time,people,day and love which rate higher on positive sentiment score occur frequently

# Which words had the most effect on sentiment scores overall
contributions %>%
  top_n(25, abs(contribution)) %>%
  mutate(word = reorder(word, contribution)) %>%
  ggplot(aes(word, contribution, fill = contribution > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip()
# love has the highest +ve sentiment score and bad has the most negative sentiment score in our text corpus

# Sentiment analysis by each Post
sentiment_messages <- usenet_words %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(topic, line) %>%
  summarize(sentiment = mean(score),
            words = n()) %>%
  ungroup() %>%
  filter(words >= 5)

sentiment_messages %>%
  arrange(desc(sentiment)) # sentiment score for each post

# Proceeding to extract features - TFIDF
# combine corpus
textcorpus <- bind_rows(train_text,test_text) # combine into 1 corpus
textcorpus = textcorpus[,c("postid", "text")]
names(textcorpus) = c("doc_id", "text")
textcorpus = VCorpus(DataframeSource(textcorpus))

# clean corpus
textcorpus.clean = tm_map(textcorpus, stripWhitespace)                          # remove extra whitespace
textcorpus.clean = tm_map(textcorpus.clean, removeNumbers)                      # remove numbers
textcorpus.clean = tm_map(textcorpus.clean, removePunctuation)                  # remove punctuation
textcorpus.clean = tm_map(textcorpus.clean, content_transformer(tolower))       # ignore case
textcorpus.clean = tm_map(textcorpus.clean, removeWords, stopwords("english"))  # remove stop words
textcorpus.clean = tm_map(textcorpus.clean, stemDocument)                       # stem all words

# compute TFIDF
textcorpus.clean.tf = DocumentTermMatrix(textcorpus.clean, control = list(weighting = weightTfIdf))
textcorpus.clean.tf <- as.matrix(removeSparseTerms(textcorpus.clean.tf, .99))
dim(textcorpus.clean.tf)

processed_data <- as.data.table(as.matrix(textcorpus.clean.tf))

# Combining Processed Data with all columns 
data_one <- cbind(data.table(gendermale = textcorpus_orig$gendermale, date = textcorpus_orig$date, 
                             year = textcorpus_orig$year,age = textcorpus_orig$age),processed_data)
colnames(data_one)[2] <- "datepub"
colnames(data_one)[3] <- "yearpub"
# Split data back into train and test for modelling purposes 
train_one <- data_one[!is.na(age)]
test_one <- data_one[is.na(age)]

# Training model on MultiLinear Regression and implementing k-fold cross-validation
train.control <- trainControl(method = "cv", number = 5) # k-folds = 5
# Train the model
model1 <- train(age ~., data = train_one, method = "lm",trControl = train.control) # multilinear regression
# Summarize the results
print(model1)
# predict on test
prediction1 <- predict(model1,newdata = test_one)
test_id <- read.csv('test.csv',stringsAsFactors = F)
solution <- data.frame(test_id$user.id)
solution$age = prediction1
solution$age = trunc(solution$age)
output1 = solution %>% group_by(user.id) %>% summarise(age = min(age))

# write to csv
write.table(output1, file = 'submission2.csv', col.names = c('user.id', 'age'), sep = ',', row.names = F)

## Lasso Model
x1 <- data.matrix(train_one[,-3])
y1 <- train_one$age

lasso.lambda <- cv.glmnet(y=y1, x=x1, nfolds = 7)
coef(lasso.lambda)

#blog.lasso1 <- glmnet(x, y, alpha = 1)

prediction1 <- predict(lasso.lambda, newx = data.matrix(test_one))
# prediction1 <- predict(blog.lasso1, newx = test_one)
test_age <- as.data.frame(cbind(test_one$user.id,pred))
names(test_age) <- c('user.id', 'age')

output2 <- test_age %>% 
  group_by(user.id) %>% 
  summarise(age = mean(age))

# write to csv
write.table(output2, file = 'submission3.csv', col.names = c('user.id', 'age'), sep = ',', row.names = F)
