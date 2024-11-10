library(ISLR2)
library(xgboost)
library(caret)
library(dplyr)

str(OJ)
?OJ  # To understand the dataset

set.seed(555)

# Encode Purchase as a factor with levels 0 and 1
OJ$Purchase <- as.numeric(OJ$Purchase == "CH")  # 1 if Purchase is CH, 0 if MM

# Create a 50:50 train-test split
trainIndex <- createDataPartition(OJ$Purchase, p = 0.5, list = FALSE)
train_data <- OJ[trainIndex, ]
test_data <- OJ[-trainIndex, ]

# Separate features and target variable
train_x <- as.matrix(train_data %>% select(-Purchase))
train_y <- train_data$Purchase

test_x <- as.matrix(test_data %>% select(-Purchase))
test_y <- test_data$Purchase
