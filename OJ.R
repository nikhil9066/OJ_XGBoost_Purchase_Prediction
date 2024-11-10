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

# Convert all factors (categorical variables) to numeric using model.matrix
train_x <- model.matrix(Purchase ~ . - 1, data = train_data)  # Omit the intercept
test_x <- model.matrix(Purchase ~ . - 1, data = test_data)

# The target variable remains the same
train_y <- train_data$Purchase
test_y <- test_data$Purchase

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# Set parameters for XGBoost
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "error",          # Classification error
  max_depth = 6,                  # Tree depth
  eta = 0.3,                      # Learning rate
  nthread = 2                     # Number of threads
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,  # Number of boosting rounds
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,  # Stop early if no improvement
  verbose = 1
)

# Make predictions on the test set
pred_prob <- predict(xgb_model, dtest)

# Convert probabilities to class labels (0 or 1)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Evaluate accuracy
confusionMatrix(factor(pred_class), factor(test_y))
