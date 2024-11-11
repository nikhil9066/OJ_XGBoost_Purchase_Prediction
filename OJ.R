# Load necessary libraries
library(ISLR2)
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2) 
library(pROC)
library(PRROC)
library(corrplot)

# Set seed for reproducibility
set.seed(555)

# Encode Purchase as binary: 1 if "CH", 0 if "MM"
OJ$Purchase <- as.numeric(OJ$Purchase == "CH")

# Split the dataset into 50:50 train-test sets
trainIndex <- createDataPartition(OJ$Purchase, p = 0.5, list = FALSE)
train_data <- OJ[trainIndex, ]
test_data <- OJ[-trainIndex, ]

# 1. Before Training: Visualize Target Variable Distribution
ggplot(OJ, aes(x = factor(Purchase))) +
  geom_bar(aes(fill = factor(Purchase)), color = "black") +
  labs(title = "Target Variable Distribution (Purchase)", x = "Purchase (CH=1, MM=0)", y = "Count") +
  theme_minimal()

# 2. Before Training: Visualize Correlation Matrix
numeric_features <- train_data %>%
  select(where(is.numeric)) %>%
  select(-Purchase)  # Exclude the target variable for correlation

cor_matrix <- cor(numeric_features, use = "complete.obs")
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.col = "black",
         title = "Feature Correlation Matrix", mar = c(0, 0, 2, 0))

# Convert categorical variables to numeric using model.matrix
train_x <- model.matrix(Purchase ~ . - 1, data = train_data)  # Exclude intercept
test_x <- model.matrix(Purchase ~ . - 1, data = test_data)

# Extract target variables
train_y <- train_data$Purchase
test_y <- test_data$Purchase

# Convert datasets to DMatrix format
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# Plot feature distributions (e.g., PriceCH, PriceMM, and StoreID)
ggplot(train_data, aes(x = PriceCH)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of PriceCH", x = "PriceCH", y = "Frequency") +
  theme_minimal()

ggplot(train_data, aes(x = PriceMM)) +
  geom_histogram(bins = 30, fill = "salmon", color = "black") +
  labs(title = "Distribution of PriceMM", x = "PriceMM", y = "Frequency") +
  theme_minimal()

ggplot(train_data, aes(x = StoreID)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Distribution of StoreID", x = "StoreID", y = "Frequency") +
  theme_minimal()

# Set base parameters for XGBoost
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "error",          # Error rate
  max_depth = 6,                  # Tree depth
  eta = 0.3,                      # Learning rate
  nthread = 2                     # Number of threads
)

# Train the XGBoost model
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
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Evaluate the model with confusion matrix for initial model
conf_matrix_base <- table(Prediction = pred_class, Reference = test_y)
accuracy_base <- sum(diag(conf_matrix_base)) / sum(conf_matrix_base)

# --- After Model Training: Plot Feature Importance ---
importance_matrix <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)
xgb.plot.importance(importance_matrix, main = "Feature Importance", top_n = 10)

# --- After Model Training: ROC Curve for Model Performance ---
# Plot the ROC curve
test_preds_prob <- predict(xgb_model, dtest)
roc_curve <- roc(test_y, test_preds_prob)

# Plot ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)
text(0.5, 0.4, paste("AUC =", round(roc_curve$auc, 3)), cex = 1.2, col = "black")

# Perform cross-validation to find the best parameters
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,  # 5-fold cross-validation
  early_stopping_rounds = 10,
  verbose = 1
)

# Get the best number of rounds from cross-validation
best_nrounds <- xgb_cv$best_iteration

# Learning curve plot using cross-validation error from the xgb.cv results
cv_error <- xgb_cv$evaluation_log
ggplot(cv_error, aes(x = iter)) +
  geom_line(aes(y = train_error_mean, color = "Train Error")) +
  geom_line(aes(y = test_error_mean, color = "Test Error")) +
  labs(title = "Learning Curve", x = "Number of Iterations (Boosting Rounds)", y = "Error") +
  theme_minimal() +
  scale_color_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  theme(legend.position = "bottom")

# Hyperparameter tuning using caret's trainControl
ctrl <- trainControl(method = "cv", number = 5)

xgb_tuned <- train(
  Purchase ~ ., data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = expand.grid(
    nrounds = c(100, 200),
    max_depth = c(3, 6, 9),
    eta = c(0.01, 0.1),
    gamma = c(0, 1),
    colsample_bytree = c(0.7, 1),
    min_child_weight = c(1, 5),
    subsample = c(0.8, 1)  # Add subsample parameter
  )
)

# Display the best tuned parameters
xgb_tuned$bestTune

# Train the final model with tuned parameters using the best number of rounds
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = xgb_tuned$bestTune$max_depth,  # From bestTune
  eta = xgb_tuned$bestTune$eta,              # From bestTune
  subsample = xgb_tuned$bestTune$subsample,  # From bestTune
  colsample_bytree = xgb_tuned$bestTune$colsample_bytree,  # From bestTune
  gamma = xgb_tuned$bestTune$gamma,          # From bestTune
  min_child_weight = xgb_tuned$bestTune$min_child_weight   # From bestTune
)

# Train final model with best parameters and best number of rounds
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds  # Best number of rounds from tuning
)

# Make final predictions on the test set
test_preds_prob <- predict(final_model, dtest, iteration_range = c(1, best_nrounds))
test_preds <- ifelse(test_preds_prob > 0.5, 1, 0)

# --- After Model Training: Plot Feature Importance ---
importance_matrix <- xgb.importance(feature_names = colnames(train_x), model = final_model)
xgb.plot.importance(importance_matrix, main = "Feature Importance", top_n = 10)

# --- After Model Training: ROC Curve for Model Performance ---
# Plot the ROC curve
test_preds_prob <- predict(final_model, dtest)
roc_curve <- roc(test_y, test_preds_prob)

# Create a confusion matrix for final model
conf_matrix_final <- table(Prediction = test_preds, Reference = test_y)
accuracy_final <- sum(diag(conf_matrix_final)) / sum(conf_matrix_final)

# Print all accuracies
cat("Accuracy of Initial Model (Base Params):", accuracy_base, "\n")
cat("Accuracy of Tuned Model (From Cross-Validation):", accuracy_base, "\n")
cat("Accuracy of Final Model (Best Params and Best Rounds):", accuracy_final, "\n")

# Create confusion matrix plot for final model
conf_matrix_df <- as.data.frame(as.table(conf_matrix_final))
colnames(conf_matrix_df) <- c("Prediction", "Reference", "Count")

ggplot(conf_matrix_df, aes(x = Prediction, y = Reference, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix", x = "Prediction", y = "Reference") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Calculate and print precision, recall, and F1 score for final model
precision <- conf_matrix_final[2, 2] / (conf_matrix_final[2, 2] + conf_matrix_final[2, 1])
recall <- conf_matrix_final[2, 2] / (conf_matrix_final[2, 2] + conf_matrix_final[1, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1_score, "\n")

# Plot ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)
text(0.5, 0.4, paste("AUC =", round(roc_curve$auc, 3)), cex = 1.2, col = "black")

# Create a data frame for ROC and Precision-Recall curves
pr_curve <- pr.curve(scores.class0 = test_preds_prob, weights.class0 = test_y, curve = TRUE)
roc_data <- data.frame(Specificity = roc_curve$specificities, Sensitivity = roc_curve$sensitivities, Curve = "ROC")
pr_data <- data.frame(Specificity = pr_curve$curve[,1], Sensitivity = pr_curve$curve[,2], Curve = "Precision-Recall")

# Combine the two data frames
combined_data <- rbind(roc_data, pr_data)

# Plot both ROC and Precision-Recall curves side by side using facets
ggplot(combined_data, aes(x = Specificity, y = Sensitivity, color = Curve)) +
  geom_line(size = 1) +
  facet_wrap(~Curve, scales = "free") +
  labs(title = "ROC Curve and Precision-Recall Curve", x = "1 - Specificity / Precision", y = "Sensitivity / Recall") +
  theme_minimal() +
  theme(legend.position = "none")
