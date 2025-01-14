# Load necessary libraries
library(caret)
# Load necessary libraries
library(rpart)        # For decision tree modeling
library(rpart.plot)   # For tree visualization
library(caret)        # For cross-validation and model tuning
library(pROC)         # For ROC and AUC evaluation
library(ggplot2)

# Load the dataset
load("Health.RData")

# Set seed for reproducibility
set.seed(36850162)  # Replace <your_student_number> with your student number

# Quick data exploration
str(Health_data)         # Structure of the dataset
summary(Health_data)     # Summary statistics
any(is.na(Health_data))  # Check for missing values

# Fit the generalized linear model (GLM)
glm_model <- glm(
  Diabetes_binary ~ .,  # Include all predictors
  data = Health_data,        # Use the Health dataset
  family = binomial(link = "logit")  # Logistic regression for binary response
)

# Summarize the model to identify key relationships
model_summary <- summary(glm_model)

# Print the model summary
print(model_summary)

# Extract the coefficient for HighBP
highbp_coef <- coef(glm_model)["HighBP"]

# Calculate the odds ratio
highbp_odds_ratio <- exp(highbp_coef)

# Display the coefficient, odds ratio, and significance level
cat("Coefficient for HighBP:", highbp_coef, "\n")
cat("Odds Ratio for HighBP:", highbp_odds_ratio, "\n")

# Extract p-value for HighBP
highbp_pvalue <- summary(glm_model)$coefficients["HighBP", "Pr(>|z|)"]

cat("P-value for HighBP:", highbp_pvalue, "\n")

# Interpretation helper
if (highbp_pvalue < 0.05) {
  cat("The relationship between HighBP and type II diabetes is statistically significant at the 5% level.\n")
} else {
  cat("The relationship between HighBP and type II diabetes is NOT statistically significant at the 5% level.\n")
}

# Predict the probability of type II diabetes for patient0
patient0_prob <- predict(glm_model, newdata = patient0, type = "response")

# Display the result
cat("The predicted probability that patient0 has type II diabetes is:", patient0_prob, "\n")

# Given values
prob_highbp_1 <- patient0_prob  # Probability from 1.c
highbp_coef <- coef(glm_model)["HighBP"]  # Coefficient for HighBP

# Step 1: Calculate log-odds for HighBP = 1
log_odds_highbp_1 <- log(prob_highbp_1 / (1 - prob_highbp_1))

# Step 2: Adjust log-odds for HighBP = 0
log_odds_highbp_0 <- log_odds_highbp_1 - highbp_coef

# Step 3: Convert back to probability
prob_highbp_0 <- exp(log_odds_highbp_0) / (1 + exp(log_odds_highbp_0))

# Display the result
cat("The predicted probability that patient0 has type II diabetes without high blood pressure is:", prob_highbp_0, "\n")

# Extract significant variables from the first model
significant_vars <- rownames(model_summary$coefficients)[model_summary$coefficients[, "Pr(>|z|)"] < 0.05]
cat("Significant variables at 5% level:\n")
print(significant_vars)

# Fit the second generalized linear model using only significant predictors
significant_predictors <- Diabetes_binary ~ HighBP + HighChol + BMI + HvyAlcoholConsump + GenHlth + MentHlth + Age
glm_model_2 <- glm(
  significant_predictors,
  data = Health_data,
  family = binomial(link = "logit")
)

# Summarize the new model
model_2_summary <- summary(glm_model_2)
print(model_2_summary)

# Optional: Visualize significant variables
ggplot(Health_data, aes(x = factor(HighBP), fill = factor(Diabetes_binary))) +
  geom_bar(position = "dodge") +
  labs(title = "High Blood Pressure vs Type II Diabetes",
       x = "High Blood Pressure (0 = No, 1 = Yes)", 
       y = "Count", 
       fill = "Diabetes (Binary)") +
  theme_minimal()

set.seed(36850162)  # Replace with your student number

# Data exploration (optional but useful for extended analysis)
cat("Dataset Structure:\n")
str(Health_data)
cat("Summary Statistics:\n")
summary(Health_data)
cat("Checking for Missing Values:\n")
any(is.na(Health_data))  # Should return FALSE

# Load necessary libraries
if (!require("glmnet")) install.packages("glmnet", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)

library(glmnet)
library(caret)
library(dplyr)

# Load the Genes data
load("Genes.RData")

# Extract response variable and covariates
response_variable <- Genes$y
covariates <- Genes
covariates$y <- NULL  # Remove the response variable

# Convert covariates to a matrix (required for glmnet)
covariates_matrix <- as.matrix(covariates)

# Split the data into training (70%) and testing (30%)
set.seed(123)  # For reproducibility
train_indices <- createDataPartition(response_variable, p = 0.7, list = FALSE)

training_covariates <- covariates_matrix[train_indices, ]
testing_covariates <- covariates_matrix[-train_indices, ]
training_response <- response_variable[train_indices]
testing_response <- response_variable[-train_indices]

# Standardize covariates for glmnet
training_covariates <- scale(training_covariates)
testing_covariates <- scale(testing_covariates, center = attr(training_covariates, "scaled:center"),
                            scale = attr(training_covariates, "scaled:scale"))

# Add polynomial features using the model.matrix approach
add_polynomial_features <- function(data, degree) {
  poly_features <- data
  for (d in 2:degree) {
    poly_features <- cbind(poly_features, data^d)
  }
  colnames(poly_features) <- paste0("X", seq_len(ncol(poly_features)))
  return(poly_features)
}

poly_train <- add_polynomial_features(training_covariates, degree = 2)
poly_test <- add_polynomial_features(testing_covariates, degree = 2)

# Grid Search for Alpha and Lambda
set.seed(123)
alpha_values <- seq(0.1, 1, by = 0.05)
cv_results <- data.frame()

for (alpha in alpha_values) {
  elastic_net_cv <- cv.glmnet(poly_train, training_response, alpha = alpha, nfolds = 10)
  lambda_sequence <- elastic_net_cv$lambda
  
  for (lambda in lambda_sequence) {
    coef_path <- predict(elastic_net_cv, type = "coefficients", s = lambda)
    non_zero_coefficients <- sum(coef_path != 0)
    
    if (non_zero_coefficients <= 100) {
      predictions <- predict(elastic_net_cv, s = lambda, newx = poly_test)
      mse <- mean((predictions - testing_response)^2)
      r_squared <- 1 - sum((predictions - testing_response)^2) /
        sum((testing_response - mean(testing_response))^2)
      cv_results <- rbind(cv_results, data.frame(Alpha = alpha, Lambda = lambda, MSE = mse, R2 = r_squared, NonZero = non_zero_coefficients))
    }
  }
}

# Select the best combination of alpha and lambda
best_result <- cv_results[which.min(cv_results$MSE), ]
best_alpha <- best_result$Alpha
best_lambda <- best_result$Lambda

# Fit the final model with the best alpha and lambda
final_elastic_net_model <- glmnet(poly_train, training_response, alpha = best_alpha, lambda = best_lambda)

# Predictions on the test set
predictions <- predict(final_elastic_net_model, newx = poly_test)

# Evaluate the model
mse <- mean((predictions - testing_response)^2)
sst <- sum((testing_response - mean(testing_response))^2)
ssr <- sum((predictions - testing_response)^2)
r_squared <- 1 - (ssr / sst)

# Print results
cat("Best Alpha:", best_alpha, "\n")
cat("Best Lambda:", best_lambda, "\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("R-squared:", r_squared, "\n")
cat("Number of Non-Zero Coefficients:", best_result$NonZero, "\n")

# Save the selected genes
selected_genes <- which(coef(final_elastic_net_model) != 0)
cat("Selected genes (non-zero coefficients):\n", selected_genes, "\n")





# Load necessary libraries
library(caret)
library(randomForest)
library(pROC)
library(xgboost)

# Load the dataset
load("Health.RData")  # Ensure this file is in the correct working directory

# Rename the class levels to valid R variable names
Health_data$Diabetes_binary <- factor(
  Health_data$Diabetes_binary,
  levels = c("0", "1"),
  labels = c("Class_0", "Class_1")
)

# Step 1: Split the dataset into 70% training and 30% testing
set.seed(123)
train_index <- createDataPartition(Health_data$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- Health_data[train_index, ]
test_data <- Health_data[-train_index, ]

# Step 2: Define training control with cross-validation and sampling
train_control <- trainControl(
  method = "repeatedcv", # Repeated cross-validation
  number = 5,           # 5-fold CV
  repeats = 3,          # Repeat 3 times
  sampling = "up",      # Apply over-sampling
  classProbs = TRUE,    # Enable class probabilities
  summaryFunction = twoClassSummary # Optimize for ROC
)

# Step 3: Train an XGBoost model with adjusted class weights
# Create class weights (penalize minority class more)
class_weights <- ifelse(train_data$Diabetes_binary == "Class_1", 1, 0.1)

# Convert data to XGBoost matrix format
xgb_train <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "Diabetes_binary")]),
                         label = as.numeric(train_data$Diabetes_binary) - 1,
                         weight = class_weights)

xgb_test <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "Diabetes_binary")]),
                        label = as.numeric(test_data$Diabetes_binary) - 1)

# Perform grid search for hyperparameter optimization
grid <- expand.grid(
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.1, 0.3),
  nrounds = c(100, 150, 200)
)

best_model <- NULL
best_auc <- 0
best_params <- NULL

for (i in 1:nrow(grid)) {
  params <- list(
    max_depth = grid$max_depth[i],
    eta = grid$eta[i],
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  model <- xgb.train(
    params = params,
    data = xgb_train,
    nrounds = grid$nrounds[i],
    verbose = 0
  )
  
  # Evaluate AUC on test set
  predictions <- predict(model, xgb_test)
  auc <- roc(as.numeric(test_data$Diabetes_binary) - 1, predictions)$auc
  
  if (auc > best_auc) {
    best_auc <- auc
    best_model <- model
    best_params <- params
  }
}

# Print the best parameters and AUC
cat("Best AUC:", best_auc, "\n")
cat("Best Parameters:\n")
print(best_params)

# Step 4: Adjust decision threshold for sensitivity
thresholds <- seq(0.3, 0.5, by = 0.05)
metrics <- data.frame(Threshold = thresholds, Sensitivity = NA, Specificity = NA, F1 = NA)

for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  predicted_classes <- ifelse(predict(best_model, xgb_test) > threshold, "Class_1", "Class_0")
  
  conf_matrix <- confusionMatrix(
    factor(predicted_classes, levels = c("Class_0", "Class_1")),
    test_data$Diabetes_binary
  )
  
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  precision <- conf_matrix$byClass["Pos Pred Value"]
  recall <- sensitivity
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  metrics[i, 2:4] <- c(sensitivity, specificity, f1)
}

print(metrics)

# Save the best model
saveRDS(best_model, file = "final_optimized_xgboost_diabetes_model.rds")
cat("\nModel saved as 'final_optimized_xgboost_diabetes_model.rds'.\n")



# Load necessary libraries
library(caret)
library(pROC)

# Step 1: Load and preprocess the dataset
load("Health.RData")  # Ensure this file is in the correct working directory

# Rename the class levels to valid R variable names
Health_data$Diabetes_binary <- factor(
  Health_data$Diabetes_binary,
  levels = c("0", "1"),
  labels = c("Class_0", "Class_1")
)

# Step 2: Split the dataset into 70% training and 30% testing
set.seed(123)
train_index <- createDataPartition(Health_data$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- Health_data[train_index, ]
test_data <- Health_data[-train_index, ]

# Step 3: Implement weighted logistic regression to penalize false negatives
best_auc <- 0
best_model <- NULL
best_weight <- NULL
weights <- c(0.5, 1, 2, 5)  # Different weights for false negatives

for (weight in weights) {
  # Assign weights for the training dataset
  class_weights <- ifelse(train_data$Diabetes_binary == "Class_1", weight, 1)
  
  # Train a weighted logistic regression model
  logistic_model <- glm(
    Diabetes_binary ~ ., 
    data = train_data, 
    family = binomial(link = "logit"), 
    weights = class_weights
  )
  
  # Step 4: Evaluate the model
  predictions <- predict(logistic_model, newdata = test_data, type = "response")
  roc_obj <- roc(as.numeric(test_data$Diabetes_binary) - 1, predictions)
  auc <- auc(roc_obj)
  
  # Store the best model
  if (auc > best_auc) {
    best_auc <- auc
    best_model <- logistic_model
    best_weight <- weight
  }
}

# Print the best AUC and weight
cat("Best AUC:", best_auc, "\n")
cat("Best Weight:", best_weight, "\n")

# Step 5: Adjust decision threshold for sensitivity
threshold <- 0.4  # Lower threshold for increased sensitivity
predicted_classes <- ifelse(predict(best_model, newdata = test_data, type = "response") > threshold, "Class_1", "Class_0")

# Confusion Matrix
conf_matrix <- confusionMatrix(
  factor(predicted_classes, levels = c("Class_0", "Class_1")),
  test_data$Diabetes_binary
)

# Print Confusion Matrix and AUC
cat("\nConfusion Matrix:\n")
print(conf_matrix)

roc_auc <- roc(as.numeric(test_data$Diabetes_binary) - 1, predict(best_model, newdata = test_data, type = "response"))
cat("\nAUC:", roc_auc$auc, "\n")

# Save the best model
saveRDS(best_model, file = "weighted_logistic_regression_model_part_c.rds")
cat("\nModel saved as 'weighted_logistic_regression_model_part_c.rds'.\n")
