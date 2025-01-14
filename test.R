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
library(pROC)
library(xgboost)

# Load the dataset
load("Health.RData")  # Ensure this file is in your working directory

# Check data structure
str(Health_data)

# Rename class levels to valid R names
Health_data$Diabetes_binary <- factor(
  Health_data$Diabetes_binary,
  levels = c("0", "1"),
  labels = c("Class_0", "Class_1")
)

# Split dataset: 70% training, 30% testing
set.seed(123)
train_index <- createDataPartition(Health_data$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- Health_data[train_index, ]
test_data <- Health_data[-train_index, ]

# TrainControl for sensitivity (use ROC optimization)
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  sampling = "up",  # Upsample to balance data
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train XGBoost Model
set.seed(123)
xgb_model <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  trControl = train_control
)

# Evaluate Confusion Matrix
xgb_predictions <- predict(xgb_model, newdata = test_data)
conf_matrix <- confusionMatrix(xgb_predictions, test_data$Diabetes_binary)
print(conf_matrix)

# Adjust decision threshold for sensitivity
xgb_probabilities <- predict(xgb_model, newdata = test_data, type = "prob")[, "Class_1"]
threshold <- 0.4
xgb_adjusted_predictions <- factor(ifelse(xgb_probabilities > threshold, "Class_1", "Class_0"),
                                   levels = c("Class_0", "Class_1"))

adjusted_conf_matrix <- confusionMatrix(xgb_adjusted_predictions, test_data$Diabetes_binary)
print(adjusted_conf_matrix)

# Evaluate model performance using ROC curve and AUC
xgb_roc <- roc(as.numeric(test_data$Diabetes_binary) - 1, xgb_probabilities)

# Plot ROC Curve
plot(xgb_roc, main = "ROC Curve for XGBoost Model")
auc_value <- auc(xgb_roc)
cat("AUC Value: ", auc_value, "\n")

# Sensitivity and Specificity
sensitivity <- adjusted_conf_matrix$byClass["Sensitivity"]
specificity <- adjusted_conf_matrix$byClass["Specificity"]

# Print Performance Metrics
cat("Adjusted Sensitivity: ", sensitivity, "\n")
cat("Adjusted Specificity: ", specificity, "\n")




# Load necessary libraries
library(caret)
library(e1071)
library(pROC)
library(ROSE) # For oversampling and undersampling

# Load dataset
load("Health.RData")

# Rename class levels for compatibility
Health_data$Diabetes_binary <- factor(
  Health_data$Diabetes_binary,
  levels = c("0", "1"),
  labels = c("Class_0", "Class_1")
)

# Split dataset: 70% training, 30% testing
set.seed(123)
train_index <- createDataPartition(Health_data$Diabetes_binary, p = 0.7, list = FALSE)
train_data <- Health_data[train_index, ]
test_data <- Health_data[-train_index, ]

# Apply ROSE to balance the training data
train_data_balanced <- ROSE(Diabetes_binary ~ ., data = train_data, seed = 123)$data

# Define custom trainControl with class probabilities and ROC metric
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train a cost-sensitive SVM model with tuned parameters
set.seed(123)
svm_model <- train(
  Diabetes_binary ~ .,
  data = train_data_balanced,
  method = "svmRadial",
  metric = "ROC",
  trControl = train_control,
  tuneLength = 5
)

# Generate predicted probabilities on the test set
svm_probabilities <- predict(svm_model, newdata = test_data, type = "prob")[, "Class_1"]

# Adjust decision threshold to prioritize sensitivity
threshold <- 0.3
svm_adjusted_predictions <- factor(
  ifelse(svm_probabilities > threshold, "Class_1", "Class_0"),
  levels = c("Class_0", "Class_1")
)

# Confusion matrix for adjusted threshold
svm_adjusted_conf_matrix <- confusionMatrix(svm_adjusted_predictions, test_data$Diabetes_binary)
print(svm_adjusted_conf_matrix)

# Evaluate model performance using ROC curve and AUC
svm_roc <- roc(as.numeric(test_data$Diabetes_binary) - 1, svm_probabilities)

# Plot ROC curve
plot(svm_roc, main = "ROC Curve for Optimized SVM Model")
svm_auc <- auc(svm_roc)
cat("AUC Value: ", svm_auc, "\n")

# Extract sensitivity and specificity
svm_sensitivity <- svm_adjusted_conf_matrix$byClass["Sensitivity"]
svm_specificity <- svm_adjusted_conf_matrix$byClass["Specificity"]

# Print sensitivity, specificity, and additional metrics
cat("Adjusted Sensitivity: ", svm_sensitivity, "\n")
cat("Adjusted Specificity: ", svm_specificity, "\n")
