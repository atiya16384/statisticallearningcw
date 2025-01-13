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
if (!require("rpart")) install.packages("rpart", dependencies = TRUE)
if (!require("rpart.plot")) install.packages("rpart.plot", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("pROC")) install.packages("pROC", dependencies = TRUE)

library(rpart)
library(rpart.plot)
library(caret)
library(pROC)

# Ensure Diabetes_binary is a factor
Health_data$Diabetes_binary <- as.factor(Health_data$Diabetes_binary)

# Step 1: Split the data into training and testing sets
set.seed(36850162)  # Replace with your student number
train_index <- createDataPartition(Health_data$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- Health_data[train_index, ]
test_data <- Health_data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# Step 2: Fit an initial decision tree model
tree_model <- rpart(
  Diabetes_binary ~ .,  # Include all predictors
  data = train_data,
  method = "class",     # Classification tree
  control = rpart.control(cp = 0.01)  # Initial complexity parameter
)

cat("Initial Decision Tree Model:\n")
print(tree_model)

# Step 3: Visualize the initial tree
rpart.plot(tree_model, type = 3, extra = 102, fallen.leaves = TRUE,
           main = "Initial Decision Tree for Type II Diabetes")

# Step 4: Perform cross-validation to tune the complexity parameter (cp)
cat("Performing Cross-Validation to Tune cp...\n")
tree_tuned <- train(
  Diabetes_binary ~ .,
  data = train_data,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
  tuneLength = 10  # Test 10 different cp values
)

# Display the best cp value and final model
best_cp <- tree_tuned$bestTune$cp
cat("Best complexity parameter (cp):", best_cp, "\n")

final_tree <- tree_tuned$finalModel
cat("Final Decision Tree Model:\n")
print(final_tree)

# Step 5: Visualize the final tree
rpart.plot(final_tree, type = 3, extra = 102, fallen.leaves = TRUE,
           main = "Final Tuned Decision Tree for Type II Diabetes")

# Step 6: Evaluate the final model
# Predict on the test set
predicted_probs <- predict(final_tree, newdata = test_data, type = "prob")[, 2]  # Probability for class 1
predicted_classes <- predict(final_tree, newdata = test_data, type = "class")

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, as.factor(test_data$Diabetes_binary))
cat("Confusion Matrix:\n")
print(conf_matrix)

# ROC curve and AUC
test_data$Diabetes_binary <- factor(test_data$Diabetes_binary, levels = c(0, 1))  # Ensure proper factor levels
roc_curve <- roc(response = test_data$Diabetes_binary, 
                 predictor = predicted_probs, 
                 levels = c("0", "1"))  # Specify control and case levels
cat("AUC for the Final Model:\n")
print(auc(roc_curve))

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Decision Tree", col = "blue", lwd = 2)

