library(INLA)

# Load and prep data
housing <- read.csv("house.txt")
housing <- housing[c("price", "bedrooms", "bathrooms", "sqft_living", "grade", "sqft_above")]
housing$log_price <- log(housing$price)

# Train/test split
set.seed(1234)
n <- nrow(housing)
train_idx <- sample(1:n, size = 0.8 * n)
housing$train <- 0
housing$train[train_idx] <- 1

housing_train <- housing[housing$train == 1, ]
housing_test <- housing[housing$train == 0, ]

# Fit INLA model on training data only
inla_model <- inla(
  log_price ~ bedrooms + bathrooms + sqft_living + grade + sqft_above,
  data = housing_train,
  family = "gaussian",
  control.predictor = list(compute = TRUE)
)

# Extract posterior summaries for fixed effects
summary_fixed <- inla_model$summary.fixed

# Print 95% credible intervals for each parameter
ci_table <- data.frame(
  Coefficient = rownames(summary_fixed),
  Mean = summary_fixed$mean,
  Lower_95 = summary_fixed$`0.025quant`,
  Upper_95 = summary_fixed$`0.975quant`
)

cat("Posterior Means and 95% Credible Intervals:\n")
print(ci_table)

# Prepare test set design matrix (include intercept)
X_test <- model.matrix(~ bedrooms + bathrooms + sqft_living + grade + sqft_above, data = housing_test)

# Predict log_price for test set
coef_means <- summary_fixed$mean
pred_log_price <- X_test %*% coef_means

# True log_price values
true_log_price <- housing_test$log_price

# Evaluate performance
rmse <- sqrt(mean((true_log_price - pred_log_price)^2))
mae <- mean(abs(true_log_price - pred_log_price))
ss_total <- sum((true_log_price - mean(true_log_price))^2)
ss_resid <- sum((true_log_price - pred_log_price)^2)
r_squared <- 1 - (ss_resid / ss_total)

cat(sprintf("\nINLA RMSE: %.4f\n", rmse))
cat(sprintf("INLA MAE: %.4f\n", mae))
cat(sprintf("INLA R^2: %.4f\n", r_squared))
