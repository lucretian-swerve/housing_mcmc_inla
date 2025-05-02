# Load and preprocess data
housing <- read.csv("house.txt")
housing <- housing[c("price", "bedrooms", "bathrooms", "sqft_living", "grade", "sqft_above")]
housing$log_price <- log(housing$price)

# Train/test split
set.seed(1234)
n <- nrow(housing)
train_idx <- sample(1:n, size = 0.8 * n)
housing_train <- housing[train_idx, ]
housing_test <- housing[-train_idx, ]

# Standardize predictors
X_train <- scale(as.matrix(housing_train[, c("bedrooms", "bathrooms", "sqft_living", "grade", "sqft_above")]))
X_test  <- scale(as.matrix(housing_test[, c("bedrooms", "bathrooms", "sqft_living", "grade", "sqft_above")]),
                 center = attr(X_train, "scaled:center"),
                 scale = attr(X_train, "scaled:scale"))

# Add intercept
X_train <- cbind(1, X_train)
X_test <- cbind(1, X_test)

y_train <- housing_train$log_price
y_test  <- housing_test$log_price

# Log-likelihood
log_likelihood <- function(X, y, beta, sigma2) {
  mu <- X %*% beta
  -0.5 * length(y) * log(2 * pi * sigma2) - sum((y - mu)^2) / (2 * sigma2)
}

# Log-prior for beta
log_prior_beta <- function(beta, tau2 = 100) {
  -0.5 * sum(beta^2) / tau2
}

# Metropolis-within-Gibbs sampler
bayes_lm_sampler <- function(X, y, iterations = 10000, proposal_sd = 0.001,
                             tau2 = 100, a0 = 2, b0 = 1, beta_init = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  
  if (is.null(beta_init)) {
    beta <- rep(0, p)
  } else {
    beta <- beta_init
  }
  
  sigma2 <- 1
  beta_samples <- matrix(NA, nrow = iterations, ncol = p)
  sigma2_samples <- numeric(iterations)
  accept_count <- 0
  
  current_lp <- log_likelihood(X, y, beta, sigma2) + log_prior_beta(beta, tau2)
  
  for (i in 1:iterations) {
    # Propose new beta
    proposal <- beta + rnorm(p, 0, proposal_sd)
    proposal_lp <- log_likelihood(X, y, proposal, sigma2) + log_prior_beta(proposal, tau2)
    
    # Accept/reject
    accept_prob <- exp(proposal_lp - current_lp)
    if (runif(1) < accept_prob) {
      beta <- proposal
      current_lp <- proposal_lp
      accept_count <- accept_count + 1
    }
    
    # Sample sigma2 from full conditional
    resid <- y - X %*% beta
    a_post <- a0 + n / 2
    b_post <- b0 + sum(resid^2) / 2
    sigma2 <- 1 / rgamma(1, shape = a_post, rate = b_post)
    
    # Store samples
    beta_samples[i, ] <- beta
    sigma2_samples[i] <- sigma2
  }
  
  cat(sprintf("Acceptance rate: %.2f%%\n", 100 * accept_count / iterations))
  
  return(list(beta = beta_samples, sigma2 = sigma2_samples))
}

# Initialize at OLS estimates
lm_init <- lm(log_price ~ bedrooms + bathrooms + sqft_living + grade + sqft_above, data = housing_train)
beta_init <- coef(lm_init)

# Set timer
start_time <- Sys.time()
# Run MCMC
set.seed(2024)
iterations <- 100000
burn_in <- 10000
samples <- bayes_lm_sampler(X_train, y_train, iterations = iterations, proposal_sd = 0.001,
                            beta_init = beta_init)
# Stop timer
end_time <- Sys.time()

# Discard burn-in
beta_post <- samples$beta[(burn_in + 1):iterations, ]
sigma2_post <- samples$sigma2[(burn_in + 1):iterations]

# Posterior summaries
posterior_means <- colMeans(beta_post)
posterior_sds <- apply(beta_post, 2, sd)

# Calculate 95% credible intervals
posterior_ci_lower <- apply(beta_post, 2, quantile, probs = 0.025)
posterior_ci_upper <- apply(beta_post, 2, quantile, probs = 0.975)

# Clean summary table
coefficient_names <- c("Intercept", "Bedrooms", "Bathrooms", "Sqft_living", "Grade", "Sqft_above")
ci_table <- data.frame(
  Coefficient = coefficient_names,
  Mean = posterior_means,
  Lower_95 = posterior_ci_lower,
  Upper_95 = posterior_ci_upper
)

cat("\nPosterior Means and 95% Credible Intervals (Beta):\n")
print(ci_table)

# Sigma^2 posterior mean
sigma2_mean <- mean(sigma2_post)
cat(sprintf("\nPosterior Mean of Sigma^2: %.4f\n", sigma2_mean))

# Trace plots (post burn-in)
par(mfrow = c(2, 3))
for (j in 1:ncol(beta_post)) {
  plot(beta_post[, j], type = "l",
       main = paste("Trace: Beta", j, "(post burn-in)"),
       xlab = "Iteration", ylab = "Value")
}
plot(sigma2_post, type = "l", main = "Trace: Sigma^2 (post burn-in)",
     xlab = "Iteration", ylab = "Value")
par(mfrow = c(1, 1))

# Histograms (posterior densities)
par(mfrow = c(2, 3))
for (j in 1:ncol(beta_post)) {
  hist(beta_post[, j], breaks = 40, main = paste("Posterior: Beta", j),
       xlab = "", col = "lightblue", border = "white")
}
hist(sigma2_post, breaks = 40, main = "Posterior: Sigma^2",
     xlab = "", col = "lightblue", border = "white")
par(mfrow = c(1, 1))

# Predict on test set using posterior mean
y_pred <- X_test %*% posterior_means

# Evaluate RMSE, MAE, R-squared
rmse <- sqrt(mean((y_test - y_pred)^2))
mae <- mean(abs(y_test - y_pred))
ss_total <- sum((y_test - mean(y_test))^2)
ss_resid <- sum((y_test - y_pred)^2)
r_squared <- 1 - (ss_resid / ss_total)

# Print performance metrics
cat(sprintf("\nTest RMSE: %.4f\n", rmse))
cat(sprintf("Test MAE: %.4f\n", mae))
cat(sprintf("Test R^2: %.4f\n", r_squared))

# Calculate and print elapsed time
elapsed_time <- end_time - start_time
cat(sprintf("\nElapsed Time for INLA: %.2f seconds\n", as.numeric(elapsed_time, units="secs")))
