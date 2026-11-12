# ==============================================================================
# 4. Can a developer’s network characteristics help predict whether
#    they are a web or machine learning developer?
# ==============================================================================
# Plan:
# 1) Construct data where each node is an observation and each
# predictor is derived from network characteristics such as degree,
# clustering coefficient, betweenness centrality, eigenvector
# centrality, and spectral embedding coordinates derived from singular
# value decomposition.
# 2) Fit a classification model (regularized logistic regression (Lasso)) to predict whether the node is a web
# or ML dev.

# ------------------------------------------------------------------------------
# Load Required Libraries
# ------------------------------------------------------------------------------
library(tidyverse)
library(igraph)
library(Matrix)
library(rARPACK) # Efficient computation of top eigenvalues for large sparse matrices
library(glmnet)  # Penalized logistic regression
library(caret)   # Data splitting and preprocessing
library(pROC)    # ROC Curve & ROC-AUC
library(PRROC)   # Precision-Recall Curve & PR-AUC

# ------------------------------------------------------------------------------
# Import Data & Build Network Graph
# ------------------------------------------------------------------------------
file_path <- "/Users/liddleroche/Documents/TAMU/stat_636/github/"
setwd(file_path)

temp_zip <- tempfile(fileext = ".zip")
download.file("https://snap.stanford.edu/data/git_web_ml.zip", temp_zip, mode = "wb")

temp_dir <- tempdir()
unzip(temp_zip, exdir = temp_dir)

edges_path  <- file.path(temp_dir, "git_web_ml/musae_git_edges.csv")
targets_path <- file.path(temp_dir, "git_web_ml/musae_git_target.csv")

edges <- read_csv(edges_path)
nodes <- read_csv(targets_path)

# Build undirected follower graph
g <- graph_from_data_frame(d = edges, directed = FALSE)
cat("Network built successfully: n =", vcount(g), "nodes, m =", ecount(g), "edges.\n")

# ------------------------------------------------------------------------------
# Spectral Analysis & Selecting Embedding Dimension "d" (Scree Plot)
# ------------------------------------------------------------------------------
cat("\n--- STEP 2: Spectral Gap Analysis ---\n")
A <- get.adjacency(g)

# Compute top 40 absolute eigenvalues using Lanczos iteration (rARPACK)
top_evals <- abs(eigs(A, k = 40)$values)

# 2a. Plot Scree Plot to inspect the spectral gap
plot(
  top_evals, 
  type = "b", 
  pch  = 19, 
  col  = "#4E79A7",
  xlab = "Dimension / Eigenvalue Index (k)",
  ylab = "Absolute Eigenvalue |lambda_k|",
  main = "Scree Plot of GitHub Network Adjacency Matrix"
)
grid(col = "gray85", lty = "dotted")

# Add visual markers for candidate cutoffs
abline(v = c(3, 5, 10), col = c("red", "green", "blue"), lty = 2, lwd = 2)
legend(
  "topright", 
  legend = c("Candidate d = 3", "Candidate d = 5", "Candidate d = 10"),
  col    = c("red", "green", "blue"), 
  lty    = 2, 
  lwd    = 2, 
  bty    = "n"
)

# COMMENT: Based on the scree plot d = 5 is a good choice for the embedding dimension that will
#          capture most of the structural variance.

# 2b. Automated Elbow Selection via Profile Likelihood (Zhu & Ghodsi Algorithm)
optimal_d <- dim_select(top_evals)
cat("Automated optimal embedding dimension (Zhu & Ghodsi): d =", optimal_d, "\n")

# Set chosen dimension for ASE (e.g., d = 5 or optimal_d)
d_chosen <- 5
cat("Selected embedding dimension for ASE: d =", d_chosen, "\n")

# ------------------------------------------------------------------------------
# Feature Extraction (Centralities + Adjacency Spectral Embedding)
# ------------------------------------------------------------------------------
cat("\n--- STEP 3: Extracting Network Features ---\n")

# 3a. Compute Topological Centralities
centrality_df <- tibble(
  id          = as.integer(V(g)$name),
  degree      = degree(g),
  pagerank    = page_rank(g)$vector,
  clustering  = transitivity(g, type = "local", isolates = "zero"),
  betweenness = betweenness(g, directed = FALSE, normalized = TRUE),
  eigenvector = eigen_centrality(g)$vector
)

# 3b. Compute d-dimensional Adjacency Spectral Embedding (ASE)
set.seed(42)
ase_res  <- embed_adjacency_matrix(g, no = d_chosen, weights = NA)
ase_df   <- as.data.frame(ase_res$X)
colnames(ase_df) <- paste0("ASE_Dim", 1:d_chosen)
ase_df$id <- as.integer(V(g)$name)

# 3c. Combine Centralities + ASE Coordinates with Target Labels
model_df <- nodes %>%
  # mutate(id = row_number() - 1) %>%
  inner_join(centrality_df, by = "id") %>%
  inner_join(ase_df, by = "id")

# ------------------------------------------------------------------------------
# Train / Test Partitioning & Standardization
# ------------------------------------------------------------------------------
cat("\n--- STEP 4: Data Partitioning & Preprocessing ---\n")

# Target variable y (binary factor)
y <- as.factor(model_df$ml_target)

# Predictor matrix X
X_raw <- model_df %>%
  dplyr::select(degree, pagerank, clustering, betweenness, eigenvector, starts_with("ASE_Dim")) %>%
  as.matrix()

# Stratified 70/30 Train/Test Split
set.seed(42)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)

X_train_raw <- X_raw[train_idx, ]
y_train     <- y[train_idx]

X_test_raw  <- X_raw[-train_idx, ]
y_test      <- y[-train_idx]

# Standardize predictors using training parameters (prevents data leakage)
preProcValues <- preProcess(X_train_raw, method = c("center", "scale"))
X_train       <- predict(preProcValues, X_train_raw)
X_test        <- predict(preProcValues, X_test_raw)

# ------------------------------------------------------------------------------
# Model Fitting (Lasso Logistic Regression)
# ------------------------------------------------------------------------------
cat("\n--- STEP 5: Fitting Regularized Model ---\n")

set.seed(42)
cv_lasso_q4 <- cv.glmnet(
  x            = X_train,
  y            = y_train,
  family       = "binomial",
  alpha        = 1,          # Lasso penalty
  type.measure = "auc"        # Optimize ROC-AUC
)

cat("Optimal Lambda (min):", cv_lasso_q4$lambda.min, "\n")

# Display Lasso non-zero coefficients
cat("\nSelected Model Coefficients:\n")
print(coef(cv_lasso_q4, s = "lambda.min"))

# ------------------------------------------------------------------------------
# Model Evaluation (ROC & Precision-Recall Curves)
# ------------------------------------------------------------------------------
cat("\n--- STEP 6: Model Evaluation ---\n")

# Predict test set probabilities
test_probs <- predict(cv_lasso_q4, newx = X_test, s = "lambda.min", type = "response")

# 6a. ROC Curve & ROC-AUC
roc_q4 <- roc(y_test, as.vector(test_probs))
auc_q4 <- auc(roc_q4)
cat("Question 4 - Test ROC-AUC:", round(auc_q4, 4), "\n")

plot(
  roc_q4, 
  col  = "red", 
  lwd  = 2, 
  main = paste0("Question 4: ROC Curve (AUC = ", round(auc_q4, 3), ")")
)
grid(col = "gray85", lty = "dotted")

# 6b. Precision-Recall Curve & PR-AUC
y_test_numeric <- as.numeric(as.character(y_test))

pr_q4 <- pr.curve(
  scores.class0 = as.vector(test_probs), 
  weights.class0 = y_test_numeric, 
  curve          = TRUE
)

cat("Question 4 - Test PR-AUC:", round(pr_q4$auc.integral, 4), "\n")

plot(
  pr_q4, 
  color = "blue", 
  main  = paste0("Question 4: Precision-Recall Curve (PR-AUC = ", round(pr_q4$auc.integral, 3), ")")
)