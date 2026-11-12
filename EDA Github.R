# Stat 636
# Project Ideas for Network Data

# ==============================================================================
# GITHUB DATA
# ==============================================================================

library(tidyverse)
library(igraph)
library(patchwork) # for combining plots

library(jsonlite)
library(glmnet)
library(caret) # for data splitting and evaluation metrics
library(Matrix)
library(pROC)

# ------------------------------------------------------------------------------
# 1. Import Data
# ------------------------------------------------------------------------------
setwd("~/Library/CloudStorage/OneDrive-Personal/Documents/Student/TAMU Grad School/2026 Fall/Stat 636/Project")

# Create a temporary file and download the GitHub dataset zip from SNAP
temp_zip <- tempfile(fileext = ".zip")
download.file("https://snap.stanford.edu/data/git_web_ml.zip", temp_zip, mode = "wb")

# Unzip into a temporary directory
temp_dir <- tempdir()
unzip(temp_zip, exdir = temp_dir)

# List the extracted files to verify names
list.files(temp_dir, recursive = TRUE)

# ------------------------------------------------------------------------------
# 2. Load Data and Build Network
# ------------------------------------------------------------------------------

# Update filenames if your unzipped list differs slightly
edges_path <- file.path(temp_dir, "git_web_ml/musae_git_edges.csv")
targets_path <- file.path(temp_dir, "git_web_ml/musae_git_target.csv")

edges <- read_csv(edges_path)
nodes <- read_csv(targets_path)

# Build the undirected graph (follower relationships are mutual/undirected here)
g <- graph_from_data_frame(d = edges, directed = FALSE)

print(g)

# ------------------------------------------------------------------------------
# 3. Initial EDA
# ------------------------------------------------------------------------------

# Calculate safe network metrics
node_metrics <- tibble(
  id = as.integer(V(g)$name),
  degree = degree(g),
  pagerank = page_rank(g)$vector
)

# Merge network metrics with the developer target labels and attributes
# (Assuming the target table has an index/id column matching node names)
eda_df <- nodes %>%
  mutate(id = row_number() - 1) %>% # adjust index mapping if IDs match directly
  inner_join(node_metrics, by = "id")

# Quick summary of degree by developer category (e.g., 'ml_target' or 'target' column)
eda_df %>%
  group_by(ml_target) %>% # replace with the exact binary target column name
  summarise(
    mean_degree = mean(degree, na.rm = TRUE),
    median_degree = median(degree, na.rm = TRUE),
    count = n()
  )

wilcox.test(degree ~ ml_target, data = eda_df)

# ------------------------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------------------------
ggplot(eda_df, aes(x = factor(ml_target), y = degree, fill = factor(ml_target))) +
  geom_boxplot() +
  scale_y_log10() +
  labs(
    title = "Degree Centrality by Developer Specialization",
    subtitle = "GitHub Social Network",
    x = "Developer Type (0 = Web, 1 = Machine Learning)",
    y = "Degree Centrality (Log Scale)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# ------------------------------------------------------------------------------
# 4b. Visualization of Whole Network (TIME CONSUMING TO RUN)
# ------------------------------------------------------------------------------

print("Are you sure you want to run this?")
print("I am repeating this for when I press Enter one too many times.")
print("Are you really sure?")
print("What happens next is on you.")

# 1. Map ml_target to the vertices of g using the node IDs
vertex_ids <- as.integer(V(g)$name)
target_vector <- eda_df$ml_target[match(vertex_ids, eda_df$id)]

# 2. Assign colors based on the target value 
# (e.g., Light blue for Web developers [0], Orange for Machine Learning developers [1])
V(g)$color <- ifelse(target_vector == 1, "orange", "skyblue")

# 3. Plot the graph with the new node colors and clean up the look
plot(g, 
     vertex.label = NA,      # Hides cluttered labels
     vertex.size = 3,        # Adjusts node size
     vertex.color = V(g)$color,
     edge.color = "gray85")  # Optional: makes edges subtle so colors stand out

# 4. Add a legend to explain the colors
legend("topright", 
       legend = c("Web Developer (0)", "Machine Learning (1)"), 
       col = c("skyblue", "orange"), 
       pch = 19, 
       pt.cex = 1.5, 
       bty = "n")

# ------------------------------------------------------------------------------
# 5. Questions to Answer
# ------------------------------------------------------------------------------

# ==================================================================
# 1. Do machine learning and web developers differ in their average
#    number of connections?
# ==================================================================
# Plan: perform EDA using network summary stats (mean, median, sd,
# degree distribution plots) and hypothesis testing

# Summary statistics by developer specialization
eda_df %>%
  group_by(ml_target) %>%
  summarise(
    n = n(),
    mean_connections   = mean(degree, na.rm = TRUE),
    sd_connections     = sd(degree, na.rm = TRUE),
    median_connections = median(degree, na.rm = TRUE),
    iqr_connections    = IQR(degree, na.rm = TRUE)
  )

# Wilcoxon Rank-Sum / Mann-Whitney U Test - primary choice, heavy tailed
wilcox.test(degree ~ ml_target, data = eda_df)

# Welch's Two-Sample t-test - secondary choice to validate
# log transformed for normality
t.test(log1p(degree) ~ ml_target, data = eda_df)

# Result: Reject the null, yes they are statistically different

# ==================================================================
# 2. Do the two groups differ in clustering, centrality, or other
#    network characteristics?
# ==================================================================
# Plan: Look at transitivity (clustering coefficient), betweenness, etc

# Calculate network metrics for every node
node_metrics <- tibble(
  id = as.integer(V(g)$name),
  degree = degree(g),
  pagerank = page_rank(g)$vector,
  clustering = transitivity(g, type = "local", isolates = "zero"),
  betweenness = betweenness(g, directed = FALSE, normalized = TRUE),
  eigenvector = eigen_centrality(g)$vector
)

# Merge back into eda_df
eda_df <- nodes %>%
  mutate(id = row_number() - 1) %>% # ensure ID mapping aligns
  inner_join(node_metrics, by = "id")

# Compare medians and means across all network metrics
eda_df %>%
  group_by(ml_target) %>%
  summarise(
    n = n(),
    med_degree      = median(degree),
    med_clustering  = median(clustering, na.rm = TRUE),
    med_pagerank    = median(pagerank),
    med_betweenness = median(betweenness),
    med_eigenvector = median(eigenvector)
  )

# Statistical Tests
# 1. Local Clustering Coefficient
wilcox.test(clustering ~ ml_target, data = eda_df)

# 2. PageRank Centrality
wilcox.test(pagerank ~ ml_target, data = eda_df)

# 3. Betweenness Centrality
wilcox.test(betweenness ~ ml_target, data = eda_df)

# 4. Eigenvector Centrality
wilcox.test(eigenvector ~ ml_target, data = eda_df)

# Visualizations
p1 <- ggplot(eda_df, aes(x = factor(ml_target), y = clustering, fill = factor(ml_target))) +
  geom_boxplot() + labs(title = "Local Clustering", x = "Developer Type", y = "Clustering Coeff") + theme_minimal() + theme(legend.position = "none")

p2 <- ggplot(eda_df, aes(x = factor(ml_target), y = pagerank, fill = factor(ml_target))) +
  geom_boxplot() + scale_y_log10() + labs(title = "PageRank (Log Scale)", x = "Developer Type", y = "PageRank") + theme_minimal() + theme(legend.position = "none")

(p1 | p2)

# Result: Reject the null, yes they are statistically different

# ==================================================================
# 3. Do developers of the same type tend to connect to one another
#    more often than would be expected by chance?
# ==================================================================
# Plan: Compute assortativity coefficient and fit an ERGM to test
# whether homophily is greater than expected by chance.


# ==================================================================
# 3. Do developers of the same type tend to connect to one another
#    more often than would be expected by chance?
# ==================================================================
# Plan:
# 1) Summarize connections within and between developer types.
# 2) Calculate categorical assortativity.
# 3) Use a permutation test to assess whether assortativity is
#    greater than expected under randomly assigned developer labels.

# ------------------------------------------------------------------
# Align developer labels with the graph's vertex order
# ------------------------------------------------------------------

# Use actual node IDs rather than assuming table rows match vertices.
stopifnot(
  !anyDuplicated(nodes$id),
  !is_directed(g),
  is_simple(g)
)

node_order <- match(V(g)$name, as.character(nodes$id))
developer_type <- nodes$ml_target[node_order]

stopifnot(
  !anyNA(node_order),
  !anyNA(developer_type),
  all(developer_type %in% c(0, 1)),
  length(unique(developer_type)) == 2
)

# igraph's nominal assortativity uses integer categories starting at 1.
# Here: 1 = Web, 2 = Machine Learning.
developer_type <- as.integer(developer_type) + 1L

# ------------------------------------------------------------------
# Summarize connections within and between developer types
# ------------------------------------------------------------------

# Each row gives the vertex indices at the two ends of an edge.
edge_endpoints <- as_edgelist(g, names = FALSE)

type_from <- developer_type[edge_endpoints[, 1]]
type_to   <- developer_type[edge_endpoints[, 2]]

edge_category <- ifelse(
  type_from == 1 & type_to == 1, "Web-Web",
  ifelse(type_from == 2 & type_to == 2, "ML-ML", "Web-ML")
)

mixing_summary <- tibble(
  connection = factor(
    edge_category,
    levels = c("Web-Web", "Web-ML", "ML-ML")
  )
) %>%
  count(connection, .drop = FALSE) %>%
  mutate(percent = 100 * n / sum(n))

print(mixing_summary)

same_type_fraction <- mean(type_from == type_to)

cat(
  "Percentage of edges connecting the same developer type:",
  round(100 * same_type_fraction, 2), "%\n"
)

# A high same-type percentage alone does not establish homophily:
# the two developer groups have different sizes.

# ------------------------------------------------------------------
# Calculate observed categorical assortativity
# ------------------------------------------------------------------

observed_assortativity <- assortativity_nominal(
  g,
  types = developer_type,
  directed = FALSE
)

cat("Observed assortativity:", observed_assortativity, "\n")

# Positive assortativity indicates a tendency toward same-type
# connections; negative assortativity indicates cross-type mixing.

# ------------------------------------------------------------------
# Permutation test
# ------------------------------------------------------------------
# H0: Developer labels are exchangeable across nodes, conditional on
#     the observed network and the number of developers in each group.
# H1: Observed assortativity is greater than expected under H0.
#
# Keep all edges fixed and shuffle labels. This preserves the network
# structure and group sizes, but not group-specific degree patterns.

set.seed(636)

B <- 999L # Increase to 9999 for the final analysis.

permuted_assortativity <- replicate(B, {
  shuffled_type <- sample(developer_type)

  assortativity_nominal(
    g,
    types = shuffled_type,
    directed = FALSE
  )
})

# One-sided Monte Carlo p-value, including the observed assignment.
permutation_p <- (
  1 + sum(permuted_assortativity >= observed_assortativity)
) / (B + 1)

homophily_results <- tibble(
  observed_assortativity = observed_assortativity,
  mean_permuted_assortativity = mean(permuted_assortativity),
  excess_over_permutation_mean =
    observed_assortativity - mean(permuted_assortativity),
  same_type_percent = 100 * same_type_fraction,
  permutations = B,
  p_value = permutation_p
)

print(homophily_results, width = Inf)

# ------------------------------------------------------------------
# Visualize the permutation distribution
# ------------------------------------------------------------------

ggplot(
  tibble(assortativity = permuted_assortativity),
  aes(x = assortativity)
) +
  geom_histogram(
    bins = 40,
    fill = "skyblue",
    color = "white"
  ) +
  geom_vline(
    xintercept = observed_assortativity,
    color = "firebrick",
    linewidth = 1
  ) +
  labs(
    title = "Do Developers Connect Preferentially Within Their Type?",
    subtitle = paste0(
      "Observed assortativity = ",
      round(observed_assortativity, 3),
      "; permutation p = ",
      format.pval(permutation_p, digits = 3)
    ),
    x = "Categorical assortativity",
    y = "Number of permutations",
    caption = paste(
      "Bars: randomly shuffled developer labels.",
      "Red line: observed labels. Network held fixed."
    )
  ) +
  theme_minimal()

# ------------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------------

if (observed_assortativity > 0 && permutation_p < 0.05) {
  cat(
    "\nThere is evidence of same-type mixing: observed positive",
    "assortativity exceeds what is expected when developer labels",
    "are randomly assigned to the fixed network.\n"
  )
} else {
  cat(
    "\nThis analysis does not establish positive same-type mixing",
    "beyond the random-label baseline at the 5% significance level.\n"
  )
}

# This is evidence of association, not proof that specialization
# causes developers to connect. Other characteristics may contribute.
#
# This test does not control for differences in degree between groups.
# A possible follow-up is to permute labels within degree groups.


# If full network is too complicated, use random permutation tests or
# fitting ERGM on a representative subgraph.

# ==================================================================
# 4. Can a developer’s network characteristics help predict whether
#    they are a web or machine learning developer?
# ==================================================================
# Plan:
# 1) Construct data where each node is an observation and each
# predictor is derived from network characteristics such as degree,
# clustering coefficient, betweenness centrality, eigenvector
# centrality, and spectral embedding coordinates derived from singular
# value decomposition.
# 2) Fit a classification model to predict whether the node is a web
# or ML dev.

# ==================================================================
# 5. Does adding network information improve classification compared
#    with using the supplied user attributes alone?
# ==================================================================
# Plan: Compare 3 models
# 1) Using only supplied user attributes (location, employer)
# 2) Using only network-derived characteristics (degree, clustering, etc)
# 3) Using both supplied and network-derived characteristics

#########
# Get User Attributes
#########
# Path to the features JSON file
features_path <- file.path(temp_dir, "git_web_ml/musae_git_features.json")

# Load the JSON file (it represents a list where each element corresponds to a node ID)
node_features <- fromJSON(features_path)

# Example: Check the feature indices active for the first user
print(node_features[[1]])


# Create Feature Matrix of User Supplied Data
# 'node_features' is list loaded from jsonlite::fromJSON()
# Find the total number of unique features (the max index across all users)
max_feature_index <- max(unlist(node_features))

# Create a sparse binary matrix (Rows = Users, Columns = Features)
# Note: R lists are 1-indexed, but JSON indices are often 0-indexed or 1-indexed. 
# If max() corresponds to 4004, it's 0-indexed (so add 1) or 1-indexed (if max is 4005).
feature_matrix <- sparseMatrix(
  i = rep(seq_along(node_features), lengths(node_features)),
  j = unlist(node_features) + 1, # Add 1 if JSON indices start at 0
  x = 1,
  dims = c(length(node_features), max_feature_index + 1)
)

#########
# Build Model 1 - Only User Supplied Data
#########
# Step 1: Prepare Dataset
# Ensure feature_matrix rows match the order of eda_df
# (Assuming eda_df is ordered by node ID 0 to N-1)
y <- eda_df$ml_target

# Set a random seed for reproducibility
set.seed(42)

# Create a 70/30 train/test split
train_index <- createDataPartition(y, p = 0.7, list = FALSE)

X_train <- feature_matrix[train_index, ]
y_train <- y[train_index]

X_test <- feature_matrix[-train_index, ]
y_test <- y[-train_index]

# Step 2: Train Model 1 (Penalized Logistic Regression)
# Train a logistic regression model with Lasso penalty (alpha = 1)
cv_model <- cv.glmnet(
  x = X_train, 
  y = y_train, 
  family = "binomial", 
  alpha = 1,          # 1 = Lasso (performs feature selection)
  type.measure = "auc"# Optimize for Area Under the ROC Curve
)

# View the optimal lambda value
print(cv_model$lambda.min)

# Step 3: Evaluate Model 1
# Predict probabilities on the test set
test_probs <- predict(cv_model, newx = X_test, s = "lambda.min", type = "response")

# Calculate ROC-AUC
roc_obj <- roc(y_test, as.vector(test_probs))
auc_value <- auc(roc_obj)

cat("Model 1 Test ROC-AUC:", auc_value, "\n")

#########
# Build Model 2 - Only Network Data
#########
# Step 1: Create the Network Metrics Matrix
# Extract network metrics into a matrix
network_metrics_matrix <- eda_df %>%
  dplyr::select(degree, pagerank, clustering, betweenness, eigenvector) %>%
  as.matrix()

# Standardize the network metrics (crucial for regularized regression / penalization)
network_metrics_matrix <- scale(network_metrics_matrix)

# Align train/test sets using the same indices from Model 1
X_net_train <- network_metrics_matrix[train_index, ]
X_net_test  <- network_metrics_matrix[-train_index, ]

# Step 2: Train and Evalue Model 2
# Train Model 2
set.seed(42)
cv_model_net <- cv.glmnet(
  x = X_net_train, 
  y = y_train, 
  family = "binomial", 
  alpha = 1, 
  type.measure = "auc"
)

# Test and evaluate
test_probs_net_2 <- predict(cv_model_net, newx = X_net_test, s = "lambda.min", type = "response")
roc_net_2 <- roc(y_test, as.vector(test_probs_net_2))
auc_net_2 <- auc(roc_net_2)

cat("Model 2 (Network Only) Test ROC-AUC:", auc_net_2, "\n")

#########
# Build Model 3 - Combined Attributes + Network Data
#########
# Step 1: Combine the matrices
combined_matrix <- cbind(feature_matrix, network_metrics_matrix)

X_comb_train <- combined_matrix[train_index, ]
X_comb_test  <- combined_matrix[-train_index, ]

# Step 2: Train Model 3
set.seed(42)
cv_model_comb <- cv.glmnet(
  x = X_comb_train,  
  y = y_train,  
  family = "binomial",  
  alpha = 1,  
  type.measure = "auc"
)

# Step 3: Test and evaluate Model 3
test_probs_net_3 <- predict(cv_model_comb, newx = X_comb_test, s = "lambda.min", type = "response")
roc_net_3 <- roc(y_test, as.vector(test_probs_net_3))
auc_net_3 <- auc(roc_net_3)

cat("Model 3 (Combined) Test ROC-AUC:", auc_net_3, "\n")


#########
# Plot ROC Curves and AUCs
#########

# 1. Plot Model 1 (Attributes Only) in blue
plot(roc_obj, col = "blue", main = "Model Comparison: ROC Curves", lwd = 2)

# 2. Add Model 2 (Network Only) in red
lines(roc_net_2, col = "red", lwd = 2)

# 3. Add Model 3 (Combined) in dark green
lines(roc_net_3, col = "darkgreen", lwd = 2)

# 4. Add a legend in the bottom right corner showing the AUC values
legend("bottomright", 
       legend = c(
         paste("Model 1: Attributes (AUC =", round(auc_value, 3), ")"),
         paste("Model 2: Network (AUC =", round(auc_net_2, 3), ")"),
         paste("Model 3: Combined (AUC =", round(auc_net_3, 3), ")")
       ), 
       col = c("blue", "red", "darkgreen"), 
       lwd = 2, 
       bty = "n")

# Conclusions:
# Model 1 (Attributes Only - Blue, AUC = 0.899)
# Model 2 (Network Only - Red, AUC = 0.713)
# Model 3 (Combined - Dark Green, AUC = 0.902)

# Model 1 shows that profile attributes alone are strong predictors.
# Model 2 shows that network connectivity and centrality carry real-world
# predictive signal, but are less powerful on their own than metadata.
# Model 3 is slightly better than Model 1, showing that adding network
# structure provides complementary information and improves classification.

