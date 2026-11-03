# STAT 636 | Project Proposal

**Final Project 9**  
**Authors:** John Whitney, Josh Boehm, Neil Roche

## Summary

Using the GitHub Social Network dataset, we will determine whether web developers and machine learning developers have measurably different positions or connection patterns within the GitHub social network.

The nodes are GitHub developers (derived from the job title), and the edges represent the mutual follower relationships between them. Each developer has node features that include location, repositories starred, employer, and email address. There are no edge features.

These data were originally used to predict whether the GitHub user is a web or a machine learning developer; however, we plan to use these data to investigate group differences in the GitHub network, such as centrality and homophily, and evaluate whether a developer's network characteristics provide predictive power beyond user attributes using the methods from the course and Kolaczyk's *Statistical Analysis of Network Data with R*.

## Preliminary Ideas

More specifically, we plan to explore the following questions:

### 1. Do machine learning and web developers differ in their average number of connections?

Our goal would be to perform exploratory data analysis, including network summary statistics (mean, median, standard deviation, degree distribution plots) and hypothesis testing to test whether degree distributions differ between web and machine learning developers.

### 2. Do the two groups differ in clustering, centrality, or other network characteristics?

We plan to answer this question by looking at transitivity (clustering coefficient), betweenness, and other measures, including looking at their distribution to understand differences across web and machine learning developers.

### 3. Do developers of the same type tend to connect to one another more often than would be expected by chance?

We hope to accomplish this by computing the assortativity coefficient and fitting an Exponential Random Graph Model (ERGM) to test whether homophily is greater than expected by chance. If computational limits arise with the full network, we will evaluate homophily using random permutation tests or by fitting the ERGM on a representative subgraph.

### 4. Can a developer's network characteristics help predict whether they are a web or machine learning developer?

We hope to achieve this by constructing a dataset where each node is an observation and each predictor is derived from network characteristics such as:

- Degree
- Clustering coefficient
- Betweenness centrality
- Eigenvector centrality
- Spectral embedding coordinates derived from singular value decomposition

We would then fit a classification model such as logistic regression to predict whether the node is a web or machine learning developer.

Because connected developers can influence each other, violating the standard independence assumption of traditional regression, we will interpret standard errors and significance tests with caution, recognizing potential network autocorrelation.

While web developers outnumber machine learning developers roughly 3-to-1, we can use ROC-AUC, Precision-Recall AUC (PR-AUC), or balanced accuracy as our primary evaluation metric to account for this class skew.

### 5. Does adding network information improve classification compared with using the supplied user attributes alone?

One possible structure would be to compare three different models:

| Model | Predictors |
| --- | --- |
| User attributes only | Supplied user attributes (e.g., location, employer) |
| Network characteristics only | Network-derived characteristics (e.g., degree, clustering, and centrality) |
| Combined | Both user attributes and network-derived attributes |

This would allow us to ask not only whether the groups differ, but whether the structure of a developer's social network contains useful information beyond their ordinary profile characteristics.

## Network Data

The project will utilize the **GitHub Social Network** dataset on the Stanford Large Network Dataset Collection website. It is an undirected graph that includes node features and node labels.

| Property | Value |
| --- | --- |
| Graph type | Undirected |
| Nodes | 37,700 |
| Edges | 289,003 |
| Density | 0.001 |
| Transitivity | 0.013 |

- **Source:** [GitHub Social Network dataset](https://snap.stanford.edu/data/github-social.html)
- **Paper:** [Research paper on arXiv](https://arxiv.org/abs/1909.13021)

## Final Output

The final output will be a **data analysis report** and **reproducible code**. The output will focus on our primary research question about understanding whether web and machine learning developers have different positions or connection patterns within the GitHub social network.

Moreover, we will use the methods presented in class to explore the secondary research questions, which will inform our primary research.
