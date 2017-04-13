# Recommender-Algorithms
Some useful recommender algorithms implemented in Python including pmf, bpmf, rbm, ctr and so on.
# Dependencies
- Python 2.7 (Better on Linux)
- numpy
- scipy
- matplotlib
# Getting Started
## Datasets

The moviedata.mat file is a small subset of Netflix data, which contains 6040 users and 3952 movies. The format of the dataset is triplet as {user_id, movie_id, rating}, which includes 900000 records as training set and 100209 records as validation set.

## PMF

PMF(Probabilistic Matrix Factorization) model which scales linearly with the number of observations and more importantly, performs well on large, sparse, and very imbalanced Netflix dataset.

To run the code just type:
```
python pmf.py
```
The error curve on training set and validation set is as follows:
![](https://github.com/cszachary/Recommender-Algorithms/blob/master/pmf/plot.png)

## BPR

BPR(Bayesian Personlized Ranking) model is the maximum posterior estimator derived from a Bayesian analysis of the problem. The learning method is based on stochastic gradient descent with bootstrap sampling. This is a different approach by using item pairs as training data and optimize for correctly ranking item pairs instead of scoring single items as this better represents the problem than just replacing missing values with negative ones.

Due to our model directly optimizing for ranking pairs, we omit the rating information here, that is, regarding the rating item as positive samples and others as the negative samples. We try to reconstruct for each user pairs of "perference". If an item i has been viewed by user u, then we assume that the user prefers this item over all other non-observed items.

To run the code just type:
```
python bpr.py
```
