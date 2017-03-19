# Recommender-Algorithms
Some useful recommender algorithms implemented in Python including pmf, bpmf, rbm, ctr and so on.
# Dependencies
- Python 2.7 (Better on Linux)
- numpy
- scipy
- matplotlib
# Getting Started
- Datasets

The moviedata.mat file is a small subset of Netflix data, which contains 6040 users and 3952 movies. The format of the dataset is triplet as {user_id, movie_id, rating}, which includes 900000 records as training set and 100209 records as validation set.

- pmf

PMF(Probabilistic Matrix Factorization) model which scales linearly with the number of observations and more importantly, performs well on large, sparse, and very imbalanced Netflix dataset.

To run the code just type:
```
python pmf.py
```
The error curve on training set and validation set is as follows:
![](https://github.com/cszachary/Recommender-Algorithms/blob/master/pmf/plot.png)
