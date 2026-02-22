# Tensor-Based Recommender System for Board Games

A Python project implementing a **tensor factorization recommender** for board games using user ratings and game metadata.  
Unlike classical matrix factorization, this model uses Tucker decomposition to incorporate multiple game features into recommendations.

---

## Problem
Traditional collaborative filtering only considers user–item interactions.  
This project extends it by modeling interactions as a multi-dimensional tensor:
$ \hat r = S \times_1 U \times_2 M \times_3 C_1 \times_4 C_2 \dots $

Where:  

- **U** – user latent factors  
- **M** – game latent factors  
- **C₁, C₂…** – feature latent factors (e.g., year bin, popularity bin)  
- **S** – core interaction tensor  

This allows capturing higher-order interactions between users, games, and features.


## Dataset
- **BoardGameGeek reviews dataset** (~26M reviews)  
- **Game metadata**: year, number of ratings, min/max players, etc.  

**Features used in the tensor**:

- `yearpublished_bin`  
- `usersrated_bin`  
- `minplayers`  
- `maxplayers`  

Continuous variables are discretized into bins using functions built in data_prep.py. Our code is suitable for any number of features you may wish to select, although it will effect training time significantly.

## Model Architecture

- Tucker tensor decomposition  
- L2 regularization  
- Gradient descent with learning rate decay  
- Gradient clipping  


## Evaluation
- Test set is filtered to include only users and games present in the training set.  
- Performance metric: **RMSE**.


In our notebook 'datafortraining.ipynb' is example for data preparation, training and evaluating a model.

```bash
pip install numpy pandas
