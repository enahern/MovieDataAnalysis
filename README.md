Collaborative Filtering Recommender System

This project implements a movie recommendation system using collaborative filtering on the MovieLens 100k dataset
. It explores both user-based and item-based approaches, evaluates prediction accuracy, and compares performance using RMSE.

Dataset

File: u.data (from MovieLens 100k)

Columns:

- user_id

- item_id

- rating

- timestamp

Methodology
1. Data Preparation

Load the dataset into a pandas DataFrame.

Create user–item and item–user rating matrices.

Hold out 20% of ratings for evaluation.

2. Similarity and Neighbors

Use cosine similarity to measure closeness between users or items.

For each prediction, find the k-nearest neighbors (k=10).

3. Predictions

Estimate missing ratings using the weighted average of neighbors’ ratings.

4. Evaluation

Compute RMSE (Root Mean Squared Error) to evaluate accuracy.

Compare user-based and item-based collaborative filtering.

Results

User-based CF: RMSE ≈ 1.08

Item-based CF: Lower RMSE, indicating that similar movies are better at predicting ratings than similar users.

Requirements

Python 3.8 or higher

Packages: pandas, numpy, math
