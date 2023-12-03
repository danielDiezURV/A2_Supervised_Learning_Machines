# A2_Supervised_Learning_Machines



This project is an implementation of supervised learning machines for classifying bank and ring shape data. The following algorithms are used:

- Support Vector Machines (SVM)
- Backpropagation with TensorFlow
- Multiple Linear Regression (MLR) with scikit-learn

## Part 1: Selecting and analyzing the dataset
1. Import necessary libraries: Import the required Python libraries for data manipulation, visualization, and machine learning.

2. Load the dataset: Load the dataset from a CSV file.

3. Drop unused columns: Remove any columns from the dataset that are not needed for the analysis.

4. Handle missing values: For each column in the dataset, replace unknown or missing values with an appropriate value. In this case, the mode (most frequent value) of each column is used.

5. Convert categorical data to numerical: Convert categorical data in the dataset to numerical data to allow for machine learning processing. This is done using factorization, which assigns a unique numerical value to each category in a column.

6. Split the dataset into features and target variable: Separate the dataset into a matrix of feature variables and a vector of the target variable.

7. Scale the data: Scale the feature variables using MinMaxScaler to ensure all features have the same scale. This is important for many machine learning algorithms.

8. Split the dataset into training and test sets: Divide the dataset into a training set and a test set. The test set is a certain percentage (e.g., 20%) of the total dataset. This allows for evaluation of the model's performance on unseen data.

## Part 2: Selecting and initializing the model (BP, MLR, SVM)
1. Define the model: Define the machine learning model to be used. This could be any classifier or regressor depending on the problem at hand.

2. Hyperparameter tuning: Use GridSearchCV to tune the hyperparameters of the model to find the best parameters for the model.

## Part 3: Fit the data and predictions and summarize results

1. Fit the model: Fit the model to the training data. This involves passing the feature matrix and the target variable vector to the fit method of the model.

2. Evaluate the model: Evaluate the model's performance on the test data. This involves predicting the target variable for the test data and comparing the predictions to the actual values. The performance is measured using accuracy metrics.

3. Analyze the results: Analyze the results of the model evaluation. This could involve looking at the confusion matrix and ROC curve.
This analysis helps in understanding how well the model is performing and where it is making mistakes.




