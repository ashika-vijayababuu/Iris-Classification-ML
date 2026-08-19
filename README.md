# 🌸 Iris Classification Using Machine Learning

## 📌 Project Overview

This project implements a machine learning classification system using the Iris dataset provided by Scikit-learn.

The objective of this project is to predict the species of an Iris flower based on four physical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Three machine learning classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

The final selected model was saved as a `.pkl` file using Python's Pickle library and was successfully loaded again to make predictions.

---

## 🎯 Objective

The main objectives of this project are:

- Load the Iris dataset from Scikit-learn
- Perform Exploratory Data Analysis (EDA)
- Analyze feature distributions
- Analyze class distributions
- Analyze relationships between features
- Check for missing values
- Preprocess the data where required
- Split the dataset into training and testing sets
- Train multiple classification models
- Evaluate model performance
- Generate predictions on unseen test samples
- Test the model using custom input samples
- Save the trained model as a `.pkl` file
- Load the saved model and make predictions

---

## 🛠️ Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

## 📊 Dataset

The Iris dataset is a well-known classification dataset available through Scikit-learn.

It contains:

- 150 samples
- 4 numerical features
- 3 target classes
- 50 samples per class
- No missing values

### Features

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

### Target Classes

| Target Value | Species |
|---:|---|
| 0 | Iris Setosa |
| 1 | Iris Versicolor |
| 2 | Iris Virginica |

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics and patterns present in the dataset.

### Dataset Inspection

The dataset was examined using:

- `head()`
- `shape`
- `columns`
- `describe()`
- `isnull().sum()`

The dataset contains 150 observations and four input features.

### Missing Values

The dataset was checked for missing values.

Result:

```text
No missing values were found.