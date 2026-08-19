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
```

### Class Distribution

The dataset is perfectly balanced, with 50 samples for each of the three species (Setosa, Versicolor, Virginica). This means no class imbalance handling was required.

### Feature Distributions and Relationships

Histograms, a pairplot, and boxplots were used to examine how each feature is distributed and how well the features separate the three species.

Key observations:

- **Petal length** and **petal width** are the most discriminative features — Setosa is linearly separable from the other two species on these alone.
- **Versicolor** and **Virginica** show some overlap, particularly in sepal measurements, which explains most of the misclassifications seen later.
- **Sepal width** has the weakest correlation with species and contributes the least to classification.

### Feature Correlation

A correlation heatmap was generated to examine relationships between features.

- Petal length and petal width are highly correlated with each other (~0.96).
- Sepal length is moderately correlated with petal length and petal width.
- Sepal width shows weak/negative correlation with the other features.

---

## ⚙️ Data Preprocessing

- Features (`X`) and target (`y`) were separated.
- The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to preserve class balance in both sets.
- Features were standardized using `StandardScaler` (fit on the training set only, then applied to the test set) before training Logistic Regression, to prevent data leakage.

---

## 🤖 Model Training & Results

Three classification models were trained and evaluated on the same train/test split:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |
| Random Forest | 90.00% | 90.24% | 90.00% | 89.97% |

**Best model: Logistic Regression** (tied with Decision Tree on all metrics, selected as the final model).

The confusion matrix for the final model showed that all misclassifications occurred between Versicolor and Virginica — consistent with the overlap observed during EDA. Setosa was classified with 100% accuracy across all models, since it is linearly separable from the other two species.

---

## 🔮 Prediction on Custom Sample

The final model was tested on a new, unseen flower measurement:

| Sepal Length | Sepal Width | Petal Length | Petal Width | Predicted Species |
|---:|---:|---:|---:|---|
| 5.1 | 3.5 | 1.4 | 0.2 | Setosa |

---

## 💾 Model Persistence

The trained Logistic Regression model, along with the fitted `StandardScaler`, was saved to `iris_classifier.pkl` using `pickle`. The saved file was reloaded in a separate step and used to make predictions again, confirming the model persists and reproduces results correctly.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ashika-vijayababuu/Iris-Classification-ML.git
   cd Iris-Classification-ML
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open `Iris_Classification_ML.ipynb` in Jupyter or Google Colab and run all cells.

### Loading the saved model for predictions

```python
import pickle
import pandas as pd

with open("iris_classifier.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=data["feature_names"])
scaled = scaler.transform(new_sample)
prediction = model.predict(scaled)

print(data["target_names"][prediction[0]])
```

---

## 📁 Project Structure

```
Iris-Classification-ML/
├── Iris_Classification_ML.ipynb   # Full notebook: EDA, training, evaluation
├── iris_classifier.pkl            # Saved trained model + scaler
├── requirements.txt                # Python dependencies
├── README.md
└── .gitignore
```

---

## 📈 Key Takeaways

- Petal measurements are far more informative than sepal measurements for classifying Iris species.
- Simpler models (Logistic Regression, Decision Tree) outperformed Random Forest on this dataset — likely because the dataset is small (150 samples) and the classes are nearly linearly separable, so Random Forest's added complexity didn't provide an advantage.
- All models struggled with the same pair of classes (Versicolor vs. Virginica), which reflects genuine overlap in the underlying data rather than a modeling flaw.