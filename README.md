# Banknote Authentication using Random Forest

A machine learning classification project built to practice Random Forest, exploratory data analysis, hyperparameter tuning, model evaluation, and model persistence using the Banknote Authentication dataset.

## Project Overview

This project explores a complete machine learning workflow for a binary classification problem using a Random Forest Classifier.

The main focus of the project was not only to train a model, but also to understand how Random Forest behaves with different hyperparameters, how GridSearchCV can be used for parameter tuning, how model performance can be evaluated, and how a trained model can be saved for later use.

The project includes:

- Dataset inspection
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature relationship visualization
- Train-test split
- Random Forest classification
- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Out-of-Bag (OOB) score experimentation
- Confusion matrix evaluation
- Classification report
- Misclassification and error analysis across different numbers of trees
- Feature importance inspection
- Model persistence using Joblib

## Dataset

The dataset contains **1,372 samples** and **5 columns**.

All five columns contain complete data with no missing values.

### Features

The model uses four numerical input features:

| Feature | Description |
|---|---|
| `Variance_Wavelet` | Wavelet-transformed variance feature |
| `Skewness_Wavelet` | Wavelet-transformed skewness feature |
| `Curtosis_Wavelet` | Wavelet-transformed curtosis feature |
| `Image_Entropy` | Entropy-related image feature |

### Target

| Column | Type |
|---|---|
| `Class` | Integer |

`Class` is used as the target variable for the binary classification task.

## Exploratory Data Analysis

The notebook begins with basic dataset inspection using Pandas, followed by correlation analysis and visualization.

The following were explored:

- First few records using `df.head()`
- Dataset structure and data types using `df.info()`
- Feature and target correlations using `df.corr()`
- Pairwise feature relationships
- Class-based feature relationships
- Scatter plot visualization

### Correlation Analysis

The correlation matrix produced the following relationships with the target:

| Feature | Correlation with Class |
|---|---:|
| `Variance_Wavelet` | -0.724843 |
| `Skewness_Wavelet` | -0.444688 |
| `Curtosis_Wavelet` | 0.155883 |
| `Image_Entropy` | -0.023424 |

The correlation analysis was used as part of the exploratory process to understand relationships between the available features and the target.

The notebook also showed a relatively strong negative correlation between `Skewness_Wavelet` and `Curtosis_Wavelet` of approximately **-0.786895**.

## Feature and Target Preparation

The target column was separated from the input features.

The feature matrix was created by removing the `Class` column:

`X = df.drop('Class', axis=1)`

The target was assigned as:

`y = df['Class']`

The model therefore uses four input features for prediction.

## Train-Test Split

The dataset was divided into training and testing sets using `train_test_split`.

The split used:

- `test_size = 0.15`
- `random_state = 101`

This resulted in:

- **Training samples: 1,166**
- **Testing samples: 206**

The test set was kept separate and used for the final model evaluation.

## Model

The main machine learning algorithm used in this project is the **Random Forest Classifier** from Scikit-learn.

Random Forest is an ensemble learning method that combines multiple decision trees to make a classification prediction.

The project also included experimentation with Random Forest parameters to understand their effect on model performance.

## Hyperparameter Tuning

`GridSearchCV` was used to search through different Random Forest configurations.

The parameter search included:

| Parameter | Values Tested |
|---|---|
| `n_estimators` | 64, 100, 128, 200 |
| `max_features` | 2, 3, 4 |
| `bootstrap` | True, False |
| `oob_score` | True, False |

This resulted in **48 parameter combinations**.

With the default **5-fold cross-validation** used by `GridSearchCV`, a total of **240 model fits** were performed.

### Best Parameters

The best parameter combination found by GridSearchCV was:

| Parameter | Best Value |
|---|---:|
| `n_estimators` | 64 |
| `max_features` | 2 |
| `bootstrap` | True |
| `oob_score` | False |

### Best Cross-Validation Score

The best mean cross-validation score obtained during GridSearchCV was:

**0.9966**

or approximately:

**99.66%**

The saved GridSearchCV object contains the fitted best estimator, which is a `RandomForestClassifier` with `max_features=2` and `n_estimators=64`.

## Model Evaluation

The tuned model was evaluated on the **206 unseen test samples**.

### Test Set Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.98 | 0.99 | 124 |
| 1 | 0.98 | 1.00 | 0.99 | 82 |
| **Accuracy** | | | **0.99** | **206** |
| **Macro Avg** | **0.99** | **0.99** | **0.99** | **206** |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **206** |

The reported test accuracy is approximately **99.03%**, corresponding to **204 correct predictions out of 206 test samples**.

## Confusion Matrix

The confusion matrix obtained from the test predictions was:

`[[122, 2], [0, 82]]`

This means:

- **122** Class 0 samples were correctly classified as Class 0.
- **2** Class 0 samples were classified as Class 1.
- **0** Class 1 samples were classified as Class 0.
- **82** Class 1 samples were correctly classified as Class 1.

Therefore, the model made **2 incorrect predictions** on the 206-sample test set.

## Out-of-Bag Score Experiment

In addition to GridSearchCV, the notebook included a separate Random Forest experiment using:

- `n_estimators = 64`
- `max_features = 2`
- `bootstrap = True`
- `oob_score = True`

The resulting Out-of-Bag score was:

**0.9957118353**

or approximately:

**99.57%**

This experiment was included to explore how the OOB score can be used with bootstrap-based Random Forest training.

## Number of Trees Experiment

The notebook also experimented with different values of `n_estimators`.

Random Forest models were trained for values from **1 to 199 trees**, while keeping `max_features = 2`.

For each value, the notebook recorded:

- Classification error
- Number of misclassified test samples

These values were then plotted to observe how the number of trees affected model performance.

This experiment was useful for understanding the effect of the number of estimators rather than relying only on a single Random Forest configuration.

## Feature Importance

The fitted Random Forest also provided impurity-based feature importance values.

The approximate feature importance values were:

| Feature | Importance |
|---|---:|
| `Variance_Wavelet` | 0.58 |
| `Skewness_Wavelet` | 0.20 |
| `Curtosis_Wavelet` | 0.16 |
| `Image_Entropy` | 0.06 |

Among the four features, `Variance_Wavelet` had the highest feature importance in the fitted Random Forest used for this experiment.

These values represent the model's impurity-based feature importance and are included as part of the model exploration.

## GridSearch Experiment Note

During the GridSearchCV experiment, some parameter combinations produced failed fits.

The reason was that `oob_score=True` requires `bootstrap=True`. The search included both `bootstrap=True` and `bootstrap=False` together with both OOB settings, resulting in invalid combinations such as:

`bootstrap=False` and `oob_score=True`

Out of the **240 total fits**, **60 fits failed** because of this parameter combination.

GridSearchCV assigned non-finite scores to those invalid combinations and continued the search. The best parameter combination was still identified successfully.

This was part of the experimentation process while learning Random Forest hyperparameters and GridSearchCV.

## Model Persistence

The trained GridSearchCV model was saved using Joblib.

The notebook used:

`dump(grid, 'banknote_authentication_(random_forest).joblib')`

The model can be loaded later using Joblib without repeating the training process.

The recommended repository filename is:

`banknote_authentication_rf.joblib`

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

The overall workflow followed in the notebook was:

**Dataset → Inspection → EDA → Correlation Analysis → Visualization → Feature/Target Separation → Train-Test Split → Random Forest → GridSearchCV → Best Parameters → Model Evaluation → OOB Experiment → Tree Count Experiment → Model Saving**

## Project Files

- `data_banknote_authentication.csv` — Dataset used for the project
- `banknote_authentication_random_forest.ipynb` — Jupyter Notebook containing the complete analysis and implementation
- `banknote_authentication_rf.joblib` — Saved Random Forest/GridSearchCV model
- `README.md` — Project documentation
- `requirements.txt` — Python dependencies

## What I Learned and Practiced

This project helped me practice and understand:

- How to inspect and understand a dataset before modelling
- Exploratory Data Analysis using Pandas and Seaborn
- Correlation analysis
- Feature relationship visualization
- Preparing features and target variables
- Train-test splitting
- Building a Random Forest Classifier
- Understanding Random Forest hyperparameters
- Using `n_estimators` and `max_features`
- Understanding the relationship between `bootstrap` and `oob_score`
- Hyperparameter tuning with `GridSearchCV`
- Cross-validation
- Classification metrics such as precision, recall, and F1-score
- Confusion matrix interpretation
- Out-of-Bag evaluation
- Studying model behaviour with different numbers of trees
- Inspecting feature importance
- Saving and loading trained models with Joblib

## Results Summary

| Evaluation | Result |
|---|---:|
| Dataset Size | 1,372 samples |
| Number of Features | 4 |
| Training Samples | 1,166 |
| Test Samples | 206 |
| GridSearchCV Folds | 5 |
| GridSearchCV Fits | 240 |
| Best CV Score | 99.66% |
| Test Accuracy | 99.03% |
| Correct Test Predictions | 204 / 206 |
| Misclassified Test Samples | 2 |
| OOB Score Experiment | 99.57% |

----------------------------------------
                                        
👨‍💻 Author

Siddharth Patil

This repository represents my work and learning while studying machine learning and exploring classification algorithms.

## Note

This project was created as part of my machine learning learning process. The notebook focuses on understanding Random Forest classification, hyperparameter tuning, model evaluation, OOB scoring, and model behaviour through experimentation.

The notebook contains the complete implementation and the experiments performed during the project.
