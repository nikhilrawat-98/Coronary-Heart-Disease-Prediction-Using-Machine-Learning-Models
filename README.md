# Coronary Heart Disease Prediction Using Machine Learning Models

This project focuses on predicting Coronary Heart Disease (CHD) using machine learning models. The aim is to develop an effective model for early CHD diagnosis using clinical and lifestyle data. Several machine learning classifiers, including Logistic Regression, Support Vector Machines (SVM), Quadratic Discriminant Analysis (QDA), and others, were applied and compared.

## Project Overview

- **Objective**: Predict the presence of CHD using machine learning models based on clinical and lifestyle data.
- **Data Source**: The dataset consists of 462 entries and 10 features, including variables like systolic blood pressure (SBP), tobacco use, low-density lipoprotein (LDL) cholesterol, adiposity, and family history.
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Statsmodels, Imbalanced-learn

## Dataset

The dataset includes 462 observations with 9 clinical and lifestyle features as predictors and 1 target variable for CHD prediction. Data features include systolic blood pressure, tobacco use, LDL cholesterol, family history, age, and others.

### Dataset Columns:
- `sbp`: Systolic Blood Pressure
- `tobacco`: Tobacco usage (in lifetime pack-years)
- `ldl`: Low-Density Lipoprotein Cholesterol (LDL)
- `adiposity`: Body Fat Percentage
- `famhist`: Family History of Heart Disease
- `typea`: Type-A Behavior
- `obesity`: Body Mass Index (BMI)
- `alcohol`: Alcohol Consumption (grams per day)
- `age`: Age of the patient
- `chd`: Coronary Heart Disease Event (1 = CHD, 0 = No CHD)

## File Structure

- **`Code.ipynb`**: The Jupyter notebook containing the code for data pre-processing, exploratory data analysis (EDA), and model training.
- **`Code.pdf`**: PDF version of the Jupyter notebook for easier viewing.
- **`Report.pdf`**: Detailed report discussing the methodology, model performance, and results.
- **`heart-disease.csv`**: Raw dataset containing clinical and lifestyle features.
- **`preprocessed_data.csv`**: Preprocessed dataset saved after data cleaning and transformations.
- **`requirements.txt`**: Python dependencies for the project.

## Model and Performance

Several machine learning classifiers were applied and evaluated. Notably:
- **Quadratic Discriminant Analysis (QDA)**:
  - Best cross-validation score: 0.7209
  - Accuracy: 79.57%
  - Precision: 80.07%
  - Recall: 79.57%
  - Specificity: 81.36%

- **Support Vector Machine (SVM)**:
  - Best cross-validation score: 0.6968
  - Accuracy: 78.49%
  - Precision: 78.28%
  - Recall: 78.49%
  - Specificity: 84.75%

- **Logistic Regression (Ridge Penalty)**:
  - Accuracy: 75.27%
  - Precision: 76.30%
  - Recall: 75.27%
  - Specificity: 76.27%

Other models such as K-Nearest Neighbours (KNN), Naive Bayes, and Multilayer Perceptron (MLP) were also explored, with performance metrics in the range of 72-76%.

## Usage

The Jupyter notebook demonstrates the following:
1. **Data Pre-processing**:
    - Handling missing values.
    - Encoding categorical variables.
    - Feature scaling and balancing the dataset using SMOTE.
2. **Exploratory Data Analysis (EDA)**:
    - Descriptive statistics and correlation analysis.
    - Visualization of class distributions, feature importance, and more.
3. **Model Training and Evaluation**:
    - Training multiple classifiers, including Logistic Regression, SVM, QDA, and others.
    - Hyperparameter tuning using GridSearchCV.
    - Evaluation using metrics such as accuracy, precision, recall, F1 score, and specificity.

## Installation

To run the project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CHD-Prediction.git
    cd CHD-Prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```

## Acknowledgements

This project was developed as part of an MSc in Business Analytics at Bayes Business School.

## License

MIT License
