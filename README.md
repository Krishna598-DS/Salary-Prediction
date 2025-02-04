# Salary-Prediction
Using ML Models
Salary Prediction Model Using Machine Learning
Overview
This project focuses on predicting the Mean Salary of employees based on various job-related attributes such as job title, company, location, sector, and other factors. The primary goal is to build a machine learning model that can accurately predict employee salaries based on historical data. The solution involves data preprocessing, feature engineering, model training, and evaluation using multiple regression models. The final model is serialized for easy deployment.

Problem Statement
The objective is to predict the Mean Salary for various job profiles in the dataset, utilizing attributes like job titles, company details, location, sector, job profiles, and other relevant features. This is a regression problem where the target variable is the continuous value of Mean Salary.

Target Variable:
Mean Salary: The average salary of employees for each job profile in USD.
Input Features:
Job: Job title
Jobs_Group: Job categories
Profile: Job profile (Lead, Senior, Junior, or none)
Remote: Work flexibility (Remote, Hybrid, or none)
Company: Company name
Location: Company location
City: City where the company operates
State: State of the company
Frequency_Salary: Salary frequency (year, month, week, day)
Skills: Skills required for the job
Sector: The industry sector the company belongs to
Sector_Group: Grouped industry sectors
Revenue: Company’s revenue size
Employee: Number of employees in the company
Company_Score: Company rating score
Reviews: Number of reviews for the company
Director: Director's name
Director_Score: Rating of the company director
URL: Company’s website
Dataset
The dataset contains job-related information, including job profiles, company details, location, and salary information. The dataset is stored in a CSV format and is divided into training and test sets. The Mean Salary column is the target variable for training, and is used to predict the salary in the test dataset.

Data Preprocessing
The dataset underwent extensive preprocessing to handle missing values, drop irrelevant columns, and encode categorical features. Below are the main steps involved:

1. Missing Values Handling:
Categorical Columns: Missing values in columns like City, State, Company, and Location were filled using the mode (most frequent value) or based on relationships between columns (e.g., filling State based on City).
Numerical Columns: Missing values in Company_Score were filled using the median, while missing values in Reviews were filled with zero.
2. Dropping Irrelevant Columns:
Columns with high missing values or low relevance to salary prediction were dropped, including Profile, Remote, Revenue, Employee, Director, Director_Score, and URL.
3. Feature Encoding:
Target Encoding: High-cardinality categorical features like Job, Company, Location, and City were encoded using target encoding.
One-Hot Encoding: Moderate-cardinality categorical features like Jobs_Group, State, Frequency_Salary, Sector, and Sector_Group were converted into binary features using one-hot encoding.
TF-IDF Encoding: The Skills column, which contains a list of skills, was transformed using the TF-IDF (Term Frequency-Inverse Document Frequency) method to represent the relevance of each word in relation to the entire dataset.
4. Reindexing:
The training and test datasets were reindexed to ensure consistency between the datasets when making predictions.
Model Selection & Training
Three regression models were tested to predict the Mean Salary:

Linear Regression: A baseline model that assumes a linear relationship between the features and target variable.
Random Forest Regressor: An ensemble learning method that uses multiple decision trees to make predictions. Known for handling non-linear relationships well and being robust against overfitting.
XGBoost: A gradient boosting algorithm that builds decision trees sequentially, optimizing accuracy through boosting.
After evaluating the models, the Random Forest Regressor performed the best based on evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

Model Evaluation
The following metrics were used to evaluate the performance of the models:

Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
Mean Squared Error (MSE): Measures the average squared differences between predicted and actual values.
R-squared (R²): Represents the proportion of variance in the target variable explained by the independent variables.
The Random Forest Regressor outperformed the other models with the lowest error and highest R² score.

Model Serialization
The best-performing model, Random Forest Regressor, was serialized using the Pickle library to save the trained model and load it for future predictions without retraining. This allows for easy deployment and sharing of the model.

Conclusion
This project demonstrates how to process job-related data and use machine learning models for salary prediction. The final model predicts salaries based on various job and company attributes with high accuracy.

Future Improvements
Hyperparameter Tuning: Fine-tune the model's hyperparameters to further improve its performance.
Feature Engineering: Explore additional features, such as job experience or education level, to improve predictions.
Model Deployment: Deploy the trained model as a web service to allow real-time predictions.
Key Takeaways
Data preprocessing and feature encoding are critical steps in building effective machine learning models.
Model evaluation with multiple metrics helps in selecting the best model.
Pickling the model allows saving and reusing it, which saves time and resources.
Installation & Setup
Prerequisites
To run this project, ensure you have the following Python libraries installed:

pandas
numpy
scikit-learn
xgboost
pickle
