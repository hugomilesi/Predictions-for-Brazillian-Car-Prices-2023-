# Brazillian Car Price Predictions
![image](https://github.com/hugomilesi/Predictions-for-Brazillian-Car-Prices-2023-/assets/71730507/6f2f6d42-9e56-40d2-b380-eddc220f2e51)
This project demonstrates the creation of a machine learning model for predicting car prices in Brazil for the year 2023. Here is a summary of the key components:

## Resources used:
- Python.
- Packages: Pandas, numpy, plotly, seaborn, sklearn, pickle,scipy , 

## File1: brazil-car-price-EDA.ipynb
- **Data Preprocessing:** Initial preprocessing steps include removing the first column, changing data types of some columns, and checking the summary statistics of numerical columns.
Hypothesis Testing:

- **Hypothesis testing:**  Performed using ANOVA (Analysis of Variance) on categorical variables to check if there are significant differences in average prices among different categories within each feature. All p-values indicate rejection of the null hypothesis, suggesting that there are significant differences.

- **Exploratory Data Analysis (EDA):**
  ![image](https://github.com/hugomilesi/Predictions-for-Brazillian-Car-Prices-2023-/assets/71730507/6cd0e93a-c16e-4abd-b683-29702750f948)
  - Distribution of numerical features (engine_size, age_years, year_model) is visualized using histograms.
  - Top 10 brands and models are visualized using bar plots.
  - Fuel distribution and gear count are visualized using a combination of count plots and pie charts.
  - Correlation between categorical features (age_years, month_of_reference, fuel, gear) and average prices is visualized using bar plots.

- **Insights from EDA:**
  - The age of the car is negatively correlated with its price.
  - Automatic cars tend to have higher prices.
  - Cars equipped with alcohol-based fuel systems are typically more affordable.
  - The year model exhibits a positive correlation with the car's price.

## File2: model_build.ipynb

- **Data Preprocessing:**
  - Categorical features with high cardinality (e.g., 'model', 'brand', 'year_model', 'month_of_reference') are target-encoded, while categorical features with low cardinality are ordinal-encoded.
  - The data is split into features (X) and the target variable (y).
  - A preprocessing pipeline is created using scikit-learn's ColumnTransformer to handle numerical and categorical features.

- **Model Training:**
  - Training four different regression models: Linear Regression, Gradient Boosting Regressor, Random Forest Regressor, and Decision Tree Regressor.
  - The models are trained on the preprocessed training data, and their performance is evaluated on the test set.

- **Model Evaluation:**
  - Metrics such as Mean Squared Error (MSE), R-squared (R2) Score, Explained Variance Score, and Mean Absolute Error (MAE) are calculated for each model.
  - The results are displayed in a table, and the Random Forest Regressor is identified as the best-performing with a R-Squared Score of **99.5**.

- **Model Testing on New Data:**
![image](https://github.com/hugomilesi/Predictions-for-Brazillian-Car-Prices-2023-/assets/71730507/dd31afa2-403c-429a-9f61-6dc49b7ec266)
  - The best-performing model (Random Forest Regressor) is loaded using the saved pipeline.
  - The model is tested on the validation dataset for the year 2023, and the predicted prices are compared to the actual prices.
  - The R-squared score for the model on the validation data is also displayed.

- **Interactive Prediction Widget:**
  - The notebook includes an interactive widget for predicting car prices based on user input.
  - Users can select values for features such as year of reference, month of reference, car brand, model, fuel type, gear type, engine size, and year model. Clicking the "Predict" button provides the predicted car price.
