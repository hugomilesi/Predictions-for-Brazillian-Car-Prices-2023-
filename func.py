import pandas as pd
import pickle
from text_elements import *
import json
import streamlit as st
# preprocess_data

from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
# metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


#load model, encoders and scalers
model = pickle.load(open('model/model.sav', 'rb'))
scaler = pickle.load(open('model/scaler.pickle', 'rb'))
encoder = pickle.load(open('model/target-encoder.pickle', 'rb'))


def preprocess_data(data):
    data['age_years'] = data['year_of_reference'] - data['year_model']
    data['age_years'] = data['age_years'].replace(-1, 0)

    if 'avg_price_brl' in data.columns:
        new_x, new_y = data.drop('avg_price_brl', axis = 1), data['avg_price_brl']

        # get numerical and categorical data
        cat_data = ['month_of_reference', 'brand', 'model']
        num_data = ['engine_size', 'year_model', 'age_years', 'year_of_reference']
        # binary encoding gear
        new_x['gear'] = new_x['gear'].replace({'manual': 0, 'automatic': 1})
        # ordinal encoding fuel
        fuel_type = {'Gasoline': 0, 'Diesel': 1, 'Alcohol': 2}
        new_x['fuel'] = new_x['fuel'].replace(fuel_type)
        # applying the fittef target encoder to new data
        new_x[cat_data] = encoder.transform(new_x[cat_data])
        # scaling
        new_x[num_data] = scaler.transform(new_x[num_data])

        return pd.concat([new_x, new_y], axis = 1)
    
    else:
        new_x = data
        # get numerical and categorical data
        cat_data = ['month_of_reference', 'brand', 'model']
        num_data = ['engine_size', 'year_model', 'age_years', 'year_of_reference']
        # binary encoding gear
        new_x['gear'] = new_x['gear'].replace({'manual': 0, 'automatic': 1})
        # ordinal encoding fuel
        fuel_type = {'Gasoline': 0, 'Diesel': 1, 'Alcohol': 2}
        new_x['fuel'] = new_x['fuel'].replace(fuel_type)
        # applying the fittef target encoder to new data
        new_x[cat_data] = encoder.transform(new_x[cat_data])
        # scaling
        new_x[num_data] = scaler.transform(new_x[num_data])

        return new_x



@st.cache_data
def get_score(data):
    """Score the model with new data and create a dataframe with the metrics"""
    x, y = data.drop('avg_price_brl', axis = 1), data['avg_price_brl']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

    # Making predictions
    y_pred = model.predict(X_test)  
    # Calculate various metrics
    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    explained_variance = round(explained_variance_score(y_test, y_pred), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)

    model_results = {
        "Model Name" : str(model).split('(')[0],
        "Mean Squared Error (MSE)" : mse,
        "R-squared (R2) Score" : r2,
        "Explained Variance Score" : explained_variance,
        "Mean Absolute Error (MAE)": mae,
    }

    # Fit the model to the training data

    # Create a pandas DataFrame from the results
    results_df = pd.DataFrame(model_results, index = [0])
    #results_df = results_df.iloc[:, 0].rename('results')

    return results_df

@st.cache_data
def model_predict(year_of_reference, month_of_reference, brand, car_model, fuel, gear, engine_size, year_model): 

    input_data = {
    'year_of_reference': year_of_reference,
    'month_of_reference': month_of_reference,
    'brand': brand,
    'model': car_model,
    'fuel': fuel,
    'gear': gear,
    'engine_size': engine_size,
    'year_model': year_model,
    }
    
    input_df = pd.DataFrame(input_data, index = [0])
    result = preprocess_data(input_df)
    #input_df['age_years'] = input_df['year_of_reference'] - input_df['year_model']    

    return model.predict(result)[0]



@st.cache_data
def load_categories():
    filename = 'categories.json'
    with open(filename, 'r') as json_file:
        category = json.load(json_file)
    return category
  