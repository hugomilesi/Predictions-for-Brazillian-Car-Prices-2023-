import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from func import preprocess_data, get_score, model_predict, load_categories
from text_elements import *


df = pd.read_csv('data/2023_cars.csv')
df = df.drop("Unnamed: 0", axis = 1)

# page config
def main():
    st.set_page_config(
        page_title= "Brazillian Car Prices Prediction",
        page_icon= "🧊",
        layout = 'centered',
        initial_sidebar_state = 'auto',
        menu_items={
            'report a bug': 'mailto:hugomilesi@gmail.com'
        })
    
    st.header("Brazillian Car Prices Prediction")
    st.markdown(overview)

    st.title('Average Car Price Predictions for 2023')
    # Prediction chart
    fig = px.scatter(
                    x=df['avg_price_brl'], y=df['predicted_price'], labels={'x': 'Actual Average Price', 'y': 'Predicted Average Price'},
                    opacity=0.65,
                    trendline='ols', trendline_color_override='cyan'
                )
    st.plotly_chart(fig)

    st.title('Metrics')
     #score
    with st.spinner('Loading model metrics'):
        df_preprocessed = preprocess_data(df.drop('predicted_price', axis = 1))
        results = get_score(df_preprocessed)

        # display metrics table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(results.columns),
                        align='left'),
            cells=dict(values=[
                results['Model Name'],  
                results['Mean Squared Error (MSE)'], 
                results['R-squared (R2) Score'], 
                results['Explained Variance Score'],
                results['Mean Absolute Error (MAE)']
                ],
                    align='left'))
        ])
        fig.update_layout(height=250)
        st.plotly_chart(fig)

    
    st.markdown(metrics_explanation)

    # Loading catogories on cache
    category = load_categories()
    #selectionboxes
    col1, col2 = st.columns(2)

    with col1:
        year_of_reference = st.selectbox('Current Year', sorted(category['year_model']))
        car_year = st.selectbox('Model Year', sorted(category['year_model']))
        fuel_type = st.selectbox('Fuel Type', sorted(category['fuel']))
        gear_type = st.selectbox('Gear Type', sorted(category['gear']))
    with col2:
        engine_type = st.selectbox('Engine Size', sorted(category['engine_size']))
        car_brand = st.selectbox('Car Brand:', sorted(category['brand']))
        car = st.selectbox('Car Name', sorted(category['car_name']))
        month = st.selectbox('Month of Reference', sorted(category['month_of_reference']))
    # Predict button
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    if col4.button('Predict'):
        with st.spinner('Making prediction...'):
            st.success(f'Predicted price: {model_predict(year_of_reference, month, car_brand, car, fuel_type, gear_type, engine_type, car_year)}')

    

if __name__ == '__main__':
    main()