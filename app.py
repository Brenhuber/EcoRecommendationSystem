import streamlit as st
import pandas as pd
import scikit_learn as skl
import Surprise as sp
import os

st.set_page_config(
    page_title="EcoRec",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

primary_color = "#2e7d32"  
background_color = "#ecb658"  
sidebar_color = "#b2dfdb"  

def colored_header(text, color=primary_color):
    st.markdown(f'<h1 style="color:{color};">{text}</h1>', unsafe_allow_html=True)

# Pages

def contact():
    st.title("Contact Us")
    
def about():
    st.title("About EcoRec")
    st.write("EcoRec is a platform dedicated to promoting eco-friendly products. Our mission is to help consumers make informed choices that benefit the environment.")
    
def home():
    st.title("EcoRec")
    st.write("Welcome to EcoRec! We help you make eco-friendly choices by recommending sustainable products.")
    
    df = pd.read_csv('amazon_eco-friendly_products.csv')
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]+', '', regex=True),errors='coerce')
    df.drop_duplicates(inplace=True)
    df['material'] = df['material'].fillna('Not Listed')
    df['brand'] = df['brand'].fillna('Not Listed')
    
    def set_price_range():
        st.write("### Set Price Range")
        min_price = st.number_input("Minimum Price", min_value=0.0, value=0.0, step=1.0)
        max_price = st.number_input("Maximum Price", min_value=0.0, value=100.0, step=1.0)
        return min_price, max_price
    
    def find_top_rated_products(dataframe):
        return dataframe[(dataframe['rating'] >= 4.5) & (dataframe['reviewsCount'] > 1)].sort_values(by='rating', ascending=False)

    st.write("### Search In-Stock Products")
    search_term = st.text_input("Enter a product name or category to search:")
    selected_option = st.selectbox("Filter:", ["None", "Find Top Rated Products", "Set Price Range"])
    
    if search_term:
        filtered_df = df[df['name'].str.contains(search_term, case=False) | df['category'].str.contains(search_term, case=False)]
        if selected_option == "Find Top Rated Products":
            filtered_df = find_top_rated_products(filtered_df)
        elif selected_option == "Set Price Range":
            min_price, max_price = set_price_range()
            filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
        if not filtered_df.empty:
            st.write("### Search Results")
            st.dataframe(filtered_df[['name','category','material','brand','price','rating','reviewsCount']])
        else:
            st.write("No products found matching your criteria.")
    
    st.write("### Recommended Products")
            
pages = {
    "Home": home,
    "About": about,
    "Contact": contact
}
selected_page = st.sidebar.radio("Navigate", list(pages.keys()), index=0)
pages[selected_page]()
