import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import os

st.set_page_config(
    page_title="EcoRec",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------- USER INTERFACE -----------------------------------

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #111112;
        color: #f5f5f5;
    }
    /* Sidebar (Navigation tab) */
    section[data-testid="stSidebar"] {
        background: #232629;
        color: #f5f5f5;
        border-right: 4px solid #FF9900;
        box-shadow: 2px 0 8px 0 #FF990033;
    }
    /* Hide default radio label in sidebar */
    [data-testid="stSidebar"] .stRadio label[for^="radio"] {
        display: none !important;
    }
    /* Modern sidebar radio buttons - neat and professional */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0;
        margin-top: 32px;
        margin-bottom: 32px;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        background: #18191a !important;
        color: #f5f5f5 !important;
        border: 2px solid transparent !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-size: 1.08rem !important;
        font-weight: 600 !important;
        margin: 0 0 12px 0 !important;
        min-height: 48px;
        box-shadow: 0 1px 4px 0 #0001;
        transition: border 0.2s, background 0.2s, color 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:last-child {
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-selected="true"],
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
        background: #FF9900 !important;
        color: #232629 !important;
        border: 2px solid #FF9900 !important;
        box-shadow: 0 2px 8px 0 #FF990044 !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        border: 2px solid #FF9900 !important;
        background: #232629 !important;
        color: #FF9900 !important;
    }
    /* Hide default radio circle */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label span:first-child {
        display: none !important;
    }
    /* Sidebar radio and headings */
    .stRadio > label, .stSidebar .st-bb, .stSidebar .st-c3 {
        color: #f5f5f5 !important;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FF9900;
        border-bottom: 2px solid #FF9900;
        padding-bottom: 2px;
    }
    hr, .stDivider {
        border-top: 2px solid #FF9900 !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #FF9900;
        color: #fff;
        border: 2px solid #FF9900;
        border-radius: 6px;
        font-weight: bold;
        transition: background 0.2s, border 0.2s;
        box-shadow: 0 2px 8px 0 #FF990044;
    }
    .stButton > button:hover {
        background-color: #b85e13;
        border: 2px solid #b85e13;
        color: #fff;
    }
    /* Inputs */
    .stTextInput > div > input, .stNumberInput input {
        background: #2d2f31;
        color: #fff;
        border: 2px solid #FF9900;
    }
    .stTextInput > div > input:focus, .stNumberInput input:focus {
        outline: 2px solid #FF9900;
        box-shadow: 0 0 0 2px #FF9900;
    }
    /* Dataframe styling */
    .stDataFrame, .stTable {
        background: #111112;
        color: #fff;
        border: 2px solid #FF9900;
        border-radius: 6px;
        box-shadow: 0 2px 8px 0 #FF990044;
    }
    /* Highlighted text and links */
    a, .stMarkdown a {
        color: #FF9900;
        text-decoration: underline;
    }
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #232629;
    }
    ::-webkit-scrollbar-thumb {
        background: #FF9900;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<div style="font-size:1.6rem;font-weight:900;letter-spacing:1px;color:#FF9900;margin-bottom:8px;margin-top:8px;text-transform:uppercase;border-bottom:3px solid #FF9900;padding-bottom:2px;width:100%;text-align:center;">Navigate</div>',
    unsafe_allow_html=True
)
st.markdown(
    '''<style>
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        align-items: center !important;
        width: 100%;
        margin-left: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        justify-content: center !important;
        min-width: 180px;
        width: 80%;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    </style>''',
    unsafe_allow_html=True
)

# ----------------------------------- END USER INTERFACE -----------------------------------

# ----------------------------------- DATA CLEANING -----------------------------------

@st.cache_data
def load_and_clean_data(path):

    df = pd.read_csv(path)

    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]+', '', regex=True), errors='coerce')
    df = df.dropna(subset=['price', 'material', 'brand', 'name', 'category', 'rating', 'reviewsCount'])

    material_copy = df['material'].str.lower()

    color_names = [
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple',
        'pink', 'brown', 'gray', 'grey', 'beige', 'ivory', 'teal', 'navy', 'gold',
        'silver', 'bronze', 'maroon', 'violet', 'indigo', 'turquoise', 'magenta',
        'lime', 'peach', 'olive', 'coral', 'aqua', 'mint', 'mustard', 'lavender',
        'tan', 'charcoal', 'burgundy', 'cream', 'amber', 'apricot', 'azure',
        'chocolate', 'copper', 'crimson', 'cyan', 'emerald', 'fuchsia', 'jade',
        'khaki', 'lemon', 'mauve', 'ochre', 'plum', 'rose', 'ruby', 'salmon',
        'sapphire', 'scarlet', 'taupe', 'topaz', 'ultramarine', 'vermilion',
        'wine', 'zinc'
    ]

    shape_names = [
        'circle', 'square', 'rectangle', 'triangle', 'oval', 'hexagon', 'octagon',
        'pentagon', 'cylinder', 'sphere', 'cube', 'cone', 'pyramid', 'diamond',
        'ellipse', 'star', 'heart', 'crescent', 'torus', 'rhombus',
        'parallelogram', 'trapezoid', 'semicircle', 'octahedron', 'tetrahedron',
        'dodecahedron', 'icosahedron', 'prism', 'cuboid'
    ]

    color_regex = r'|'.join([fr'\b{c}\b' for c in color_names])
    shape_regex = r'|'.join([fr'\b{s}\b' for s in shape_names])

    material_regex = (
        r'"|\.|count|piece|scent|scented|scentless|pack|ounces|oz|ml|g|kg|mm|'
        r'in|ft|ply|inches|cm|gallons|pounds|free|travel size|portable|all|^\d+$|'
        + color_regex + r'|' + shape_regex
    )

    df = df[~material_copy.str.contains(material_regex, regex=True, na=False)]

    def more_numbers_than_letters(s):
        s = str(s)
        return sum(c.isdigit() for c in s) > sum(c.isalpha() for c in s)

    df = df[~df['material'].apply(more_numbers_than_letters)]

    return df.reset_index(drop=True)

# ----------------------------------- END DATA CLEANING -----------------------------------

# ----------------------------------- RECOMMENDATION SYSTEM ----------------------------------- 

def build_similarity(df):
    df['text'] = df['name'] + ' ' + df['category'] + " " + df["material"]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(product_name, df, cosine_sim, n=5):
    idx_list = df.index[df['name'] == product_name].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]
    sim_scores = [(i, float(score)) for i, score in enumerate(cosine_sim[idx])]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top = sim_scores[1:n+1]
    rec_indices = [i for i, _ in top]
    return df.iloc[rec_indices]

# ----------------------------------- END RECOMMENDATION SYSTEM ----------------------------------- 

def set_price_range():
    st.write("### Set Price Range")
    min_price = st.number_input("Minimum Price", min_value=0.0, value=0.0, step=1.0)
    max_price = st.number_input("Maximum Price", min_value=0.0, value=100.0, step=1.0)
    return min_price, max_price
    
def find_top_rated_products(dataframe):
    return dataframe[(dataframe['rating'] >= 4.5) & (dataframe['reviewsCount'] > 1)].sort_values(by='rating', ascending=False)

def show_rating_distribution(df, title):
    fig = px.histogram(df, x='rating', nbins=20, title=title, color_discrete_sequence=['#FF9900'])
    st.plotly_chart(fig, use_container_width=True)

def show_price_vs_rating(df, title):
    fig = px.scatter(df, x='price', y='rating', hover_data=['name', 'brand'], title=title, color='rating', color_continuous_scale='Oranges')
    st.plotly_chart(fig, use_container_width=True)
    
def home():
    st.title("EcoRec")
    st.write("Welcome to EcoRec! It's a platform dedicated to promoting eco-friendly products and helping you make eco-friendly choices by recommending sustainable products.")

    df = load_and_clean_data('amazon_eco-friendly_products.csv')
    cosine_sim = build_similarity(df)

    st.write("### Search In-Stock Products")
    search_term = st.text_input("Enter a product name or category to search:")
    selected_option = st.selectbox("Filter:", ["None", "Find Top Rated Products", "Set Price Range"])

    filtered_df = df.copy()
    if search_term:
        filtered_df = df[df['name'].str.contains(search_term, case=False) | df['category'].str.contains(search_term, case=False)]
        if selected_option == "Find Top Rated Products":
            filtered_df = find_top_rated_products(filtered_df)
        elif selected_option == "Set Price Range":
            min_price, max_price = set_price_range()
            filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
        if not filtered_df.empty:
            st.write("### Search Results")
            filtered_df = filtered_df.reset_index(drop=True)
            product_labels = filtered_df.apply(lambda row: f"{row['name']} | {row['brand']} | ${row['price']:.2f}", axis=1)
            selected_idx = st.selectbox("Select a product:", options=filtered_df.index, format_func=lambda i: product_labels[i])
            st.write('###### Currently Selected Product')
            st.dataframe(filtered_df.loc[[selected_idx], ['name','category','material','brand','price','rating','reviewsCount']])
            selected_product_name = filtered_df.loc[selected_idx, 'name']
            st.write(f'###### All {search_term} Products')
            st.dataframe(filtered_df[['name','category','material','brand','price','rating','reviewsCount']])
            show_rating_distribution(filtered_df, f"Rating Distribution for '{search_term}' Products")
            show_price_vs_rating(filtered_df, f"Price vs. Rating for '{search_term}' Products")
            recs = get_recommendations(selected_product_name, df, cosine_sim, n=5)
            if not recs.empty:
                st.write("### Recommended for You")
                st.dataframe(recs.sort_values(by=['name','brand','price','rating','reviewsCount'], ascending=False)
                             [['name','category','material','brand','price','rating','reviewsCount']])
                show_rating_distribution(recs, "Rating Distribution for Recommended Products")
                show_price_vs_rating(recs, "Price vs. Rating for Recommended Products")
            else:
                st.write("No recommendations available for this selection.")
        else:
            st.write("No products found matching your criteria.")
    else:
        st.write("Enter a search term to get started.")
            
def contact():
    st.title("Contact Me")
    st.markdown("""
    **LinkedIn:** [https://linkedin.com/in/brenhuber](https://linkedin.com/in/brenhuber)
    
    **Email:** [brenhuberbusiness@gmail.com](mailto:brenhuberbusiness@gmail.com)
    
    **GitHub:** [https://github.com/Brenhuber](https://github.com/Brenhuber)
    """)

def about():
    st.title("About EcoRec")
    st.write("""
    EcoRec is a platform dedicated to promoting eco-friendly products and helping users make sustainable choices.
    The app allows you to search for a wide range of eco-friendly products, filter them by category, price, and rating, 
    and discover top-rated items. Using advanced text analysis and recommendation algorithms, EcoRec suggests similar 
    products based on your interests, making it easier to find sustainable alternatives. It is built using Streamlit, 
    a Python library for building interactive web applications, and it uses pandas for data manipulation and scikit-learn 
    for machine learning. The app cleans and processes product data to ensure high-quality recommendations, and the user-friendly 
    interface makes exploring green products simple and enjoyable. Join me on this eco-conscious journey in making a positive 
    impact on the environment by choosing products that are better for the planet!"""
    )

pages = {
    "Home": home,
    "About": about,
    "Contact": contact
}
selected_page = st.sidebar.radio("", list(pages.keys()), index=0)
pages[selected_page]()
