<h1 align="center">🌱 EcoRec</h1>
<p align="center"><em>Discover eco-friendly products that match your values and preferences</em></p>

---

### 🚀 Overview

**EcoRec** is a Streamlit-based app that helps you discover and explore eco-friendly products. It uses both collaborative filtering and content-based filtering to recommend products similar to your interests.

---

### ✨ Features

- Search and filter eco-friendly products by name, category, or price  
- Personalized recommendations based on your selection  
- Always shows relevant recommendations (by similarity or top-rated)  
- Visual insights with interactive Plotly charts  
- Data cleaning for accurate results  
- Modern, interactive UI  

---

### 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![pandas](https://img.shields.io/badge/pandas-Data%20Handling-purple?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Graphs-orange?logo=plotly)

---

### ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Brenhuber/EcoRecommendationSystem.git
   cd EcoRec
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or with conda:
   # conda install -c conda-forge streamlit pandas scikit-learn plotly
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   
---

### 🧭 Usage

- Use the sidebar to navigate Home, About, and Contact pages.
- On Home, search for products, filter, and get recommendations.
- Recommendations are always shown: first by collaborative filtering, then by content similarity, then by top-rated.

---

### 📋 Requirements

- Python 3.8–3.11
- streamlit
- pandas
- scikit-learn
- plotly
