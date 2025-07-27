import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from datetime import datetime
from wordcloud import WordCloud

# Configure page
st.set_page_config(
    page_title="SCD Trials Analyzer",
    page_icon="🩸",
    layout="wide"
)

# Known drug/intervention list
DRUG_LIST = [
    'Hydroxyurea', 'GSK4172239D', 'Sirolimus', 'Arginine', 'Glutamine',
    'Folic Acid', 'SANGUINATE', 'Ketamine', 'Alemtuzumab', 'Fludarabine',
    'Melphalan', 'Thiotepa', 'Nitric Oxide', 'Cyclophosphamide', 'Tacrolimus',
    'Busulfan', 'Methylprednisolone', 'Deferasirox', 'Abatacept', 'Protease Inhibitors',
    'GBT021601', 'Anti-Thymocyte Globulin', 'Cyclosporine', 'Methotrexate',
    'Mycophenolate Mofetil', 'Treosulfan', 'Cannabis', 'ACTIQ'
]

@st.cache_data
def load_data():
    try:
        # Changed to read CSV instead of Excel
        df = pd.read_csv('sca.csv')
        
        # Basic data validation
        if df.empty:
            st.error("The dataset is empty. Please check your file.")
            return None
            
        # Clean and preprocess
        df.fillna('Unknown', inplace=True)
        
        # Standardize drug names in Interventions
        def extract_drugs(text):
            found_drugs = []
            if isinstance(text, str):
                for drug in DRUG_LIST:
                    if drug.lower() in text.lower():
                        found_drugs.append(drug)
            return found_drugs if found_drugs else ['Other']
        
        df['Drugs'] = df['Interventions'].apply(extract_drugs)
        df = df.explode('Drugs')
        
        # Date processing
        date_cols = ['Start Date', 'Completion Date', 'Primary Completion Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col.split()[0]}_Year"] = df[col].dt.year
        
        # Numeric columns
        if 'Enrollment' in df.columns:
            df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce').fillna(0).astype(int)
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()
if df is None:
    st.stop()

# Title
st.title("🩸 Sickle Cell Disease Clinical Trials Analysis")
st.markdown("""
**Comparative analysis of therapeutic interventions**  
Explore effectiveness across different studies.
""")

# Rest of your code remains the same...
# [Keep all the existing code for filters, tabs, and visualizations]
