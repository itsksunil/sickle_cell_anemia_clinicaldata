import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

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

def load_data():
    try:
        # Verify file exists
        if not os.path.exists('sca.csv'):
            st.error("Error: sca.csv file not found in the current directory.")
            st.error("Please ensure:")
            st.error("1. Your CSV file is named 'sca.csv'")
            st.error("2. It's in the same folder as this app")
            return None
            
        # Read CSV with error handling
        df = pd.read_csv('sca.csv', encoding='utf-8', engine='python')
        
        if df.empty:
            st.error("The CSV file is empty.")
            return None
            
        # Clean and preprocess
        df.fillna('Unknown', inplace=True)
        
        # Drug extraction
        def extract_drugs(text):
            if pd.isna(text):
                return ['Other']
            found_drugs = []
            for drug in DRUG_LIST:
                if drug.lower() in str(text).lower():
                    found_drugs.append(drug)
            return found_drugs if found_drugs else ['Other']
        
        df['Drugs'] = df['Interventions'].apply(extract_drugs)
        df = df.explode('Drugs')
        
        # Date processing
        date_cols = [col for col in ['Start Date', 'Completion Date'] if col in df.columns]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f"{col.split()[0]}_Year"] = df[col].dt.year
        
        # Numeric columns
        if 'Enrollment' in df.columns:
            df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce').fillna(0).astype(int)
            
        return df
    
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

# Main app
def main():
    st.title("🩸 Sickle Cell Disease Clinical Trials Analysis")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Show raw data preview
    if st.checkbox("Show raw data preview"):
        st.write(df.head())
    
    # Filters
    st.sidebar.header("Filters")
    
    # Drug selection
    available_drugs = sorted(df['Drugs'].unique())
    selected_drugs = st.sidebar.multiselect(
        "Select drugs:",
        options=available_drugs,
        default=['Hydroxyurea', 'Sirolimus'] if 'Hydroxyurea' in available_drugs else available_drugs[:2]
    )
    
    # Filter data
    filtered_df = df[df['Drugs'].isin(selected_drugs)] if selected_drugs else df
    
    # Analysis
    st.header("Basic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Studies by Phase")
        if 'Phases' in filtered_df.columns:
            phase_counts = filtered_df['Phases'].value_counts()
            plt.figure(figsize=(8,4))
            phase_counts.plot(kind='bar')
            plt.xticks(rotation=45)
            st.pyplot(plt)
    
    with col2:
        st.subheader("Study Status")
        if 'Study Status' in filtered_df.columns:
            status_counts = filtered_df['Study Status'].value_counts()
            plt.figure(figsize=(8,4))
            plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
            st.pyplot(plt)
    
    # Data table
    st.header("Study Data")
    st.dataframe(filtered_df[['NCT Number', 'Study Title', 'Drugs', 'Phases', 'Study Status']])

if __name__ == "__main__":
    main()
