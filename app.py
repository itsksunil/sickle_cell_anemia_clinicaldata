import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        # Try reading CSV file
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
    
    except FileNotFoundError:
        st.error("Error: File 'sca.csv' not found. Please ensure:")
        st.error("1. The file is named 'sca.csv'")
        st.error("2. It's in the same directory as this app")
        return None
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

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Drug selection
selected_drugs = st.sidebar.multiselect(
    "Select Interventions/Drugs:",
    options=DRUG_LIST + ['Other'],
    default=['Hydroxyurea', 'Sirolimus']
)

# Status filter
status_options = sorted(df['Study Status'].unique())
selected_status = st.sidebar.multiselect(
    "Study Status:",
    options=status_options,
    default=['Completed', 'Recruiting']
)

# Phase filter
phase_options = sorted([p for p in df['Phases'].unique() if p != 'Unknown'])
selected_phase = st.sidebar.multiselect(
    "Phase:",
    options=phase_options,
    default=phase_options
)

# Apply filters
filtered_df = df[
    (df['Drugs'].isin(selected_drugs)) &
    (df['Study Status'].isin(selected_status)) &
    (df['Phases'].isin(selected_phase))
]

# Main analysis tabs
tab1, tab2, tab3 = st.tabs(["📊 Drug Analysis", "📈 Outcomes", "🔍 Study Browser"])

with tab1:
    st.header("Therapeutic Intervention Comparison")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        # Trials by drug and phase
        with col1:
            st.subheader("Studies by Drug and Phase")
            phase_counts = filtered_df.groupby(['Drugs', 'Phases']).size().unstack()
            phase_counts.plot(kind='bar', stacked=True, figsize=(10,6))
            plt.xticks(rotation=45)
            plt.ylabel("Number of Studies")
            plt.tight_layout()
            st.pyplot(plt)
        
        # Enrollment distribution
        with col2:
            st.subheader("Enrollment Distribution")
            plt.figure(figsize=(10,6))
            for drug in selected_drugs:
                subset = filtered_df[filtered_df['Drugs'] == drug]
                plt.hist(subset['Enrollment'], alpha=0.5, label=drug, bins=20)
            plt.xlabel("Number of Participants")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(plt)
            
        # Completion rates
        st.subheader("Completion Status")
        status_counts = filtered_df['Study Status'].value_counts()
        plt.figure(figsize=(8,6))
        plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
        st.pyplot(plt)
        
    else:
        st.warning("No data matching selected filters")

with tab2:
    st.header("Outcomes Analysis")
    
    if not filtered_df.empty:
        # Outcome measures word cloud
        st.subheader("Frequent Outcome Measures")
        text = ' '.join(filtered_df['Primary Outcome Measures'].dropna().astype(str))
        
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.warning("No outcome measures data available")
        
        # Conditions analysis
        st.subheader("Associated Conditions")
        if 'Conditions' in filtered_df.columns:
            conditions = filtered_df['Conditions'].value_counts().head(10)
            plt.figure(figsize=(10,5))
            conditions.plot(kind='barh')
            plt.xlabel("Number of Studies")
            st.pyplot(plt)
    else:
        st.warning("No data available for outcomes analysis")

with tab3:
    st.header("Study Browser")
    
    # Search functionality
    search_term = st.text_input("Search studies by title, drug, or NCT number:")
    
    if search_term:
        search_results = df[
            df['Study Title'].str.contains(search_term, case=False) |
            df['Interventions'].str.contains(search_term, case=False) |
            df['NCT Number'].str.contains(search_term, case=False)
        ]
    else:
        search_results = filtered_df
    
    if not search_results.empty:
        # Display important columns
        st.dataframe(
            search_results[[
                'NCT Number', 'Study Title', 'Study Status', 'Phases',
                'Drugs', 'Enrollment', 'Start Date', 'Completion Date',
                'Conditions', 'Locations'
            ]].sort_values('Completion Date', ascending=False),
            height=600,
            use_container_width=True
        )
        
        # Export option
        st.download_button(
            label="📥 Export Selected Studies",
            data=search_results.to_csv(index=False),
            file_name="scd_studies_export.csv",
            mime="text/csv"
        )
    else:
        st.warning("No studies match your search criteria")

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** ClinicalTrials.gov | **Analysis:** SCD Research Dashboard  
*For research purposes only - consult medical professionals for treatment decisions*
""")
