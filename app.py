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

# Known drug list
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
        df = pd.read_excel('sca_trials.xlsx')
        
        if df.empty:
            st.error("The dataset is empty. Please check your file.")
            return None
            
        df.fillna('Unknown', inplace=True)
        
        def extract_drugs(text):
            found_drugs = []
            if isinstance(text, str):
                for drug in DRUG_LIST:
                    if drug.lower() in text.lower():
                        found_drugs.append(drug)
            return found_drugs if found_drugs else ['Other']
        
        df['Drugs'] = df['Interventions'].apply(extract_drugs)
        df = df.explode('Drugs')
        
        date_cols = ['Start Date', 'Completion Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col.split()[0]}_Year"] = df[col].dt.year
        
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
**Basic version with core functionality**  
Drug comparison and study analysis.
""")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

selected_drugs = st.sidebar.multiselect(
    "Select Interventions/Drugs:",
    options=DRUG_LIST + ['Other'],
    default=['Hydroxyurea', 'Sirolimus']
)

status_options = sorted(df['Study Status'].unique())
selected_status = st.sidebar.multiselect(
    "Study Status:",
    options=status_options,
    default=['Completed', 'Recruiting']
)

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

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Drug Analysis", "📈 Outcomes", "🔍 Studies"])

with tab1:
    st.header("Drug Comparison")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Studies by Phase")
            phase_counts = filtered_df.groupby(['Drugs', 'Phases']).size().unstack()
            phase_counts.plot(kind='bar', stacked=True, figsize=(10,6))
            plt.xticks(rotation=45)
            plt.ylabel("Number of Studies")
            plt.tight_layout()
            st.pyplot(plt)
        
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
    else:
        st.warning("No data matching selected filters")

with tab2:
    st.header("Outcomes Analysis")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Outcome Measures")
            text = ' '.join(filtered_df['Primary Outcome Measures'].dropna().astype(str))
            if text.strip():
                wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10,6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("No outcome measures available")
        
        with col2:
            st.subheader("Status Distribution")
            status_counts = filtered_df['Study Status'].value_counts()
            plt.figure(figsize=(6,6))
            plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
            st.pyplot(plt)
    else:
        st.warning("No data available for analysis")

with tab3:
    st.header("Study Browser")
    
    search_term = st.text_input("Search studies:")
    if search_term:
        results = df[
            df['Study Title'].str.contains(search_term, case=False) |
            df['Interventions'].str.contains(search_term, case=False) |
            df['NCT Number'].str.contains(search_term, case=False)
        ]
    else:
        results = filtered_df
    
    if not results.empty:
        st.dataframe(
            results[[
                'NCT Number', 'Study Title', 'Study Status', 'Phases',
                'Drugs', 'Enrollment', 'Completion Date', 'Locations'
            ]].sort_values('Completion Date', ascending=False),
            height=600,
            use_container_width=True
        )
    else:
        st.warning("No matching studies found")

st.markdown("---")
st.markdown("""
**Basic Analysis Version** | Data Source: ClinicalTrials.gov  
*Using only core Python libraries for reliability*
""")
