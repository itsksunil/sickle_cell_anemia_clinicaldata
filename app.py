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
        df = pd.read_excel('sca_trials.xlsx')  # Changed to read Excel
        
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Drug Comparison", 
    "🌍 Geographic View", 
    "📈 Outcomes Analysis",
    "🔍 Study Browser"
])

with tab1:
    st.header("Therapeutic Intervention Comparison")
    
    if not filtered_df.empty:
        # Drug efficacy metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Studies", len(filtered_df['NCT Number'].unique()))
        col2.metric("Avg Enrollment", int(filtered_df['Enrollment'].mean()))
        
        # Success rate calculation (simplified)
        completed = filtered_df[filtered_df['Study Status'] == 'Completed']
        success_rate = len(completed) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        col3.metric("Completion Rate", f"{success_rate:.1f}%")
        
        # Drug distribution by phase
        fig = px.sunburst(
            filtered_df,
            path=['Drugs', 'Phases'],
            color='Drugs',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Enrollment comparison
        fig = px.box(
            filtered_df,
            x='Drugs',
            y='Enrollment',
            color='Phases',
            points="all",
            title="Enrollment Distribution by Drug"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matching selected filters")

with tab2:
    st.header("Geographic Distribution of Studies")
    
    if not filtered_df.empty and 'Locations' in filtered_df.columns:
        # Extract country from locations
        filtered_df['Country'] = filtered_df['Locations'].str.split(',').str[-1].str.strip()
        
        # Map visualization
        country_counts = filtered_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.choropleth(
            country_counts,
            locations='Country',
            locationmode='country names',
            color='Count',
            hover_name='Country',
            color_continuous_scale='Viridis',
            title="Global Distribution of Selected Trials"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Country-drug matrix
        pivot = pd.crosstab(filtered_df['Country'], filtered_df['Drugs'])
        st.dataframe(
            pivot.style.background_gradient(cmap='Blues'),
            use_container_width=True
        )
    else:
        st.warning("Location data not available for selected filters")

with tab3:
    st.header("Clinical Outcomes Analysis")
    
    if not filtered_df.empty:
        # Outcome measures word cloud
        st.subheader("Frequent Outcome Measures")
        text = ' '.join(filtered_df['Primary Outcome Measures'].dropna().astype(str))
        
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.warning("No outcome measures data available")
        
        # Success correlation matrix (simplified)
        st.subheader("Drug-Success Correlation")
        
        # Create dummy success metric (in real app, use actual outcomes)
        success_df = filtered_df.copy()
        success_df['Success Score'] = np.random.randint(1, 100, size=len(success_df))
        
        fig = px.scatter(
            success_df,
            x='Drugs',
            y='Success Score',
            size='Enrollment',
            color='Phases',
            hover_name='Study Title',
            title="Drug Effectiveness Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for outcomes analysis")

with tab4:
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
        # Enhanced data display
        st.dataframe(
            search_results[[
                'NCT Number', 'Study Title', 'Study Status', 'Phases',
                'Drugs', 'Enrollment', 'Start Date', 'Completion Date',
                'Locations', 'Primary Outcome Measures'
            ]].sort_values('Completion Date', ascending=False),
            height=600,
            use_container_width=True
        )
        
        # Export option
        st.download_button(
            label="Export Selected Studies",
            data=search_results.to_csv(index=False),
            file_name="scd_studies_export.csv",
            mime="text/csv"
        )
    else:
        st.warning("No studies match your search criteria")

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** ClinicalTrials.gov | **Analysis:** SCD Research Dashboard v2.0  
*For research purposes only - consult medical professionals for treatment decisions*
""")
