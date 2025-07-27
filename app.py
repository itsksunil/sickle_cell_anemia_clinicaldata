import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Sickle Cell Trials Analysis",
    page_icon="🩸",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sca.csv')
        
        # Basic data validation
        if df.empty:
            st.error("The dataset is empty. Please check your CSV file.")
            return None
            
        # Clean and preprocess data
        df.fillna('', inplace=True)
        
        # Date processing with error handling
        date_cols = ['Start Date', 'Completion Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col.split()[0]}_Year"] = df[col].dt.year
        
        # Numeric columns processing
        if 'Enrollment' in df.columns:
            df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
            df['Enrollment'] = df['Enrollment'].fillna(0).astype(int)
        
        # Calculate study duration if both dates exist
        if all(col in df.columns for col in date_cols):
            df['Study Duration'] = (df['Completion Date'] - df['Start Date']).dt.days
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data with error handling
df = load_data()

if df is None:
    st.stop()

# Title and description
st.title("🩸 Sickle Cell Anemia Clinical Trials Analysis")
st.markdown("""
This interactive dashboard analyzes clinical trials data for sickle cell disease (SCD).
Explore the data using the filters below.
""")

# Sidebar filters
st.sidebar.header("🔍 Filter Trials")

# Safe multiselect for status
available_statuses = sorted(df['Study Status'].dropna().unique())
default_status = ['RECRUITING'] if 'RECRUITING' in available_statuses else available_statuses[:1]
selected_status = st.sidebar.multiselect(
    "Study Status",
    options=available_statuses,
    default=default_status
)

# Safe multiselect for phases
available_phases = sorted([p for p in df['Phases'].dropna().unique() if p])
selected_phase = st.sidebar.multiselect(
    "Phase",
    options=available_phases,
    default=available_phases[:3] if len(available_phases) > 3 else available_phases
)

# Safe country selection
if 'Country' in df.columns:
    available_countries = sorted(df['Country'].dropna().unique())
    selected_country = st.sidebar.multiselect(
        "Country",
        options=available_countries,
        default=available_countries[:3] if len(available_countries) > 3 else available_countries
    )
else:
    selected_country = []

# Safe enrollment range
max_enroll = int(df['Enrollment'].max()) if 'Enrollment' in df.columns else 1000
min_enroll, max_enroll = st.sidebar.slider(
    "Enrollment Range",
    min_value=0,
    max_value=max_enroll,
    value=(0, max_enroll)
)

# Apply filters
filtered_df = df.copy()
if selected_status:
    filtered_df = filtered_df[filtered_df['Study Status'].isin(selected_status)]
if selected_phase:
    filtered_df = filtered_df[filtered_df['Phases'].isin(selected_phase)]
if selected_country and 'Country' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_country)]
if 'Enrollment' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['Enrollment'] >= min_enroll) & 
        (filtered_df['Enrollment'] <= max_enroll)
    ]

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trials", len(filtered_df))
col2.metric("Avg Enrollment", 
            int(filtered_df['Enrollment'].mean()) if 'Enrollment' in filtered_df.columns else "N/A",
            help="Average number of participants")
col3.metric("Completed", 
            len(filtered_df[filtered_df['Study Status'] == 'COMPLETED']) if 'Study Status' in filtered_df.columns else "N/A")
col4.metric("Active Trials", 
            len(filtered_df[filtered_df['Study Status'].str.contains('ACTIVE|RECRUIT', case=False)]) 
            if 'Study Status' in filtered_df.columns else "N/A")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📈 Overview", "💊 Interventions", "🌍 Locations"])

with tab1:
    st.header("Trial Overview")
    
    # Row 1: Phase and Status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty and 'Phases' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            phase_counts = filtered_df['Phases'].value_counts()
            phase_counts.plot(kind='bar', color=plt.cm.viridis(np.linspace(0, 1, len(phase_counts))), ax=ax)
            plt.title('Trials by Phase')
            plt.xlabel('Phase')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with col2:
        if not filtered_df.empty and 'Study Status' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            status_counts = filtered_df['Study Status'].value_counts()
            status_counts.plot(kind='pie', autopct='%1.1f%%', 
                             colors=plt.cm.Pastel1.colors, ax=ax)
            plt.title('Trial Status')
            plt.ylabel('')
            st.pyplot(fig)
    
    # Row 2: Trials over time
    if not filtered_df.empty and 'Start_Year' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_data = filtered_df.groupby(['Start_Year', 'Study Status']).size().unstack().fillna(0)
        yearly_data.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
        plt.title('Trials Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Trials')
        plt.legend(title='Status', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

with tab2:
    st.header("Treatment Interventions")
    
    if not filtered_df.empty and 'Interventions' in filtered_df.columns:
        # Extract interventions
        interventions = []
        for val in filtered_df['Interventions']:
            if val:
                interventions.extend([x.strip() for x in val.split('|')])
        
        if interventions:
            # Top interventions
            st.subheader("Most Common Interventions")
            top_interventions = pd.Series(interventions).value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_interventions.plot(kind='barh', color='darkblue', ax=ax)
            plt.title('Top 15 Interventions')
            plt.xlabel('Number of Trials')
            st.pyplot(fig)
            
            # Intervention types breakdown
            st.subheader("Intervention Types")
            intervention_types = []
            for val in interventions:
                if 'DRUG' in val: intervention_types.append('Drug')
                elif 'BIOLOGICAL' in val: intervention_types.append('Biological')
                elif 'DEVICE' in val: intervention_types.append('Device')
                elif 'PROCEDURE' in val: intervention_types.append('Procedure')
                else: intervention_types.append('Other')
            
            fig, ax = plt.subplots(figsize=(8, 6))
            pd.Series(intervention_types).value_counts().plot(
                kind='pie', autopct='%1.1f%%', 
                colors=plt.cm.Set3.colors, ax=ax)
            plt.ylabel('')
            st.pyplot(fig)

with tab3:
    st.header("Geographic Distribution")
    
    if not filtered_df.empty and 'Country' in filtered_df.columns:
        # Country distribution
        country_counts = filtered_df['Country'].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        country_counts.plot(kind='barh', color='darkgreen', ax=ax)
        plt.title('Top 15 Countries')
        plt.xlabel('Number of Trials')
        st.pyplot(fig)
        
        # Enrollment by country
        st.subheader("Enrollment by Country")
        if 'Enrollment' in filtered_df.columns:
            enroll_by_country = filtered_df.groupby('Country')['Enrollment'].sum().nlargest(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            enroll_by_country.plot(kind='bar', color='teal', ax=ax)
            plt.title('Total Participants by Country')
            plt.ylabel('Number of Participants')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Data table at bottom
st.header("🔍 Detailed Trial Data")
st.dataframe(
    filtered_df[[
        'NCT Number', 'Study Title', 'Study Status', 'Phases',
        'Enrollment', 'Start Date', 'Completion Date', 'Country',
        'Interventions', 'Primary Outcome Measures'
    ]].sort_values('Completion Date', ascending=False),
    height=500,
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** ClinicalTrials.gov  
**Note:** This dashboard is for research purposes only.
""")
