import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('ggplot')
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('sca.csv')
    # Clean and preprocess data
    df.fillna('', inplace=True)
    df['Completion Date'] = pd.to_datetime(df['Completion Date'], errors='coerce')
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['Completion Year'] = df['Completion Date'].dt.year
    df['Start Year'] = df['Start Date'].dt.year
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df['Study Duration'] = (df['Completion Date'] - df['Start Date']).dt.days
    return df

df = load_data()

# Title and description
st.title("Sickle Cell Anemia Clinical Trials Analysis")
st.markdown("""
This dashboard provides insights into clinical trials for sickle cell anemia (SCA) from around the world.
Explore the data using the filters below.
""")

# Sidebar filters
st.sidebar.header("Filter Trials")
selected_status = st.sidebar.multiselect(
    "Study Status",
    options=sorted(df['Study Status'].unique()),
    default=['RECRUITING', 'COMPLETED', 'ACTIVE']
)

selected_phase = st.sidebar.multiselect(
    "Phase",
    options=sorted([p for p in df['Phases'].unique() if p]),
    default=sorted([p for p in df['Phases'].unique() if p])
)

selected_country = st.sidebar.multiselect(
    "Country",
    options=sorted(df['Country'].unique()),
    default=sorted(df['Country'].unique())
)

min_enroll, max_enroll = st.sidebar.slider(
    "Enrollment Range",
    min_value=0,
    max_value=int(df['Enrollment'].max()),
    value=(0, int(df['Enrollment'].max()))
)

# Apply filters
filtered_df = df[
    (df['Study Status'].isin(selected_status)) &
    (df['Phases'].isin(selected_phase)) &
    (df['Country'].isin(selected_country)) &
    (df['Enrollment'] >= min_enroll) &
    (df['Enrollment'] <= max_enroll)
]

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trials", len(filtered_df))
col2.metric("Average Enrollment", int(filtered_df['Enrollment'].mean()))
col3.metric("Completed Trials", len(filtered_df[filtered_df['Study Status'] == 'COMPLETED']))
col4.metric("Recruiting Trials", len(filtered_df[filtered_df['Study Status'] == 'RECRUITING']))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Interventions", "Geographic", "Trial Details"])

with tab1:
    st.header("Trial Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        # Trials by Phase
        fig, ax = plt.subplots(figsize=(10, 6))
        phase_counts = filtered_df['Phases'].value_counts()
        phase_counts.plot(kind='bar', color=sns.color_palette("viridis", len(phase_counts)), ax=ax)
        plt.title('Trials by Phase')
        plt.xlabel('Phase')
        plt.ylabel('Number of Trials')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Trials by Status
        fig, ax = plt.subplots(figsize=(10, 6))
        status_counts = filtered_df['Study Status'].value_counts()
        status_counts.plot(kind='pie', autopct='%1.1f%%', 
                         colors=sns.color_palette("pastel", len(status_counts)), ax=ax)
        plt.title('Trial Status Distribution')
        plt.ylabel('')
        st.pyplot(fig)
    
    # Trials Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_counts = filtered_df.groupby(['Start Year', 'Study Status']).size().unstack()
    yearly_counts.plot(kind='bar', stacked=True, ax=ax, 
                      colormap='viridis', width=0.8)
    plt.title('Trials Started by Year and Status')
    plt.xlabel('Year')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
    plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

with tab2:
    st.header("Treatment Interventions")
    
    # Get unique interventions
    interventions = []
    for val in filtered_df['Interventions']:
        if val:
            interventions.extend([x.strip() for x in val.split('|')])
    intervention_counts = pd.Series(interventions).value_counts().head(15)
    
    col1, col2 = st.columns(2)
    with col1:
        # Top Interventions
        fig, ax = plt.subplots(figsize=(10, 8))
        intervention_counts.plot(kind='barh', color='darkblue', ax=ax)
        plt.title('Top 15 Treatment Interventions')
        plt.xlabel('Number of Trials')
        plt.ylabel('Intervention')
        st.pyplot(fig)
    
    with col2:
        # Intervention Types
        intervention_types = []
        for val in interventions:
            if 'DRUG' in val:
                intervention_types.append('Drug')
            elif 'BIOLOGICAL' in val:
                intervention_types.append('Biological')
            elif 'DEVICE' in val:
                intervention_types.append('Device')
            elif 'PROCEDURE' in val:
                intervention_types.append('Procedure')
            else:
                intervention_types.append('Other')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pd.Series(intervention_types).value_counts().plot(
            kind='pie', autopct='%1.1f%%', 
            colors=sns.color_palette("Set3"), ax=ax)
        plt.title('Intervention Types')
        plt.ylabel('')
        st.pyplot(fig)
    
    # Intervention by Phase
    intervention_phase = filtered_df.explode('Interventions').groupby(
        ['Interventions', 'Phases']).size().unstack().fillna(0)
    intervention_phase['Total'] = intervention_phase.sum(axis=1)
    top_interventions = intervention_phase.nlargest(10, 'Total').drop('Total', axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_interventions.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.title('Top Interventions by Phase')
    plt.xlabel('Intervention')
    plt.ylabel('Number of Trials')
    plt.xticks(rotation=45)
    plt.legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

with tab3:
    st.header("Geographic Distribution")
    
    # Country distribution
    country_counts = filtered_df['Country'].value_counts().head(15)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        country_counts.plot(kind='barh', color='darkgreen', ax=ax)
        plt.title('Top 15 Countries with SCA Trials')
        plt.xlabel('Number of Trials')
        plt.ylabel('Country')
        st.pyplot(fig)
    
    with col2:
        # Map visualization (placeholder - would use actual mapping library in production)
        st.write("""
        ### Geographic Distribution
        *Map visualization would appear here showing trial locations*
        
        This would display an interactive world map with markers for each trial location.
        """)
    
    # Enrollment by Country
    enrollment_by_country = filtered_df.groupby('Country')['Enrollment'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    enrollment_by_country.plot(kind='bar', color='teal', ax=ax)
    plt.title('Total Enrollment by Country (Top 10)')
    plt.xlabel('Country')
    plt.ylabel('Total Participants')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab4:
    st.header("Detailed Trial Information")
    
    # Search functionality
    search_term = st.text_input("Search trials by title or NCT number:")
    if search_term:
        display_df = filtered_df[
            filtered_df['Study Title'].str.contains(search_term, case=False) | 
            filtered_df['NCT Number'].str.contains(search_term, case=False)
        ]
    else:
        display_df = filtered_df
    
    # Detailed table view
    st.dataframe(
        display_df[[
            'NCT Number', 'Study Title', 'Study Status', 'Phases', 
            'Enrollment', 'Start Date', 'Completion Date', 'Country',
            'Interventions', 'Primary Outcome Measures'
        ]].sort_values('Completion Date', ascending=False),
        height=600,
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** ClinicalTrials.gov  
**Note:** This is a demo application for sickle cell anemia clinical trials analysis.
""")
