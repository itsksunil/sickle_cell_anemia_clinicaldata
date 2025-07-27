import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set matplotlib style
plt.style.use('ggplot')

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv('sickle_cell_anemia_clinicaldata.csv')
    df.fillna('', inplace=True)
    df['Completion Date'] = pd.to_datetime(df['Completion Date'], errors='coerce')
    df['Completion Year'] = df['Completion Date'].dt.year
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    return df

df = load_data()

st.title("sickle cell anemiaclinicaldataTrials Explorer 1700+ clinical trial 🧬")

# Utility: extract unique keywords from pipe-separated fields
def get_unique_keywords(series):
    keywords = set()
    for val in series:
        if val:
            keywords.update([x.strip() for x in val.split('|')])
    return sorted(keywords)

# Prepare filter values
conditions = get_unique_keywords(df['Conditions'])
interventions = get_unique_keywords(df['Interventions'])
phases = sorted([p for p in df['Phases'].unique() if p])

# Sidebar filters
st.sidebar.header("🔍 Filter Trials")

selected_condition = st.sidebar.selectbox("Cancer Type", ["All"] + conditions)
selected_intervention = st.sidebar.selectbox("Intervention/Drug", ["All"] + interventions)
selected_phase = st.sidebar.selectbox("Phase", ["All"] + phases)

# Date Range filter
min_date = df['Completion Date'].min()
max_date = df['Completion Date'].max()
start_date, end_date = st.sidebar.date_input(
    "Completion Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# Apply filters
filtered_df = df.copy()

if selected_condition != "All":
    filtered_df = filtered_df[filtered_df['Conditions'].str.contains(selected_condition, case=False, na=False)]
if selected_intervention != "All":
    filtered_df = filtered_df[filtered_df['Interventions'].str.contains(selected_intervention, case=False, na=False)]
if selected_phase != "All":
    filtered_df = filtered_df[filtered_df['Phases'] == selected_phase]

# Filter by date range
filtered_df = filtered_df[
    (filtered_df['Completion Date'] >= pd.to_datetime(start_date)) &
    (filtered_df['Completion Date'] <= pd.to_datetime(end_date))
]

st.markdown(f"### 🎯 {len(filtered_df)} trials found")

# --------------------------
# TRIALS OVERVIEW VISUALIZATION
# --------------------------
st.markdown("## 📊 Trial Overview Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Trials by Phase (Matplotlib bar chart)
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        phase_counts = filtered_df['Phases'].value_counts()
        phase_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=ax)
        plt.title('Trials by Phase')
        plt.xlabel('Phase')
        plt.ylabel('Number of Trials')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data to display for Trials by Phase")

with col2:
    # Trials Over Time (Matplotlib line chart)
    if not filtered_df.empty and 'Completion Year' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        yearly_counts = filtered_df['Completion Year'].value_counts().sort_index()
        yearly_counts.plot(kind='line', marker='o', color='green', ax=ax)
        plt.title('Trials Completed Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Trials')
        st.pyplot(fig)
    else:
        st.warning("No data to display for Trials Over Time")

# Enrollment Distribution (Matplotlib histogram)
if not filtered_df.empty and 'Enrollment' in filtered_df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(filtered_df['Enrollment'].dropna(), bins=20, color='purple', edgecolor='black')
    plt.title('Distribution of Enrollment Numbers')
    plt.xlabel('Number of Participants')
    plt.ylabel('Frequency')
    st.pyplot(fig)

# --------------------------
# MAIN TRIAL DATA TABLE
# --------------------------
st.markdown("## 📋 Trial Data Table")
st.dataframe(filtered_df[[
    'NCT Number', 'Study Title', 'Conditions', 'Interventions', 
    'Phases', 'Enrollment', 'Completion Date'
]])

st.markdown("---")

# --------------------------
# EXPLORE RELATED CANCER TYPES
# --------------------------
st.subheader("🔎 Explore Related Cancer Types")
explore_conditions = get_unique_keywords(filtered_df['Conditions'])
explore_cond = st.selectbox("Explore Cancer Type Trials:", options=["Select"] + explore_conditions)

if explore_cond != "Select":
    subset = df[df['Conditions'].str.contains(explore_cond, case=False, na=False)]
    st.markdown(f"### Trials related to **{explore_cond}** ({len(subset)})")
    
    # Visualization for the selected cancer type
    if not subset.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            subset['Phases'].value_counts().plot(
                kind='pie', autopct='%1.1f%%', 
                colors=['#ff9999','#66b3ff','#99ff99'],
                ax=ax
            )
            plt.title(f'Phase Distribution for {explore_cond}')
            plt.ylabel('')
            st.pyplot(fig)
        
        with col2:
            if 'Completion Year' in subset.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                subset['Completion Year'].value_counts().sort_index().plot(
                    kind='bar', color='skyblue', ax=ax
                )
                plt.title(f'Trials Over Time for {explore_cond}')
                plt.xlabel('Year')
                plt.ylabel('Number of Trials')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    st.dataframe(subset[[
        'NCT Number', 'Study Title', 'Primary Outcome Measures',
        'Secondary Outcome Measures', 'Other Outcome Measures'
    ]])

st.markdown("---")

# --------------------------
# COMPARE TRIAL OUTCOMES
# --------------------------
st.subheader("📈 Compare Clinical Trial Outcomes by Cancer Type")

selected_compare = st.selectbox("Select Cancer Type to Compare Trials:", ["Select"] + conditions)

if selected_compare != "Select":
    compare_df = df[df['Conditions'].str.contains(selected_compare, case=False, na=False)]
    st.markdown(f"### Comparing outcomes for **{selected_compare}** ({len(compare_df)} trials)")
    
    # Outcome comparison visualizations
    if not compare_df.empty:
        st.markdown("#### Outcome Measures Analysis")
        
        # Count outcome measures with simplified function
        def count_measures(series):
            if series.empty:
                return 0
            first_val = str(series.iloc[0])
            if '|' in first_val:
                return series.str.split('|').str.len()
            else:
                return series.apply(lambda x: 1 if x else 0)

        outcome_measures = pd.DataFrame({
            'Primary': count_measures(compare_df['Primary Outcome Measures']),
            'Secondary': count_measures(compare_df['Secondary Outcome Measures']),
            'Other': count_measures(compare_df['Other Outcome Measures'])
        })
        
        # Plot outcome measures (Matplotlib bar chart)
        fig, ax = plt.subplots(figsize=(10, 5))
        outcome_measures.mean().plot(
            kind='bar', 
            color=['#4C72B0', '#55A868', '#C44E52'], 
            ax=ax
        )
        plt.title(f'Average Number of Outcome Measures for {selected_compare}')
        plt.ylabel('Average Number of Measures')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        # Enrollment vs Phase (Matplotlib boxplot)
        if 'Enrollment' in compare_df.columns and 'Phases' in compare_df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Group data by phase
            groups = []
            labels = []
            for phase in sorted(compare_df['Phases'].unique()):
                groups.append(compare_df[compare_df['Phases'] == phase]['Enrollment'].dropna())
                labels.append(phase)
            
            ax.boxplot(groups, labels=labels)
            plt.title(f'Enrollment Distribution by Phase for {selected_compare}')
            plt.xlabel('Phase')
            plt.ylabel('Enrollment')
            st.pyplot(fig)
    
    st.dataframe(compare_df[[
        'NCT Number', 'Study Title', 'Phases', 'Enrollment', 'Completion Date',
        'Primary Outcome Measures', 'Secondary Outcome Measures', 'Other Outcome Measures'
    ]].reset_index(drop=True))
