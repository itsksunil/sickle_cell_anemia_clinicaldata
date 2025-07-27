# sca_clinical_app.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sickle Cell Anemia Clinical Trials", layout="wide")

st.title("🩸 Sickle Cell Anemia Clinical Trials Dashboard")
st.markdown("Explore clinical trials data for **Sickle Cell Anemia**, filter by demographics, and analyze medicine-condition relationships.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("sca.csv")
    df.fillna("", inplace=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

# Sex Filter
sexes = df['Sex'].dropna().unique().tolist()
selected_sex = st.sidebar.multiselect("Select Sex", options=sexes, default=sexes)

# Age Filter
age_values = df['Age'].dropna().unique().tolist()
selected_age = st.sidebar.multiselect("Select Age Range", options=age_values, default=age_values)

# Medicine Filter (from Interventions column)
medicines = df['Interventions'].dropna().unique().tolist()
selected_meds = st.sidebar.multiselect("Select Medicine/Intervention", options=medicines)

# Filtered Data
filtered_df = df[
    (df['Sex'].isin(selected_sex)) &
    (df['Age'].isin(selected_age))
]

if selected_meds:
    filtered_df = filtered_df[filtered_df['Interventions'].apply(lambda x: any(med in x for med in selected_meds))]

st.subheader(f"📋 Filtered Clinical Trials: {filtered_df.shape[0]} records")
st.dataframe(filtered_df, use_container_width=True)

# --- Correlation Analysis ---
st.markdown("---")
st.subheader("🔗 Correlation: Interventions vs Conditions")

# Create medicine-condition pair dataframe
pairs = []

for _, row in filtered_df.iterrows():
    meds = row['Interventions'].split(';')
    conds = row['Conditions'].split(';')
    for med in meds:
        for cond in conds:
            if med.strip() and cond.strip():
                pairs.append((med.strip(), cond.strip()))

pair_df = pd.DataFrame(pairs, columns=["Medicine", "Condition"])

# Group and count
pair_count = pair_df.groupby(['Medicine', 'Condition']).size().reset_index(name='Count')

# Visualize using heatmap-style bubble chart
if not pair_count.empty:
    fig = px.scatter(pair_count, x="Medicine", y="Condition", size="Count", color="Count",
                     title="Intervention vs Condition Frequency",
                     labels={"Count": "Frequency"}, height=600)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for selected filters to show correlation.")

# Option to download filtered data
st.markdown("---")
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_sca_trials.csv',
    mime='text/csv'
)
