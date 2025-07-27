# sca_clinical_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Sickle Cell Anemia Clinical Trials", layout="wide")

st.title("🩸 Sickle Cell Anemia Clinical Trials Dashboard")
st.markdown("Explore clinical trials for **Sickle Cell Anemia**, filter by demographics and medicines, and analyze outcome relationships.")

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

# Medicine Filter
medicines = df['Interventions'].dropna().unique().tolist()
selected_meds = st.sidebar.multiselect("Select Medicine/Intervention", options=medicines)

# Apply filters
filtered_df = df[
    (df['Sex'].isin(selected_sex)) &
    (df['Age'].isin(selected_age))
]

if selected_meds:
    filtered_df = filtered_df[filtered_df['Interventions'].apply(lambda x: any(med in x for med in selected_meds))]

st.subheader(f"📋 Filtered Clinical Trials: {filtered_df.shape[0]} records")
st.dataframe(filtered_df[['NCT Number', 'Study Title', 'Sex', 'Age', 'Interventions']], use_container_width=True)

# --- Medicine-Trial Frequency Bubble Chart ---
st.markdown("---")
st.subheader("💊 Frequency of Medicines in Trials")

exploded_df = df[['NCT Number', 'Study Title', 'Interventions', 'Primary Outcome Measures', 'Study URL']].copy()
exploded_df['Interventions'] = exploded_df['Interventions'].str.split(';')
exploded_df = exploded_df.explode('Interventions')
exploded_df['Interventions'] = exploded_df['Interventions'].str.strip()

med_count = exploded_df.groupby('Interventions')['NCT Number'].count().reset_index()
med_count.columns = ['Medicine', 'Trial Count']

fig = px.scatter(med_count, x="Medicine", y="Trial Count", size="Trial Count", color="Trial Count",
                 title="Medicine Appearance in Clinical Trials",
                 hover_name="Medicine", height=600)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# --- Trial Outcomes for Selected Medicine ---
st.markdown("---")
st.subheader("🔬 View Trial Outcomes by Medicine")

selected_med = st.selectbox("Select Medicine to View Trials", med_count['Medicine'].unique())

related_trials = exploded_df[exploded_df['Interventions'] == selected_med]

st.markdown(f"### Trials using **{selected_med}** ({related_trials.shape[0]} trials)")

for _, row in related_trials.iterrows():
    st.markdown(f"**{row['Study Title']}**")
    st.markdown(f"[🔗 View on ClinicalTrials.gov]({row['Study URL']})", unsafe_allow_html=True)
    with st.expander("Primary Outcome"):
        st.markdown(row['Primary Outcome Measures'])
    st.markdown("---")

# --- Find Similar Outcomes ---
st.markdown("---")
st.subheader("🔗 Link Trials with Similar Outcomes")

outcome_texts = df[['NCT Number', 'Study Title', 'Primary Outcome Measures', 'Study URL']].copy()
outcome_texts = outcome_texts[outcome_texts['Primary Outcome Measures'] != ""]

if not outcome_texts.empty:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(outcome_texts['Primary Outcome Measures'])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    shown = set()
    for i in range(len(outcome_texts)):
        for j in range(i + 1, len(outcome_texts)):
            if similarity_matrix[i, j] > 0.7:
                if (i, j) not in shown:
                    shown.add((i, j))
                    st.markdown(f"✅ **Similar Outcome Trials:**")
                    st.markdown(f"- [{outcome_texts.iloc[i]['Study Title']}]({outcome_texts.iloc[i]['Study URL']})")
                    st.markdown(f"  ↪ {outcome_texts.iloc[i]['Primary Outcome Measures']}")
                    st.markdown(f"- [{outcome_texts.iloc[j]['Study Title']}]({outcome_texts.iloc[j]['Study URL']})")
                    st.markdown(f"  ↪ {outcome_texts.iloc[j]['Primary Outcome Measures']}")
                    st.markdown("---")

# --- Download Filtered Data ---
st.markdown("### 📥 Download Filtered Data")
st.download_button(
    label="⬇️ Download as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_sca_trials.csv",
    mime="text/csv"
)
