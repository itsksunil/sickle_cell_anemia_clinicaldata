import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import io
from collections import defaultdict

st.set_page_config(page_title="Sickle Cell Anemia Clinical Trials", layout="wide")

st.title("🩸 Sickle Cell Anemia Clinical Trials Dashboard")
st.markdown("Explore, filter, and analyze clinical trials related to **Sickle Cell Anemia**.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("sca.csv")
    df.fillna("", inplace=True)
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['Completion Date'] = pd.to_datetime(df['Completion Date'], errors='coerce')
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce').fillna(0).astype(int)
    df['Phases'] = df['Phases'].fillna("").astype(str)
    return df

df = load_data()

# Sidebar Filters as dropdown multi-selects
st.sidebar.header("🔎 Filter Options")

sexes = df['Sex'].dropna().unique().tolist()
selected_sex = st.sidebar.multiselect("Select Sex", options=sexes, default=sexes)

ages = df['Age'].dropna().unique().tolist()
selected_age = st.sidebar.multiselect("Select Age Range", options=ages, default=ages)

locations = df['Locations'].dropna().unique().tolist()
selected_locations = st.sidebar.multiselect("Select Location(s)", options=locations, default=locations)

phases = df['Phases'].dropna().unique().tolist()
selected_phases = st.sidebar.multiselect("Select Phase(s)", options=phases, default=phases)

medicines = df['Interventions'].dropna().unique().tolist()
selected_meds = st.sidebar.multiselect("Select Medicines", options=medicines)

# Enrollment numeric inputs instead of slider
enroll_min_val = int(df['Enrollment'].min())
enroll_max_val = int(df['Enrollment'].max())

enroll_min = st.sidebar.number_input("Min Enrollment", min_value=0, max_value=enroll_max_val, value=enroll_min_val, step=1)
enroll_max = st.sidebar.number_input("Max Enrollment", min_value=enroll_min, max_value=enroll_max_val, value=enroll_max_val, step=1)

# Date range picker
start_min = df['Start Date'].min()
start_max = df['Start Date'].max()
date_range = st.sidebar.date_input("Start Date Range", [start_min, start_max])

# Apply Filters
filtered_df = df[
    (df['Sex'].isin(selected_sex)) &
    (df['Age'].isin(selected_age)) &
    (df['Locations'].isin(selected_locations)) &
    (df['Phases'].isin(selected_phases)) &
    (df['Start Date'] >= pd.to_datetime(date_range[0])) &
    (df['Start Date'] <= pd.to_datetime(date_range[1])) &
    (df['Enrollment'] >= enroll_min) &
    (df['Enrollment'] <= enroll_max)
]

if selected_meds:
    filtered_df = filtered_df[filtered_df['Interventions'].apply(lambda x: any(m in x for m in selected_meds))]

# Bubble Chart for Medicine Frequency
exploded = df[['NCT Number', 'Interventions']].copy()
exploded['Interventions'] = exploded['Interventions'].str.split(';')
exploded = exploded.explode('Interventions')
exploded['Interventions'] = exploded['Interventions'].str.strip()
med_count = exploded.groupby('Interventions')['NCT Number'].count().reset_index()
med_count.columns = ['Medicine', 'Trial Count']

fig = px.scatter(med_count, x="Medicine", y="Trial Count", size="Trial Count", color="Trial Count", height=600,
                 title="Number of Trials Using Each Medicine")
fig.update_layout(xaxis_tickangle=-45)

# Detailed Pie Chart: Trial distribution by Phases
phase_counts = filtered_df['Phases'].value_counts().reset_index()
phase_counts.columns = ['Phase', 'Trial Count']
phase_pie = px.pie(phase_counts, names='Phase', values='Trial Count',
                   title='Distribution of Filtered Trials by Clinical Trial Phase',
                   hole=0.4,
                   labels={'Phase': 'Trial Phase', 'Trial Count': 'Number of Trials'},
                   hover_data=['Trial Count'])
phase_pie.update_traces(textposition='inside', textinfo='percent+label')

# Highlight terms function
highlight_terms = ['hemoglobin', 'pain', 'hospitalization', 'vaso-occlusive', 'crisis', 'transfusion', 'fatigue']

def highlight_common(text):
    for term in highlight_terms:
        pattern = re.compile(rf"\b({term})\b", flags=re.IGNORECASE)
        text = pattern.sub(r"🔹 **\1**", text)
    return text

# Similar Outcome Groups & Pie Chart
def group_similar_outcomes(df):
    texts = df['Primary Outcome Measures'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    groups = []
    visited = set()
    for i in range(len(texts)):
        if i in visited:
            continue
        group = [i]
        for j in range(i+1, len(texts)):
            if sim[i, j] > 0.7:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)
    return groups

outcome_texts = df[['NCT Number', 'Study Title', 'Primary Outcome Measures', 'Study URL']]
outcome_texts = ou_
