import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

st.set_page_config(page_title="Clinical Trial Comparator", layout="wide")
st.title("🔬 Clinical Trial Comparator Platform")
st.markdown("Compare clinical trials, analyze outcomes, and filter insights for research and decision-making.")

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

# Sidebar filters
st.sidebar.header("📊 Filter Trials")
sexes = df['Sex'].dropna().unique().tolist()
ages = df['Age'].dropna().unique().tolist()
locations = df['Locations'].dropna().unique().tolist()
phases = df['Phases'].dropna().unique().tolist()
statuses = df['Study Status'].dropna().unique().tolist()
medicines = df['Interventions'].dropna().unique().tolist()

selected_sex = st.sidebar.multiselect("Sex", options=sexes, default=sexes)
selected_age = st.sidebar.multiselect("Age Range", options=ages, default=ages)
selected_locations = st.sidebar.multiselect("Location(s)", options=locations, default=locations)
selected_phases = st.sidebar.multiselect("Study Phases", options=phases, default=phases)
selected_status = st.sidebar.multiselect("Study Status", options=statuses, default=statuses)
selected_meds = st.sidebar.multiselect("Medicines", options=medicines)

enroll_min_val = int(df['Enrollment'].min())
enroll_max_val = int(df['Enrollment'].max())
enroll_min = st.sidebar.number_input("Min Enrollment", min_value=0, max_value=enroll_max_val, value=enroll_min_val)
enroll_max = st.sidebar.number_input("Max Enrollment", min_value=enroll_min, max_value=enroll_max_val, value=enroll_max_val)

start_min = df['Start Date'].min()
start_max = df['Start Date'].max()
date_range = st.sidebar.date_input("Start Date Range", [start_min, start_max])

# Filter data
filtered_df = df[
    (df['Sex'].isin(selected_sex)) &
    (df['Age'].isin(selected_age)) &
    (df['Locations'].isin(selected_locations)) &
    (df['Phases'].isin(selected_phases)) &
    (df['Study Status'].isin(selected_status)) &
    (df['Start Date'] >= pd.to_datetime(date_range[0])) &
    (df['Start Date'] <= pd.to_datetime(date_range[1])) &
    (df['Enrollment'] >= enroll_min) &
    (df['Enrollment'] <= enroll_max)
]

if selected_meds:
    filtered_df = filtered_df[filtered_df['Interventions'].apply(lambda x: any(m in x for m in selected_meds))]

# Chart: Medicine frequency
exploded = df[['NCT Number', 'Interventions']].copy()
exploded['Interventions'] = exploded['Interventions'].str.split(';')
exploded = exploded.explode('Interventions')
exploded['Interventions'] = exploded['Interventions'].str.strip()
med_count = exploded.groupby('Interventions')['NCT Number'].count().reset_index()
med_count.columns = ['Medicine', 'Trial Count']

med_chart = px.scatter(med_count, x="Medicine", y="Trial Count", size="Trial Count", color="Trial Count",
                       title="🔬 Number of Trials per Medicine", height=500)
med_chart.update_layout(xaxis_tickangle=-45)

# Chart: Trial phases
phase_counts = filtered_df['Phases'].value_counts().reset_index()
phase_counts.columns = ['Phase', 'Trial Count']
phase_chart = px.pie(phase_counts, names='Phase', values='Trial Count',
                     title='📈 Trial Distribution by Phase', hole=0.4)

# Outcome grouping
highlight_terms = ['hemoglobin', 'pain', 'hospitalization', 'vaso-occlusive', 'crisis', 'transfusion', 'fatigue']
def highlight_common(text):
    for term in highlight_terms:
        pattern = re.compile(rf"\b({term})\b", flags=re.IGNORECASE)
        text = pattern.sub(r"🔹 **\1**", text)
    return text

def group_outcomes(df):
    texts = df['Primary Outcome Measures'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    groups = []
    visited = set()
    for i in range(len(texts)):
        if i in visited: continue
        group = [i]
        for j in range(i+1, len(texts)):
            if sim[i, j] > 0.7:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            groups.append(group)
    return groups

outcomes_df = df[['NCT Number', 'Study Title', 'Primary Outcome Measures', 'Study URL']]
outcomes_df = outcomes_df[outcomes_df['Primary Outcome Measures'] != ""]
groups = group_outcomes(outcomes_df)

pie_data = defaultdict(int)
for i, group in enumerate(groups):
    pie_data[f"Group {i+1}"] = len(group)
group_df = pd.DataFrame(pie_data.items(), columns=['Group', 'Trial Count'])

outcome_pie = px.pie(group_df, names='Group', values='Trial Count',
                     title="🔍 Outcome Similarity Groups", hole=0.4) if not group_df.empty else None

# Tabs
tab1, tab2 = st.tabs(["📋 Compare Trials", "📊 Visual Analytics"])

with tab1:
    st.markdown(f"### Total Trials in Dataset: **{df.shape[0]}**")
    st.dataframe(df[['NCT Number', 'Study Title']], use_container_width=True)

    st.markdown("---")
    st.subheader(f"🔎 Filtered Trials: {filtered_df.shape[0]}")
    st.dataframe(
        filtered_df[['NCT Number', 'Study Title', 'Sex', 'Age', 'Enrollment', 'Locations', 'Phases',
                     'Study Status', 'Start Date', 'Interventions']],
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🧪 Medicine-wise Trial Details")
    st.plotly_chart(med_chart, use_container_width=True)

    selected_med = st.selectbox("Select a Medicine to Compare Trials", med_count['Medicine'].unique())
    med_trials = df[df['Interventions'].str.contains(selected_med, na=False)]

    for _, row in med_trials.iterrows():
        st.markdown(f"#### 🔗 [{row['Study Title']}]({row['Study URL']})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Primary Outcome**")
            st.markdown(highlight_common(row.get('Primary Outcome Measures', '')), unsafe_allow_html=True)
        with col2:
            st.markdown("**Secondary Outcome**")
            st.markdown(highlight_common(row.get('Secondary Outcome Measures', '')), unsafe_allow_html=True)
        with col3:
            st.markdown("**Other Outcome**")
            st.markdown(highlight_common(row.get('Other Outcome Measures', '')), unsafe_allow_html=True)
        st.markdown("---")

    st.subheader("⬇️ Download Filtered Data")
    st.download_button("Download CSV", data=filtered_df.to_csv(index=False), file_name="filtered_trials.csv", mime="text/csv")

with tab2:
    st.subheader("📊 Trial Phase Distribution")
    if not phase_counts.empty:
        st.plotly_chart(phase_chart, use_container_width=True)
    else:
        st.info("No phase data available for selected filters.")

    st.markdown("---")
    st.subheader("📊 Outcome Similarity")
    if outcome_pie:
        st.plotly_chart(outcome_pie, use_container_width=True)
    else:
        st.info("Not enough outcome similarity data to show chart.")
