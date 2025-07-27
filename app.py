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

# Sidebar Filters
st.sidebar.header("🔎 Filter Options")

sexes = df['Sex'].dropna().unique().tolist()
ages = df['Age'].dropna().unique().tolist()
locations = df['Locations'].dropna().unique().tolist()
phases = df['Phases'].dropna().unique().tolist()
medicines = df['Interventions'].dropna().unique().tolist()

selected_sex = st.sidebar.multiselect("Select Sex", options=sexes, default=sexes)
selected_age = st.sidebar.multiselect("Select Age Range", options=ages, default=ages)
selected_locations = st.sidebar.multiselect("Select Location(s)", options=locations, default=locations)
selected_phases = st.sidebar.multiselect("Select Phase(s)", options=phases, default=phases)
selected_meds = st.sidebar.multiselect("Select Medicines", options=medicines)

enroll_min_val = int(df['Enrollment'].min())
enroll_max_val = int(df['Enrollment'].max())

enroll_min = st.sidebar.number_input("Min Enrollment", min_value=0, max_value=enroll_max_val, value=enroll_min_val, step=1)
enroll_max = st.sidebar.number_input("Max Enrollment", min_value=enroll_min, max_value=enroll_max_val, value=enroll_max_val, step=1)

start_min = df['Start Date'].min()
start_max = df['Start Date'].max()
date_range = st.sidebar.date_input("Start Date Range", [start_min, start_max])

# Filtering
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

# Medicine bubble chart
exploded = df[['NCT Number', 'Interventions']].copy()
exploded['Interventions'] = exploded['Interventions'].str.split(';')
exploded = exploded.explode('Interventions')
exploded['Interventions'] = exploded['Interventions'].str.strip()
med_count = exploded.groupby('Interventions')['NCT Number'].count().reset_index()
med_count.columns = ['Medicine', 'Trial Count']

fig = px.scatter(med_count, x="Medicine", y="Trial Count", size="Trial Count", color="Trial Count", height=600,
                 title="Number of Trials Using Each Medicine")
fig.update_layout(xaxis_tickangle=-45)

# Pie chart for trial phases
phase_counts = filtered_df['Phases'].value_counts().reset_index()
phase_counts.columns = ['Phase', 'Trial Count']
phase_pie = px.pie(phase_counts, names='Phase', values='Trial Count',
                   title='Distribution of Filtered Trials by Clinical Trial Phase',
                   hole=0.4)

# Outcome grouping
highlight_terms = ['hemoglobin', 'pain', 'hospitalization', 'vaso-occlusive', 'crisis', 'transfusion', 'fatigue']
def highlight_common(text):
    for term in highlight_terms:
        pattern = re.compile(rf"\b({term})\b", flags=re.IGNORECASE)
        text = pattern.sub(r"🔹 **\1**", text)
    return text

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
outcome_texts = outcome_texts[outcome_texts['Primary Outcome Measures'] != ""]
groups = group_similar_outcomes(outcome_texts)

pie_data = defaultdict(int)
for i, group in enumerate(groups):
    pie_data[f"Group {i+1}"] = len(group)

pie_df = pd.DataFrame(pie_data.items(), columns=['Group', 'Trial Count'])

outcome_pie_chart = px.pie(pie_df, names='Group', values='Trial Count',
                           title="Distribution of Similar Outcome Groups", hole=0.4) if not pie_df.empty else None

# Fixed PDF generation
def generate_pdf_from_df(df):
    def safe_text(text):
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sickle Cell Anemia Trial Summary", ln=True, align='C')

    for _, row in df.iterrows():
        pdf.set_font("Arial", 'B', size=11)
        pdf.multi_cell(0, 10, safe_text(f"{row['Study Title']} (NCT: {row['NCT Number']})"))
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, safe_text(f"URL: {row.get('Study URL', '')}"))
        pdf.multi_cell(0, 8, safe_text(f"Location: {row.get('Locations', 'N/A')}"))
        pdf.multi_cell(0, 8, safe_text(f"Enrollment: {row.get('Enrollment', 'N/A')}"))
        pdf.multi_cell(0, 8, safe_text(f"Interventions: {row.get('Interventions', 'N/A')}"))

        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(0, 8, "Primary Outcome:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, safe_text(row.get("Primary Outcome Measures", "")))

        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(0, 8, "Secondary Outcome:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, safe_text(row.get("Secondary Outcome Measures", "")))

        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(0, 8, "Other Outcome:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 8, safe_text(row.get("Other Outcome Measures", "")))

        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Trial Data & Visuals",
    "Outcome Similarity Pie Chart",
    "PDF Report Download",
    "Phases Distribution Pie Chart"
])

with tab1:
    st.markdown(f"### Total Trials in Dataset: **{df.shape[0]}**")
    st.dataframe(df[['NCT Number', 'Study Title']], use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 Filtered Trials: {filtered_df.shape[0]}")
    st.dataframe(filtered_df[['NCT Number', 'Study Title', 'Sex', 'Age', 'Enrollment', 'Locations', 'Phases', 'Start Date', 'Interventions']], use_container_width=True)

    st.markdown("---")
    st.subheader("💊 Medicine Frequency in Trials")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 View Trials by Medicine")
    selected_med = st.selectbox("Select a Medicine", med_count['Medicine'].unique())
    related_trials = df[df['Interventions'].str.contains(selected_med, na=False)]

    for _, row in related_trials.iterrows():
        st.markdown(f"**{row['Study Title']}**")
        st.markdown(f"[🔗 View on ClinicalTrials.gov]({row['Study URL']})", unsafe_allow_html=True)
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

with tab2:
    st.subheader("📊 Outcome Similarity Pie Chart")
    if outcome_pie_chart:
        st.plotly_chart(outcome_pie_chart, use_container_width=True)
    else:
        st.info("Not enough similar outcomes to display chart.")

with tab3:
    st.subheader("📄 Download Trial Summary as PDF")
    if st.button("Generate PDF Report"):
        pdf_file = generate_pdf_from_df(filtered_df)
        st.download_button("⬇️ Download PDF", data=pdf_file, file_name="SCA_Trial_Summary.pdf", mime="application/pdf")

    st.subheader("📥 Download Filtered Data as CSV")
    st.download_button("⬇️ Download CSV", data=filtered_df.to_csv(index=False), file_name="filtered_sca_trials.csv", mime="text/csv")

with tab4:
    st.subheader("📊 Clinical Trial Distribution by Phase")
    if not phase_counts.empty:
        st.plotly_chart(phase_pie, use_container_width=True)
        st.markdown("""
        **Phases Explained:**
        - **Phase 1:** Safety testing with small groups.
        - **Phase 2:** Effectiveness and side effects.
        - **Phase 3:** Confirm effectiveness, monitor side effects.
        - **Phase 4:** Post-marketing studies.
        """)
    else:
        st.info("No data available for selected filters.")
