# ================================================================
# Interactive Employee Survey Dashboard - Streamlit
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

st.set_page_config(page_title="Homes First Survey Dashboard", layout="wide")
st.title("Homes First Employee Survey Dashboard")

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head(3))

    # -----------------------------
    # Identify key columns dynamically
    # -----------------------------
    role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
    recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
    years_col = [c for c in df.columns if "years" in c.lower() and "employed" in c.lower()][0]
    recognized_col = [c for c in df.columns if "recognized" in c.lower() and "acknowledged" in c.lower()][0]
    growth_col = [c for c in df.columns if "potential for growth" in c.lower()][0]
    impact_col = [c for c in df.columns if "positive impact" in c.lower()][0]
    training_pref_col = [c for c in df.columns if "live virtual training" in c.lower()][0]
    fulfillment_col = [c for c in df.columns if "fulfilling and rewarding" in c.lower()][0]

    # -----------------------------
    # Interactive filters
    # -----------------------------
    st.sidebar.header("Filter Employees")
    role_filter = st.sidebar.multiselect("Role/Department", df[role_col].unique(), default=df[role_col].unique())
    age_filter = st.sidebar.multiselect("Age Group", df[age_col].unique(), default=df[age_col].unique())
    gender_filter = st.sidebar.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())
    df_filtered = df[df[role_col].isin(role_filter) & df[age_col].isin(age_filter) & df[gender_col].isin(gender_filter)]

    st.write(f"Filtered dataset: {df_filtered.shape[0]} respondents")

    # ==============================
    # KPIs
    # ==============================
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    positive_rec = df_filtered[df_filtered[recommend_col].str.lower().str.contains("likely|yes", na=False)].shape[0]
    total = df_filtered.shape[0]
    col1.metric("Recommend Homes First", f"{positive_rec}/{total} ({positive_rec/total*100:.1f}%)")

    recognized = df_filtered[df_filtered[recognized_col].str.lower().str.contains("yes|somewhat", na=False)].shape[0]
    col2.metric("Feel Recognized", f"{recognized}/{total} ({recognized/total*100:.1f}%)")

    growth = df_filtered[df_filtered[growth_col].str.lower().str.contains("yes", na=False)].shape[0]
    col3.metric("See Potential for Growth", f"{growth}/{total} ({growth/total*100:.1f}%)")

    impact = df_filtered[df_filtered[impact_col].str.lower().str.contains("positive impact", na=False)].shape[0]
    col4.metric("Feel Positive Impact", f"{impact}/{total} ({impact/total*100:.1f}%)")

    # ==============================
    # Demographics Analysis
    # ==============================
    st.header("Demographics")
    col1, col2, col3 = st.columns(3)

    # Role
    fig, ax = plt.subplots()
    sns.countplot(y=df_filtered[role_col], order=df_filtered[role_col].value_counts().index, palette='viridis', ax=ax)
    ax.set_title("Respondents by Role/Department")
    col1.pyplot(fig)

    # Age
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[age_col], order=df_filtered[age_col].value_counts().index, palette='magma', ax=ax)
    ax.set_title("Respondents by Age Group")
    col2.pyplot(fig)

    # Gender
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[gender_col], order=df_filtered[gender_col].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title("Respondents by Gender")
    col3.pyplot(fig)

    # ==============================
    # Job Fulfillment
    # ==============================
    st.header("Job Fulfillment")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[fulfillment_col], order=df_filtered[fulfillment_col].value_counts().index, palette='plasma', ax=ax)
    ax.set_title("Job Fulfillment")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==============================
    # Training Preferences
    # ==============================
    st.header("Training Preferences")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[training_pref_col], order=df_filtered[training_pref_col].value_counts().index, palette='Set2', ax=ax)
    ax.set_title("Training Mode Preference")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ==============================
    # Sentiment & Themes (Comments)
    # ==============================
    st.header("Comments Analysis")
    text_cols = [c for c in df_filtered.columns if "comment" in c.lower()]
    STOP_WORDS = {'and','the','to','of','a','i','my','in','for','on','it','is','with','as','we','be'}

    # Combine all comments
    all_text = " ".join(df_filtered[col].dropna().astype(str).sum() for col in text_cols)

    # Top 20 words
    words = [w.lower().strip('.,!?;:()[]{}') for w in all_text.split() if w.lower() not in STOP_WORDS and len(w)>3]
    word_freq = Counter(words).most_common(20)
    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    st.write("Top Words in Comments:")
    st.dataframe(word_df)

    # Themes
    THEMES = {
        "Workload": ["understaffed", "workload", "busy", "overwhelming", "paperwork", "time", "tasks"],
        "Support": ["support", "help", "assist", "guidance", "resources", "tools"],
        "Team": ["team", "colleagues", "coworkers", "collaboration", "together", "staff"],
        "Training": ["training", "development", "learning", "skills", "education", "growth"],
        "Clients": ["client", "resident", "people", "helping", "care", "community"],
        "Management": ["management", "leadership", "supervisor", "manager", "direction"],
        "Recognition": ["recognition", "appreciation", "valued", "acknowledged", "feedback"],
        "Work-Life Balance": ["balance", "flexibility", "schedule", "hours", "time off"]
    }

    theme_counts = {theme: 0 for theme in THEMES}
    for col in text_cols:
        for text in df_filtered[col].dropna().astype(str):
            t = text.lower()
            for theme, keywords in THEMES.items():
                for kw in keywords:
                    if kw in t:
                        theme_counts[theme] += 1
                        break

    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions']).sort_values('Mentions', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Mentions', y='Theme', data=theme_df, palette='RdYlGn', ax=ax)
    ax.set_title("Key Themes in Comments")
    st.pyplot(fig)
    st.dataframe(theme_df)

    # ==============================
    # Cross Analysis Examples
    # ==============================
    st.header("Cross Analysis")
    st.subheader("Recommendation by Role")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[recommend_col], hue=df_filtered[role_col], data=df_filtered, palette='Set2', ax=ax)
    ax.set_title("Recommendation by Role/Department")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("Recommendation by Age Group")
    fig, ax = plt.subplots()
    sns.countplot(x=df_filtered[recommend_col], hue=df_filtered[age_col], data=df_filtered, palette='Set3', ax=ax)
    ax.set_title("Recommendation by Age Group")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
