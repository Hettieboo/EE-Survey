# ================================================================
# Employee Survey Analysis - Streamlit App
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# Streamlit app
st.set_page_config(page_title="Employee Survey Analysis", layout="wide")

st.title("Employee Survey Analysis")
st.markdown("Analyze employee survey data interactively with visualizations and insights.")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your survey Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head(5))

    # Identify key columns
    role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
    recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
    years_col = [c for c in df.columns if "years" in c.lower() and "employed" in c.lower()][0]
    text_cols = [c for c in df.columns if "comment" in c.lower()]

    # -----------------------------
    # Demographics Analysis
    # -----------------------------
    st.subheader("Demographics Analysis")

    # Role/Department
    fig, ax = plt.subplots()
    sns.countplot(y=df[role_col], order=df[role_col].value_counts().index, palette='viridis', ax=ax)
    ax.set_title("Respondents by Role/Department")
    st.pyplot(fig)

    # Age
    fig, ax = plt.subplots()
    sns.countplot(x=df[age_col], order=df[age_col].value_counts().index, palette='magma', ax=ax)
    ax.set_title("Respondents by Age Group")
    st.pyplot(fig)

    # Gender
    fig, ax = plt.subplots()
    sns.countplot(x=df[gender_col], order=df[gender_col].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title("Respondents by Gender")
    st.pyplot(fig)

    # Likelihood to Recommend
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], order=df[recommend_col].value_counts().index, palette='plasma', ax=ax)
    ax.set_title("Likelihood to Recommend Homes First")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Years Employed
    fig, ax = plt.subplots()
    sns.countplot(x=df[years_col], order=df[years_col].value_counts().index, palette='cividis', ax=ax)
    ax.set_title("Years Employed at Homes First")
    st.pyplot(fig)

    # -----------------------------
    # Text Analysis
    # -----------------------------
    st.subheader("Text Analysis - Comments")
    STOP_WORDS = {'and','the','to','of','a','i','my','in','for','on','it','is','with','as','we','be'}

    # Combine all comments
    all_text = " ".join(df[col].dropna().astype(str).sum() for col in text_cols)

    # Top 20 words
    words = [w.lower().strip('.,!?;:()[]{}') for w in all_text.split() if w.lower() not in STOP_WORDS and len(w)>3]
    word_freq = Counter(words).most_common(20)
    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    st.write("Top 20 Words in Comments:")
    st.dataframe(word_df)

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    POSITIVE_WORDS = ['fulfilling', 'great', 'excellent', 'positive', 'amazing', 'helpful', 'supportive', 'good', 'love', 'enjoy', 'happy', 'appreciated', 'wonderful']
    NEGATIVE_WORDS = ['challenging', 'difficult', 'poor', 'lack', 'not', 'never', 'inadequate', 'frustrating', 'stress', 'overwhelmed', 'unhappy', 'concerned']

    positive_count = sum(sum(1 for w in POSITIVE_WORDS if w in str(text).lower()) for col in text_cols for text in df[col].dropna())
    negative_count = sum(sum(1 for w in NEGATIVE_WORDS if w in str(text).lower()) for col in text_cols for text in df[col].dropna())

    st.write(f"Positive mentions: {positive_count}")
    st.write(f"Negative mentions: {negative_count}")

    # -----------------------------
    # Theme Analysis
    # -----------------------------
    st.subheader("Theme Analysis")
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
        for text in df[col].dropna().astype(str):
            t = text.lower()
            for theme, keywords in THEMES.items():
                if any(kw in t for kw in keywords):
                    theme_counts[theme] += 1

    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions']).sort_values('Mentions', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Mentions', y='Theme', data=theme_df, palette='RdYlGn', ax=ax)
    ax.set_title("Key Themes in Survey Comments")
    st.pyplot(fig)
    st.dataframe(theme_df)

    # -----------------------------
    # Cross-analysis
    # -----------------------------
    st.subheader("Cross-analysis Examples")

    # Recommendation by Role
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], hue=df[role_col], data=df, palette='Set2', ax=ax)
    ax.set_title("Recommendation by Role/Department")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    # Recommendation by Age
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], hue=df[age_col], data=df, palette='Set3', ax=ax)
    ax.set_title("Recommendation by Age Group")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

else:
    st.info("Please upload an Excel file to start the analysis.")
