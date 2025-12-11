import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8,5)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Employee Survey Analysis", layout="wide")
st.title("Employee Survey Analysis Dashboard")
st.markdown("Interactive survey insights with highlights, charts, and key metrics.")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your survey Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Identify key columns
    role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
    recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
    years_col = [c for c in df.columns if "years" in c.lower() and "employed" in c.lower()][0]
    text_cols = [c for c in df.columns if "comment" in c.lower()]

    # -----------------------------
    # Sentiment Analysis
    # -----------------------------
    POSITIVE_WORDS = ['fulfilling', 'great', 'excellent', 'positive', 'amazing', 'helpful', 'supportive',
                      'good', 'love', 'enjoy', 'happy', 'appreciated', 'wonderful']
    NEGATIVE_WORDS = ['challenging', 'difficult', 'poor', 'lack', 'not', 'never', 'inadequate',
                      'frustrating', 'stress', 'overwhelmed', 'unhappy', 'concerned']

    positive_count = sum(sum(1 for w in POSITIVE_WORDS if w in str(text).lower())
                         for col in text_cols for text in df[col].dropna())
    negative_count = sum(sum(1 for w in NEGATIVE_WORDS if w in str(text).lower())
                         for col in text_cols for text in df[col].dropna())

    # -----------------------------
    # Theme Analysis
    # -----------------------------
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

    # -----------------------------
    # Highlights Section
    # -----------------------------
    st.subheader("Highlights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", df.shape[0])
    col2.metric("Positive Mentions", positive_count)
    col3.metric("Areas of Concern", negative_count)

    col4, col5 = st.columns(2)
    # Role Distribution
    fig, ax = plt.subplots()
    sns.countplot(y=df[role_col], order=df[role_col].value_counts().index, palette='viridis', ax=ax)
    ax.set_title("Respondents by Role")
    col4.pyplot(fig)

    # Age Distribution
    fig, ax = plt.subplots()
    sns.countplot(x=df[age_col], order=df[age_col].value_counts().index, palette='magma', ax=ax)
    ax.set_title("Age Distribution")
    col5.pyplot(fig)

    # -----------------------------
    # Key Themes Horizontal Bar Chart
    # -----------------------------
    st.subheader("Key Themes in Survey Responses (Top 10)")
    top_themes = theme_df.head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='Mentions', y='Theme', data=top_themes, palette='RdYlGn', ax=ax)
    ax.set_title("Top Themes by Mentions")
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Demographics Detailed Charts
    # -----------------------------
    st.subheader("Demographics Charts")

    # Gender
    fig, ax = plt.subplots()
    sns.countplot(x=df[gender_col], order=df[gender_col].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title("Respondents by Gender")
    st.pyplot(fig)

    # Years Employed
    fig, ax = plt.subplots()
    sns.countplot(x=df[years_col], order=df[years_col].value_counts().index, palette='cividis', ax=ax)
    ax.set_title("Years Employed")
    st.pyplot(fig)

    # Likelihood to Recommend
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], order=df[recommend_col].value_counts().index, palette='plasma', ax=ax)
    ax.set_title("Likelihood to Recommend Homes First")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -----------------------------
    # Top Words Horizontal Bar Chart
    # -----------------------------
    st.subheader("Top 10 Frequent Words in Comments")
    STOP_WORDS = {'and','the','to','of','a','i','my','in','for','on','it','is','with','as','we','be'}
    all_text = " ".join(df[col].dropna().astype(str).sum() for col in text_cols)
    words = [w.lower().strip('.,!?;:()[]{}') for w in all_text.split()
             if w.lower() not in STOP_WORDS and len(w)>3]
    word_freq = Counter(words)
    word_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=False)
    top_words = word_df.head(10)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='Frequency', y='Word', data=top_words, palette='coolwarm', ax=ax)
    ax.set_title("Top 10 Frequent Words in Comments")
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Cross-analysis Charts
    # -----------------------------
    st.subheader("Cross-analysis: Recommendation by Role")
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], hue=df[role_col], data=df, palette='Set2', ax=ax)
    ax.set_title("Recommendation by Role/Department")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("Cross-analysis: Recommendation by Age Group")
    fig, ax = plt.subplots()
    sns.countplot(x=df[recommend_col], hue=df[age_col], data=df, palette='Set3', ax=ax)
    ax.set_title("Recommendation by Age Group")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

else:
    st.info("Please upload an Excel file to start the analysis.")
