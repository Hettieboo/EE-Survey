# ================================================================
# Streamlit Employee Survey Dashboard - Final Sweep
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # -----------------------------
    # Key columns
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
    disability_col = [c for c in df.columns if "disability" in c.lower()][0]

    # Convert to string where needed
    for col in [recognized_col, growth_col, impact_col, training_pref_col, fulfillment_col, disability_col]:
        df[col] = df[col].astype(str)

    # -----------------------------
    # Filters
    # -----------------------------
    st.sidebar.header("Filter Employees")
    role_filter = st.sidebar.multiselect("Role/Department", df[role_col].unique(), default=df[role_col].unique())
    age_filter = st.sidebar.multiselect("Age Group", df[age_col].unique(), default=df[age_col].unique())
    gender_filter = st.sidebar.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())
    df_filtered = df[df[role_col].isin(role_filter) & df[age_col].isin(age_filter) & df[gender_col].isin(gender_filter)]
    st.write(f"Filtered dataset: {df_filtered.shape[0]} respondents")

    # -----------------------------
    # KPIs in horizontal row
    # -----------------------------
    total = df_filtered.shape[0]

    recommend_scores = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
    positive_rec = recommend_scores[recommend_scores >= 8].count()
    avg_recommend = recommend_scores.mean()

    recognized = df_filtered[df_filtered[recognized_col].str.lower().str.contains("yes|somewhat", na=False)].shape[0]
    growth = df_filtered[df_filtered[growth_col].str.lower().str.contains("yes", na=False)].shape[0]
    impact = df_filtered[df_filtered[impact_col].str.lower().str.contains("positive impact", na=False)].shape[0]

    def kpi_tile(title, value, color):
        st.markdown(f"""
            <div style="
                display:inline-block;
                width: 150px;
                height: 120px;
                background-color: {color};
                border-radius: 10px;
                margin:5px;
                text-align:center;
                color:white;
                font-size:16px;
            ">
                <div style='padding-top:20px'><b>{title}</b></div>
                <div style='font-size:24px; padding-top:10px'>{value}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")  # spacing
    kpi_cols = st.columns(4)
    with kpi_cols[0]: kpi_tile("Recommend Homes First", f"{positive_rec}/{total} ({avg_recommend:.1f}/10 avg)", "#4b3fa0")
    with kpi_cols[1]: kpi_tile("Feel Recognized", f"{recognized}/{total} ({recognized/total*100:.1f}%)", "#007b5f")
    with kpi_cols[2]: kpi_tile("See Potential for Growth", f"{growth}/{total} ({growth/total*100:.1f}%)", "#b3396b")
    with kpi_cols[3]: kpi_tile("Feel Positive Impact", f"{impact}/{total} ({impact/total*100:.1f}%)", "#a03d2d")

    # -----------------------------
    # Helper for counts inside bars
    # -----------------------------
    def add_counts(ax, vertical=True):
        for p in ax.patches:
            if vertical:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', 
                                (p.get_x() + p.get_width() / 2., height/2),
                                ha='center', va='center', color='white', fontsize=12)
            else:
                width = p.get_width()
                if width > 0:
                    ax.annotate(f'{int(width)}',
                                (width/2, p.get_y() + p.get_height()/2),
                                ha='center', va='center', color='white', fontsize=12)

    # -----------------------------
    # Job Fulfillment Chart
    # -----------------------------
    st.header("Job Fulfillment")
    fig, ax = plt.subplots(figsize=(14,10))
    order = df_filtered[fulfillment_col].value_counts().index
    sns.countplot(x=df_filtered[fulfillment_col], order=order, palette=sns.color_palette("plasma", len(order)).as_hex(), ax=ax)
    ax.set_title("How fulfilling and rewarding do you find your work?", fontsize=16, pad=60)
    plt.xticks(rotation=0, ha='center')
    add_counts(ax)
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)

    # -----------------------------
    # Training Preferences Chart
    # -----------------------------
    st.header("Training Preferences")
    fig, ax = plt.subplots(figsize=(14,10))
    order = df_filtered[training_pref_col].value_counts().index
    sns.countplot(x=df_filtered[training_pref_col], order=order, palette=sns.color_palette("muted", len(order)).as_hex(), ax=ax)
    ax.set_title("How do you feel about live virtual training (Zoom/Teams) vs in-person?", fontsize=16, pad=60)
    plt.xticks(rotation=0, ha='center')
    add_counts(ax)
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)

    # -----------------------------
    # Disabilities Analysis - Age
    # -----------------------------
    st.header("Disabilities Analysis - Age")
    df_filtered[disability_col] = df_filtered[disability_col].fillna("No Disability")
    fig, ax = plt.subplots(figsize=(14,8))
    sns.countplot(
        x=df_filtered[age_col],
        hue=df_filtered[disability_col],
        palette=sns.color_palette("dark", len(df_filtered[disability_col].unique())).as_hex(),
        order=df_filtered[age_col].value_counts().index,
        ax=ax
    )
    ax.set_title("Do you identify as an individual living with a disability? - by Age Group", fontsize=16, pad=60)
    plt.xticks(rotation=0, ha='center')
    plt.ylabel("Number of Employees")
    plt.xlabel("Age Group")
    plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)

    # -----------------------------
    # Disabilities Analysis - Gender
    # -----------------------------
    st.header("Disabilities Analysis - Gender")
    fig, ax = plt.subplots(figsize=(14,8))
    sns.countplot(
        x=df_filtered[gender_col],
        hue=df_filtered[disability_col],
        palette=sns.color_palette("deep", len(df_filtered[disability_col].unique())).as_hex(),
        order=df_filtered[gender_col].value_counts().index,
        ax=ax
    )
    ax.set_title("Do you identify as an individual living with a disability? - by Gender", fontsize=16, pad=60)
    plt.xticks(rotation=0, ha='center')
    plt.ylabel("Number of Employees")
    plt.xlabel("Gender")
    plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)
