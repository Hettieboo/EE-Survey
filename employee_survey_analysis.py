# ================================================================
# Homes First Employee Survey Dashboard - Streamlit Version
# ================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

st.set_page_config(page_title="Homes First Survey Analysis", layout="wide")
st.title("Homes First Employee Survey Dashboard")

# -----------------------------
# Upload Excel File
# -----------------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset shape:", df.shape)
else:
    st.stop()

# -----------------------------
# Key columns
# -----------------------------
role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
age_col = [c for c in df.columns if "age" in c.lower()][0]
gender_col = [c for c in df.columns if "gender" in c.lower()][0]
recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
fulfillment_col = [c for c in df.columns if "fulfilling" in c.lower()][0]
training_pref_col = [c for c in df.columns if "live virtual training" in c.lower()][0]
disability_col = [c for c in df.columns if "disability" in c.lower() and "identify" in c.lower()][0]

# -----------------------------
# Filters
# -----------------------------
with st.sidebar:
    st.header("Filter Employees")
    role_filter = st.multiselect("Role/Department", df[role_col].unique(), default=df[role_col].unique())
    age_filter = st.multiselect("Age Group", df[age_col].unique(), default=df[age_col].unique())
    gender_filter = st.multiselect("Gender", df[gender_col].unique(), default=df[gender_col].unique())

df_filtered = df[df[role_col].isin(role_filter) & df[age_col].isin(age_filter) & df[gender_col].isin(gender_filter)]
st.write(f"Filtered dataset: {df_filtered.shape[0]} respondents")

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

# Example other KPIs (replace with actual calculations)
col1.metric("Total Respondents", f"{df_filtered.shape[0]}", delta=None)
col2.metric("Avg Years Employed", f"{df_filtered['How many years have you been employed by Homes First?'].mean():.1f}", delta=None)
col3.metric("Avg Job Fulfillment", f"{df_filtered[fulfillment_col].astype('category').cat.codes.mean():.1f}", delta=None)

# Recommend Homes First KPI: show average and median out of 10
avg_recommend = df_filtered[recommend_col].mean()
median_recommend = df_filtered[recommend_col].median()
col4.metric("Avg Recommend Homes First", f"{avg_recommend:.1f}/10", delta=None)
# Optionally display median as tooltip or additional line
st.caption(f"Median Recommendation Score: {median_recommend:.0f}/10")

# -----------------------------
# Helper function for labels inside bars + counts
# -----------------------------
def add_labels_inside_bar(ax, labels):
    for p, label in zip(ax.patches, labels):
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        # Label inside bar
        ax.text(x, y/2, label, ha='center', va='center', color='white', fontsize=12, wrap=True)
        # Count above bar
        ax.text(x, y + 0.1, int(y), ha='center', va='bottom', color='black', fontsize=12)

# -----------------------------
# Job Fulfillment Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,10))
order = df_filtered[fulfillment_col].value_counts().index
sns.countplot(
    x=df_filtered[fulfillment_col],
    order=order,
    palette=sns.color_palette("plasma", len(order)).as_hex(),
    ax=ax
)
ax.set_title("How fulfilling and rewarding do you find your work?", fontsize=16, pad=60)
ax.set_xticklabels([])  # hide overlapping x-axis labels
add_labels_inside_bar(ax, order)
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Training Preferences Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,10))
order = df_filtered[training_pref_col].value_counts().index
sns.countplot(
    x=df_filtered[training_pref_col],
    order=order,
    palette=sns.color_palette("muted", len(order)).as_hex(),
    ax=ax
)
ax.set_title("How do you feel about live virtual training (Zoom/Teams) vs in-person?", fontsize=16, pad=60)
ax.set_xticklabels([])
add_labels_inside_bar(ax, order)
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Disabilities by Age Chart
# -----------------------------
df_filtered[disability_col] = df_filtered[disability_col].fillna("No Disability")
fig, ax = plt.subplots(figsize=(14,8))
order = df_filtered[age_col].value_counts().index
sns.countplot(
    x=df_filtered[age_col],
    hue=df_filtered[disability_col],
    palette=sns.color_palette("dark", len(df_filtered[disability_col].unique())).as_hex(),
    order=order,
    ax=ax
)
ax.set_title("Do you identify as an individual living with a disability? - by Age Group", fontsize=16, pad=60)
ax.set_xticklabels([])
add_labels_inside_bar(ax, order)
plt.ylabel("Number of Employees")
plt.xlabel("Age Group")
plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(top=0.88)
st.pyplot(fig)

# -----------------------------
# Disabilities by Gender Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(14,8))
order = df_filtered[gender_col].value_counts().index
sns.countplot(
    x=df_filtered[gender_col],
    hue=df_filtered[disability_col],
    palette=sns.color_palette("deep", len(df_filtered[disability_col].unique())).as_hex(),
    order=order,
    ax=ax
)
ax.set_title("Do you identify as an individual living with a disability? - by Gender", fontsize=16, pad=60)
ax.set_xticklabels([])
add_labels_inside_bar(ax, order)
plt.ylabel("Number of Employees")
plt.xlabel("Gender")
plt.legend(title="Disability Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(top=0.88)
st.pyplot(fig)
