# ================================================================
# Streamlit Employee Survey Dashboard - Enhanced Version
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ================================================================
# CONFIGURATION & STYLING
# ================================================================

# Set page config FIRST
st.set_page_config(page_title="Homes First Survey Dashboard", layout="wide")

# Seaborn and matplotlib styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main title styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        border-bottom: 3px solid #3498db;
    }
    
    /* Section headers */
    h2 {
        color: #34495e;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        padding-left: 10px;
        border-left: 5px solid #3498db;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Metric containers */
    .metric-container {
        background: linear-gradient(135deg, var(--color1), var(--color2));
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .metric-subtitle {
        font-size: 13px;
        opacity: 0.9;
    }
    
    /* Filter section */
    .filter-header {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# TITLE
# ================================================================
st.markdown("<h1>🏠 Homes First Employee Survey Dashboard</h1>", unsafe_allow_html=True)

# ================================================================
# FILE UPLOAD
# ================================================================
uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # ================================================================
    # IDENTIFY KEY COLUMNS
    # ================================================================
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

    # ================================================================
    # SIDEBAR FILTERS
    # ================================================================
    with st.sidebar:
        st.markdown('<p class="filter-header">🔍 Filter Options</p>', unsafe_allow_html=True)
        
        role_filter = st.multiselect(
            "Role/Department",
            options=sorted(df[role_col].unique()),
            default=df[role_col].unique()
        )
        
        age_filter = st.multiselect(
            "Age Group",
            options=sorted(df[age_col].unique()),
            default=df[age_col].unique()
        )
        
        gender_filter = st.multiselect(
            "Gender",
            options=sorted(df[gender_col].unique()),
            default=df[gender_col].unique()
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"**Total Respondents:** {len(df)}")

    # Apply filters
    df_filtered = df[
        df[role_col].isin(role_filter) & 
        df[age_col].isin(age_filter) & 
        df[gender_col].isin(gender_filter)
    ]
    
    st.markdown(f"<div style='text-align: center; padding: 15px; background-color: #e8f4f8; border-radius: 10px; margin-bottom: 30px;'>"
                f"<strong>Showing data for {df_filtered.shape[0]} respondents</strong></div>", 
                unsafe_allow_html=True)

    # ================================================================
    # KPI METRICS
    # ================================================================
    total = df_filtered.shape[0]
    
    if total == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
    else:
        # Calculate metrics
        recommend_scores = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
        positive_rec = recommend_scores[recommend_scores >= 8].count()
        avg_recommend = recommend_scores.mean()
        
        recognized = df_filtered[df_filtered[recognized_col].str.lower().str.contains("yes|somewhat", na=False)].shape[0]
        growth = df_filtered[df_filtered[growth_col].str.lower().str.contains("yes", na=False)].shape[0]
        impact = df_filtered[df_filtered[impact_col].str.lower().str.contains("positive impact", na=False)].shape[0]

        # Display KPIs in centered grid
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-container" style="--color1: #667eea; --color2: #764ba2;">
                    <div class="metric-title">Recommendation Rate</div>
                    <div class="metric-value">{positive_rec}/{total}</div>
                    <div class="metric-subtitle">Avg: {avg_recommend:.1f}/10</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-container" style="--color1: #11998e; --color2: #38ef7d;">
                    <div class="metric-title">Feel Recognized</div>
                    <div class="metric-value">{recognized}/{total}</div>
                    <div class="metric-subtitle">{recognized/total*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-container" style="--color1: #ee0979; --color2: #ff6a00;">
                    <div class="metric-title">See Growth Potential</div>
                    <div class="metric-value">{growth}/{total}</div>
                    <div class="metric-subtitle">{growth/total*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-container" style="--color1: #f857a6; --color2: #ff5858;">
                    <div class="metric-title">Feel Positive Impact</div>
                    <div class="metric-value">{impact}/{total}</div>
                    <div class="metric-subtitle">{impact/total*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # ================================================================
        # HELPER FUNCTION FOR BAR LABELS
        # ================================================================
        def add_value_labels(ax, spacing=0):
            """Add labels to the end of each bar in a bar chart."""
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center',
                                va='bottom',
                                fontsize=11,
                                fontweight='bold',
                                color='#2c3e50',
                                xytext=(0, spacing),
                                textcoords='offset points')
        
        def wrap_labels(labels, max_width=40):
            """Wrap long labels to multiple lines."""
            import textwrap
            return ['\n'.join(textwrap.wrap(label, max_width)) for label in labels]

        # ================================================================
        # CHART 1: JOB FULFILLMENT
        # ================================================================
        st.markdown("---")
        st.markdown("### 💼 Job Fulfillment Analysis")
        
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        fulfillment_counts = df_filtered[fulfillment_col].value_counts()
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        sns.barplot(
            x=fulfillment_counts.index,
            y=fulfillment_counts.values,
            palette=colors[:len(fulfillment_counts)],
            ax=ax1,
            edgecolor='white',
            linewidth=2
        )
        
        ax1.set_title('How fulfilling and rewarding do you find your work?', 
                     fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
        ax1.set_xlabel('')
        ax1.set_ylabel('Number of Responses', fontsize=13, fontweight='600')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Wrap x-axis labels
        wrapped_labels = wrap_labels(fulfillment_counts.index, max_width=35)
        ax1.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=10)
        
        add_value_labels(ax1, spacing=3)
        plt.tight_layout()
        st.pyplot(fig1)

        # ================================================================
        # CHART 2: TRAINING PREFERENCES
        # ================================================================
        st.markdown("---")
        st.markdown("### 📚 Training Preferences")
        
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        training_counts = df_filtered[training_pref_col].value_counts()
        
        colors2 = ['#11998e', '#38ef7d', '#96e6a1', '#d4fc79']
        sns.barplot(
            x=training_counts.index,
            y=training_counts.values,
            palette=colors2[:len(training_counts)],
            ax=ax2,
            edgecolor='white',
            linewidth=2
        )
        
        ax2.set_title('How do you feel about live virtual training (Zoom/Teams) vs in-person?',
                     fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
        ax2.set_xlabel('')
        ax2.set_ylabel('Number of Responses', fontsize=13, fontweight='600')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Wrap x-axis labels
        wrapped_labels = wrap_labels(training_counts.index, max_width=35)
        ax2.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=10)
        
        add_value_labels(ax2, spacing=3)
        plt.tight_layout()
        st.pyplot(fig2)

        # ================================================================
        # CHART 3: DISABILITIES BY AGE
        # ================================================================
        st.markdown("---")
        st.markdown("### 🔍 Disability Analysis by Age Group")
        
        # Clean up disability column
        df_filtered[disability_col] = df_filtered[disability_col].fillna("No Disability")
        df_filtered[disability_col] = df_filtered[disability_col].replace('nan', 'No Disability')
        
        # Get unique disability values and create dynamic palette
        unique_disability_values = df_filtered[disability_col].unique()
        base_colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71', '#f39c12', '#9b59b6']
        disability_palette = {val: base_colors[i % len(base_colors)] for i, val in enumerate(unique_disability_values)}
        
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        
        sns.countplot(
            data=df_filtered,
            x=age_col,
            hue=disability_col,
            palette=disability_palette,
            order=sorted(df_filtered[age_col].unique()),
            ax=ax3,
            edgecolor='white',
            linewidth=1.5
        )
        
        ax3.set_title('Do you identify as an individual living with a disability? - by Age Group',
                     fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        ax3.set_xlabel('Age Group', fontsize=12, fontweight='600')
        ax3.set_ylabel('Number of Employees', fontsize=12, fontweight='600')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.legend(title='Disability Status', title_fontsize=11, fontsize=10, 
                  loc='upper right', frameon=True, shadow=True)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        st.pyplot(fig3)

        # ================================================================
        # CHART 4: DISABILITIES BY GENDER
        # ================================================================
        st.markdown("---")
        st.markdown("### 🔍 Disability Analysis by Gender")
        
        fig4, ax4 = plt.subplots(figsize=(14, 7))
        
        sns.countplot(
            data=df_filtered,
            x=gender_col,
            hue=disability_col,
            palette=disability_palette,
            order=sorted(df_filtered[gender_col].unique()),
            ax=ax4,
            edgecolor='white',
            linewidth=1.5
        )
        
        ax4.set_title('Do you identify as an individual living with a disability? - by Gender',
                     fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        ax4.set_xlabel('Gender', fontsize=12, fontweight='600')
        ax4.set_ylabel('Number of Employees', fontsize=12, fontweight='600')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.legend(title='Disability Status', title_fontsize=11, fontsize=10,
                  loc='upper right', frameon=True, shadow=True)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #7f8c8d; padding: 20px;'>Dashboard created with Streamlit • Homes First Employee Survey 2024</p>", unsafe_allow_html=True)

else:
    st.info("👆 Please upload an Excel file to begin analyzing the survey data.")
