# ================================================================
# Streamlit Employee Survey Dashboard - Enhanced Version
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io

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

# Define distinct color palettes
DISTINCT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                   '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#16a085']

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
# TITLE & PDF DOWNLOAD
# ================================================================
col_title, col_download = st.columns([4, 1])

with col_title:
    st.markdown("<h1>🏠 Homes First Employee Survey Dashboard</h1>", unsafe_allow_html=True)

with col_download:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    if st.button("📥 Download PDF Report", use_container_width=True, type="primary"):
        st.session_state['generate_pdf'] = True

# ================================================================
# LOAD DATA
# ================================================================
try:
    # Read the file directly from the repository
    df = pd.read_excel('EE SurveyAnalysis.xlsx')
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ Error: 'EE SurveyAnalysis.xlsx' not found. Please ensure the file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.stop()

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
enrolled_col = [c for c in df.columns if "enrolled in school" in c.lower()][0]
apply_col = [c for c in df.columns if "how did you apply" in c.lower()][0]

# Convert to string where needed
for col in [recognized_col, growth_col, impact_col, training_pref_col, fulfillment_col, disability_col, enrolled_col, apply_col]:
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
    # HELPER FUNCTIONS
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
        return ['\n'.join(textwrap.wrap(str(label), max_width)) for label in labels]

    # ================================================================
    # CHART 1: JOB FULFILLMENT - DONUT CHART
    # ================================================================
    st.markdown("---")
    st.markdown("### 💼 Job Fulfillment Analysis")
    
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Donut chart
    fulfillment_counts = df_filtered[fulfillment_col].value_counts()
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    
    wedges, texts, autotexts = ax1a.pie(
        fulfillment_counts.values, 
        labels=None,
        autopct='%1.1f%%',
        colors=colors[:len(fulfillment_counts)],
        startangle=90,
        pctdistance=0.85,
        explode=[0.05] * len(fulfillment_counts),
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3)
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax1a.set_title('Distribution of Fulfillment Responses', 
                   fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    
    # Add legend with wrapped labels
    wrapped_labels = wrap_labels(fulfillment_counts.index, max_width=30)
    legend_labels = [f'{label}\n({count} responses)' for label, count in zip(wrapped_labels, fulfillment_counts.values)]
    ax1a.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    # Right: Bar chart for comparison
    sns.barplot(
        y=fulfillment_counts.index,
        x=fulfillment_counts.values,
        palette=colors[:len(fulfillment_counts)],
        ax=ax1b,
        edgecolor='white',
        linewidth=2,
        orient='h'
    )
    
    ax1b.set_title('Response Count Breakdown', 
                   fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax1b.set_xlabel('Number of Responses', fontsize=12, fontweight='600')
    ax1b.set_ylabel('')
    wrapped_y_labels = wrap_labels(fulfillment_counts.index, max_width=30)
    ax1b.set_yticklabels(wrapped_y_labels, fontsize=9)
    ax1b.spines['top'].set_visible(False)
    ax1b.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(ax1b.patches, fulfillment_counts.values)):
        ax1b.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{int(val)}', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    
    plt.tight_layout()
    st.pyplot(fig1)

    # ================================================================
    # CHART 2: TRAINING PREFERENCES - LOLLIPOP CHART
    # ================================================================
    st.markdown("---")
    st.markdown("### 📚 Training Preferences")
    
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    training_counts = df_filtered[training_pref_col].value_counts().sort_values(ascending=True)
    
    colors2 = ['#11998e', '#38ef7d', '#96e6a1', '#d4fc79', '#20E3B2', '#29FFC6']
    
    # Create lollipop chart
    wrapped_training = wrap_labels(training_counts.index, max_width=40)
    y_pos = np.arange(len(training_counts))
    
    # Plot stems
    ax2.hlines(y=y_pos, xmin=0, xmax=training_counts.values, 
               color='#bdc3c7', alpha=0.4, linewidth=2)
    
    # Plot lollipop heads
    ax2.scatter(training_counts.values, y_pos, 
                color=colors2[:len(training_counts)], 
                s=500, zorder=3, edgecolors='white', linewidth=3)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(training_counts.values, y_pos)):
        ax2.text(x + 0.5, y, f'{int(x)}', 
                va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(wrapped_training, fontsize=10)
    ax2.set_title('Training Preference Distribution (Virtual vs In-Person)',
                 fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
    ax2.set_xlabel('Number of Responses', fontsize=13, fontweight='600')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig2)

    # ================================================================
    # NEW CROSS-ANALYSIS SECTION
    # ================================================================
    st.markdown("---")
    st.markdown("## 🔄 Cross-Analysis Insights")

    # ================================================================
    # CHART 3: RECOMMENDATION SCORE BY ROLE - GRADIENT BARS
    # ================================================================
    st.markdown("### 📊 Recommendation Score by Role/Department")
    
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    
    # Calculate average recommendation by role
    df_filtered['recommend_numeric'] = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
    role_recommend = df_filtered.groupby(role_col)['recommend_numeric'].agg(['mean', 'count']).reset_index()
    role_recommend = role_recommend[role_recommend['count'] >= 1]
    role_recommend = role_recommend.sort_values('mean', ascending=True)
    
    # Create gradient colors based on score
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('recommendation', ['#e74c3c', '#f39c12', '#2ecc71'])
    colors3 = [cmap(val/10) for val in role_recommend['mean']]
    
    bars = ax3.barh(role_recommend[role_col], role_recommend['mean'], 
                    color=colors3, edgecolor='white', linewidth=3)
    
    # Add value labels with background
    for i, (bar, val, count) in enumerate(zip(bars, role_recommend['mean'], role_recommend['count'])):
        # Add a white background box for better readability
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
        ax3.text(val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}/10 (n={int(count)})',
                va='center', fontsize=10, fontweight='bold', 
                color='#2c3e50', bbox=bbox_props)
    
    ax3.set_title('Average Recommendation Score by Role (Color-coded by Score)', 
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax3.set_xlabel('Average Score (0-10)', fontsize=13, fontweight='600')
    ax3.set_ylabel('Role/Department', fontsize=13, fontweight='600')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim(0, 11)
    ax3.axvline(x=8, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5, label='Promoter Threshold (8+)')
    ax3.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)

    # ================================================================
    # CHART 4: FULFILLMENT BY YEARS OF EMPLOYMENT
    # ================================================================
    st.markdown("---")
    st.markdown("### ⏳ Job Fulfillment by Years of Employment")
    
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    
    # Create crosstab
    fulfillment_years = pd.crosstab(df_filtered[years_col], df_filtered[fulfillment_col])
    
    # Use distinct colors
    n_categories = len(fulfillment_years.columns)
    distinct_palette = DISTINCT_COLORS[:n_categories]
    
    fulfillment_years.plot(kind='bar', stacked=False, ax=ax4, 
                          color=distinct_palette, 
                          edgecolor='white', linewidth=1.5)
    
    ax4.set_title('Job Fulfillment by Years of Employment', 
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax4.set_xlabel('Years Employed', fontsize=13, fontweight='600')
    ax4.set_ylabel('Number of Employees', fontsize=13, fontweight='600')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.legend(title='Fulfillment Level', bbox_to_anchor=(1.05, 1), loc='upper left', 
              frameon=True, shadow=True, fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig4)

    # ================================================================
    # CHART 5: GROWTH POTENTIAL PERCEPTION BY AGE
    # ================================================================
    st.markdown("---")
    st.markdown("### 📈 Growth Potential Perception by Age Group")
    
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    
    growth_age = pd.crosstab(df_filtered[age_col], df_filtered[growth_col], normalize='index') * 100
    
    # Use distinct colors for growth
    n_growth = len(growth_age.columns)
    growth_palette = [DISTINCT_COLORS[i] for i in [0, 1, 2, 3, 4, 5, 6, 7][:n_growth]]
    
    growth_age.plot(kind='bar', stacked=True, ax=ax5,
                   color=growth_palette,
                   edgecolor='white', linewidth=1.5)
    
    ax5.set_title('Growth Potential Perception by Age Group (%)', 
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax5.set_xlabel('Age Group', fontsize=13, fontweight='600')
    ax5.set_ylabel('Percentage (%)', fontsize=13, fontweight='600')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.legend(title='Growth Perception', bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, shadow=True, fontsize=9)
    ax5.set_ylim(0, 100)
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    st.pyplot(fig5)

    # ================================================================
    # CHART 6: RECOGNITION VS RECOMMENDATION CORRELATION
    # ================================================================
    st.markdown("---")
    st.markdown("### 🎯 Recognition vs Recommendation Score")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig6a, ax6a = plt.subplots(figsize=(10, 6))
        
        recognition_recommend = df_filtered.groupby(recognized_col)['recommend_numeric'].mean().sort_values(ascending=False)
        
        n_recog = len(recognition_recommend)
        recog_colors = [DISTINCT_COLORS[i] for i in range(n_recog)]
        
        bars = ax6a.barh(range(len(recognition_recommend)), recognition_recommend.values,
                       color=recog_colors,
                       edgecolor='white', linewidth=2)
        
        ax6a.set_yticks(range(len(recognition_recommend)))
        wrapped_recog_labels = wrap_labels(recognition_recommend.index, max_width=25)
        ax6a.set_yticklabels(wrapped_recog_labels, fontsize=9)
        
        for i, (bar, val) in enumerate(zip(bars, recognition_recommend.values)):
            ax6a.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}/10',
                    va='center', fontsize=10, fontweight='bold')
        
        ax6a.set_title('Avg Recommendation by Recognition Level', 
                     fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        ax6a.set_xlabel('Average Recommendation Score', fontsize=11, fontweight='600')
        ax6a.spines['top'].set_visible(False)
        ax6a.spines['right'].set_visible(False)
        ax6a.set_xlim(0, 11)
        plt.tight_layout()
        st.pyplot(fig6a)
    
    with col_right:
        fig6b, ax6b = plt.subplots(figsize=(10, 6))
        
        impact_recommend = df_filtered.groupby(impact_col)['recommend_numeric'].mean().sort_values(ascending=False)
        
        n_impact = len(impact_recommend)
        impact_colors = [DISTINCT_COLORS[i] for i in [1, 3, 5, 7][:n_impact]]
        
        bars = ax6b.barh(range(len(impact_recommend)), impact_recommend.values,
                       color=impact_colors,
                       edgecolor='white', linewidth=2)
        
        ax6b.set_yticks(range(len(impact_recommend)))
        wrapped_impact_labels = wrap_labels(impact_recommend.index, max_width=25)
        ax6b.set_yticklabels(wrapped_impact_labels, fontsize=9)
        
        for i, (bar, val) in enumerate(zip(bars, impact_recommend.values)):
            ax6b.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}/10',
                    va='center', fontsize=10, fontweight='bold')
        
        ax6b.set_title('Avg Recommendation by Impact Perception', 
                     fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        ax6b.set_xlabel('Average Recommendation Score', fontsize=11, fontweight='600')
        ax6b.spines['top'].set_visible(False)
        ax6b.spines['right'].set_visible(False)
        ax6b.set_xlim(0, 11)
        plt.tight_layout()
        st.pyplot(fig6b)

    # ================================================================
    # CHART 7: APPLICATION METHOD ANALYSIS
    # ================================================================
    st.markdown("---")
    st.markdown("### 🚪 How Employees Joined Homes First")
    
    fig7, ax7 = plt.subplots(figsize=(16, 9))
    apply_counts = df_filtered[apply_col].value_counts().sort_values(ascending=True)
    
    # Create gradient color scheme
    colors7 = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Horizontal bars with shadow effect
    y_pos = np.arange(len(apply_counts))
    bars = ax7.barh(y_pos, apply_counts.values,
                   color=colors7[:len(apply_counts)], 
                   edgecolor='white', linewidth=3,
                   height=0.7)
    
    # Add percentage labels inside bars
    total_responses = apply_counts.sum()
    for i, (bar, val) in enumerate(zip(bars, apply_counts.values)):
        percentage = (val / total_responses) * 100
        ax7.text(val/2, bar.get_y() + bar.get_height()/2,
               f'{int(val)} ({percentage:.1f}%)', 
               ha='center', va='center',
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))
    
    wrapped_apply_labels = wrap_labels(apply_counts.index, max_width=50)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(wrapped_apply_labels, fontsize=11)
    
    ax7.set_title('Distribution of Application Methods', 
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax7.set_xlabel('Number of Employees', fontsize=13, fontweight='600')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.spines['left'].set_visible(False)
    ax7.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig7)

    # ================================================================
    # CHART 8: DISABILITIES BY AGE
    # ================================================================
    st.markdown("---")
    st.markdown("### 🔍 Disability Analysis by Age Group")
    
    df_filtered[disability_col] = df_filtered[disability_col].fillna("No Disability")
    df_filtered[disability_col] = df_filtered[disability_col].replace('nan', 'No Disability')
    
    unique_disability_values = sorted(df_filtered[disability_col].unique())
    disability_palette = {val: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, val in enumerate(unique_disability_values)}
    
    fig8, ax8 = plt.subplots(figsize=(16, 8))
    
    sns.countplot(
        data=df_filtered,
        x=age_col,
        hue=disability_col,
        palette=disability_palette,
        order=sorted(df_filtered[age_col].unique()),
        ax=ax8,
        edgecolor='white',
        linewidth=1.5
    )
    
    ax8.set_title('Do you identify as an individual living with a disability? - by Age Group',
                 fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax8.set_xlabel('Age Group', fontsize=12, fontweight='600')
    ax8.set_ylabel('Number of Employees', fontsize=12, fontweight='600')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.legend(title='Disability Status', title_fontsize=11, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    st.pyplot(fig8)

    # ================================================================
    # CHART 9: DISABILITIES BY GENDER
    # ================================================================
    st.markdown("---")
    st.markdown("### 🔍 Disability Analysis by Gender")
    
    fig9, ax9 = plt.subplots(figsize=(16, 8))
    
    sns.countplot(
        data=df_filtered,
        x=gender_col,
        hue=disability_col,
        palette=disability_palette,
        order=sorted(df_filtered[gender_col].unique()),
        ax=ax9,
        edgecolor='white',
        linewidth=1.5
    )
    
    ax9.set_title('Do you identify as an individual living with a disability? - by Gender',
                 fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax9.set_xlabel('Gender', fontsize=12, fontweight='600')
    ax9.set_ylabel('Number of Employees', fontsize=12, fontweight='600')
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.legend(title='Disability Status', title_fontsize=11, fontsize=10,
              bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    
    # Wrap x-axis labels for gender chart
    gender_labels = [label.get_text() for label in ax9.get_xticklabels()]
    wrapped_gender_labels = wrap_labels(gender_labels, max_width=20)
    ax9.set_xticklabels(wrapped_gender_labels, rotation=0, ha='center')
    
    # Add extra space at bottom for labels
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    st.pyplot(fig9)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; padding: 20px;'>Dashboard created with Streamlit • Homes First Employee Survey 2024</p>", unsafe_allow_html=True)

# ================================================================
# PDF GENERATION FUNCTION
# ================================================================
def generate_pdf_report():
    """Generate a comprehensive PDF report of the dashboard"""
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    # Create PDF with multiple pages
    with PdfPages(buffer) as pdf:
        # ============ PAGE 1: COVER PAGE ============
        fig_cover = plt.figure(figsize=(11, 8.5))
        ax_cover = fig_cover.add_subplot(111)
        ax_cover.axis('off')
        
        # Add title and metadata
        fig_cover.text(0.5, 0.7, '🏠 Homes First Employee Survey', 
                      ha='center', fontsize=32, fontweight='bold', color='#2c3e50')
        fig_cover.text(0.5, 0.6, 'Comprehensive Analysis Report', 
                      ha='center', fontsize=20, color='#34495e')
        fig_cover.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 
                      ha='center', fontsize=14, color='#7f8c8d')
        fig_cover.text(0.5, 0.4, f'Total Respondents: {df_filtered.shape[0]}', 
                      ha='center', fontsize=16, fontweight='bold', color='#3498db')
        
        # Add filter information
        filter_text = f"Filters Applied:\n"
        filter_text += f"• Roles: {', '.join(role_filter) if len(role_filter) < 5 else f'{len(role_filter)} selected'}\n"
        filter_text += f"• Age Groups: {', '.join(age_filter) if len(age_filter) < 5 else f'{len(age_filter)} selected'}\n"
        filter_text += f"• Genders: {', '.join(gender_filter) if len(gender_filter) < 5 else f'{len(gender_filter)} selected'}"
        
        fig_cover.text(0.5, 0.25, filter_text, 
                      ha='center', fontsize=11, color='#7f8c8d',
                      bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
        
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)
        
        # ============ PAGE 2: KPI SUMMARY ============
        fig_kpi = plt.figure(figsize=(11, 8.5))
        ax_kpi = fig_kpi.add_subplot(111)
        ax_kpi.axis('off')
        
        fig_kpi.text(0.5, 0.95, 'Key Performance Indicators', 
                    ha='center', fontsize=24, fontweight='bold', color='#2c3e50')
        
        # KPI boxes
        kpis = [
            ('Recommendation Rate', f'{positive_rec}/{total}', f'Avg: {avg_recommend:.1f}/10', '#667eea'),
            ('Feel Recognized', f'{recognized}/{total}', f'{recognized/total*100:.1f}%', '#11998e'),
            ('See Growth Potential', f'{growth}/{total}', f'{growth/total*100:.1f}%', '#ee0979'),
            ('Feel Positive Impact', f'{impact}/{total}', f'{impact/total*100:.1f}%', '#f857a6')
        ]
        
        y_start = 0.75
        for i, (title, value, subtitle, color) in enumerate(kpis):
            y_pos = y_start - (i * 0.18)
            
            # Draw box
            rect = plt.Rectangle((0.15, y_pos-0.08), 0.7, 0.12, 
                                facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax_kpi.add_patch(rect)
            
            fig_kpi.text(0.25, y_pos, title, fontsize=14, fontweight='bold', color='#2c3e50')
            fig_kpi.text(0.75, y_pos, value, fontsize=18, fontweight='bold', 
                        color=color, ha='right')
            fig_kpi.text(0.25, y_pos-0.04, subtitle, fontsize=11, color='#7f8c8d')
        
        pdf.savefig(fig_kpi, bbox_inches='tight')
        plt.close(fig_kpi)
        
        # ============ PAGE 3: JOB FULFILLMENT ============
        fig_fulfill, (ax_donut, ax_bar) = plt.subplots(1, 2, figsize=(11, 8.5))
        
        fulfillment_counts = df_filtered[fulfillment_col].value_counts()
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
        
        # Donut chart
        wedges, texts, autotexts = ax_donut.pie(
            fulfillment_counts.values, 
            labels=None,
            autopct='%1.1f%%',
            colors=colors[:len(fulfillment_counts)],
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax_donut.set_title('Job Fulfillment Distribution', fontsize=14, fontweight='bold', pad=15)
        wrapped_labels = wrap_labels(fulfillment_counts.index, max_width=25)
        ax_donut.legend(wrapped_labels, loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
        
        # Bar chart
        sns.barplot(y=fulfillment_counts.index, x=fulfillment_counts.values,
                   palette=colors[:len(fulfillment_counts)], ax=ax_bar,
                   edgecolor='white', linewidth=2, orient='h')
        ax_bar.set_title('Response Counts', fontsize=14, fontweight='bold', pad=15)
        ax_bar.set_xlabel('Number of Responses', fontsize=11)
        ax_bar.set_ylabel('')
        wrapped_y = wrap_labels(fulfillment_counts.index, max_width=25)
        ax_bar.set_yticklabels(wrapped_y, fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig_fulfill, bbox_inches='tight')
        plt.close(fig_fulfill)
        
        # ============ PAGE 4: TRAINING PREFERENCES ============
        fig_train, ax_train = plt.subplots(figsize=(11, 8.5))
        
        training_counts = df_filtered[training_pref_col].value_counts().sort_values(ascending=True)
        colors2 = ['#11998e', '#38ef7d', '#96e6a1', '#d4fc79', '#20E3B2', '#29FFC6']
        
        wrapped_training = wrap_labels(training_counts.index, max_width=35)
        y_pos = np.arange(len(training_counts))
        
        ax_train.hlines(y=y_pos, xmin=0, xmax=training_counts.values, 
                       color='#bdc3c7', alpha=0.4, linewidth=2)
        ax_train.scatter(training_counts.values, y_pos, 
                        color=colors2[:len(training_counts)], 
                        s=400, zorder=3, edgecolors='white', linewidth=2)
        
        for i, (x, y) in enumerate(zip(training_counts.values, y_pos)):
            ax_train.text(x + 0.5, y, f'{int(x)}', 
                        va='center', fontsize=10, fontweight='bold')
        
        ax_train.set_yticks(y_pos)
        ax_train.set_yticklabels(wrapped_training, fontsize=9)
        ax_train.set_title('Training Preferences (Virtual vs In-Person)',
                          fontsize=16, fontweight='bold', pad=20)
        ax_train.set_xlabel('Number of Responses', fontsize=12)
        ax_train.spines['top'].set_visible(False)
        ax_train.spines['right'].set_visible(False)
        ax_train.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        pdf.savefig(fig_train, bbox_inches='tight')
        plt.close(fig_train)
        
        # ============ PAGE 5: RECOMMENDATION BY ROLE ============
        fig_role, ax_role = plt.subplots(figsize=(11, 8.5))
        
        role_recommend = df_filtered.groupby(role_col)['recommend_numeric'].agg(['mean', 'count']).reset_index()
        role_recommend = role_recommend[role_recommend['count'] >= 1]
        role_recommend = role_recommend.sort_values('mean', ascending=True)
        
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('recommendation', ['#e74c3c', '#f39c12', '#2ecc71'])
        colors_role = [cmap(val/10) for val in role_recommend['mean']]
        
        bars = ax_role.barh(role_recommend[role_col], role_recommend['mean'], 
                           color=colors_role, edgecolor='white', linewidth=2)
        
        for i, (bar, val, count) in enumerate(zip(bars, role_recommend['mean'], role_recommend['count'])):
            ax_role.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f} (n={int(count)})',
                        va='center', fontsize=9, fontweight='bold')
        
        ax_role.set_title('Average Recommendation Score by Role',
                         fontsize=16, fontweight='bold', pad=20)
        ax_role.set_xlabel('Average Score (0-10)', fontsize=12)
        ax_role.set_ylabel('Role/Department', fontsize=12)
        ax_role.axvline(x=8, color='#27ae60', linestyle='--', linewidth=2, alpha=0.5)
        ax_role.set_xlim(0, 11)
        
        plt.tight_layout()
        pdf.savefig(fig_role, bbox_inches='tight')
        plt.close(fig_role)
        
        # ============ PAGE 6: CROSS-ANALYSIS ============
        fig_cross, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
        
        # 1. Fulfillment by Years
        fulfillment_years = pd.crosstab(df_filtered[years_col], df_filtered[fulfillment_col])
        fulfillment_years.plot(kind='bar', stacked=False, ax=ax1, 
                              color=DISTINCT_COLORS[:len(fulfillment_years.columns)],
                              edgecolor='white', linewidth=1, legend=False)
        ax1.set_title('Fulfillment by Years Employed', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Years', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='both', labelsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Growth by Age
        growth_age = pd.crosstab(df_filtered[age_col], df_filtered[growth_col], normalize='index') * 100
        growth_age.plot(kind='bar', stacked=True, ax=ax2,
                       color=DISTINCT_COLORS[:len(growth_age.columns)],
                       edgecolor='white', linewidth=1, legend=False)
        ax2.set_title('Growth Perception by Age (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Age Group', fontsize=10)
        ax2.set_ylabel('Percentage', fontsize=10)
        ax2.tick_params(axis='both', labelsize=8)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        
        # 3. Recognition vs Recommendation
        recognition_recommend = df_filtered.groupby(recognized_col)['recommend_numeric'].mean().sort_values(ascending=True)
        wrapped_recog = wrap_labels(recognition_recommend.index, max_width=20)
        ax3.barh(range(len(recognition_recommend)), recognition_recommend.values,
                color=DISTINCT_COLORS[:len(recognition_recommend)], edgecolor='white', linewidth=1)
        ax3.set_yticks(range(len(recognition_recommend)))
        ax3.set_yticklabels(wrapped_recog, fontsize=8)
        ax3.set_title('Recommendation by Recognition', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Avg Score', fontsize=10)
        
        # 4. Application Methods
        apply_counts = df_filtered[apply_col].value_counts().sort_values(ascending=True)
        wrapped_apply = wrap_labels(apply_counts.index, max_width=25)
        ax4.barh(range(len(apply_counts)), apply_counts.values,
                color=DISTINCT_COLORS[:len(apply_counts)], edgecolor='white', linewidth=1)
        ax4.set_yticks(range(len(apply_counts)))
        ax4.set_yticklabels(wrapped_apply, fontsize=8)
        ax4.set_title('Application Methods', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Count', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig_cross, bbox_inches='tight')
        plt.close(fig_cross)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Homes First Employee Survey Report'
        d['Author'] = 'Homes First Dashboard'
        d['Subject'] = 'Employee Survey Analysis'
        d['Keywords'] = 'Employee Engagement, Survey, Analytics'
        d['CreationDate'] = datetime.now()
    
    buffer.seek(0)
    return buffer

# ================================================================
# PDF DOWNLOAD BUTTON HANDLER
# ================================================================
if 'generate_pdf' in st.session_state and st.session_state['generate_pdf']:
    with st.spinner('📄 Generating PDF report... This may take a moment.'):
        pdf_buffer = generate_pdf_report()
        
        st.download_button(
            label="⬇️ Click Here to Download Your PDF Report",
            data=pdf_buffer,
            file_name=f"Homes_First_Survey_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        
        st.success("✅ PDF report generated successfully!")
        st.session_state['generate_pdf'] = False
