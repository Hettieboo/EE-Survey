# ================================================================
# Streamlit Employee Survey Dashboard - Enhanced Version
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

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
                     fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
        ax1.set_xlabel('')
        ax1.set_ylabel('Number of Responses', fontsize=13, fontweight='600')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        wrapped_labels = wrap_labels(fulfillment_counts.index, max_width=35)
        ax1.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=10)
        
        # Add extra space at bottom for labels
        plt.subplots_adjust(bottom=0.25)
        
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
                     fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
        ax2.set_xlabel('')
        ax2.set_ylabel('Number of Responses', fontsize=13, fontweight='600')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        wrapped_labels = wrap_labels(training_counts.index, max_width=35)
        ax2.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=10)
        
        # Add extra space at bottom for labels
        plt.subplots_adjust(bottom=0.25)
        
        add_value_labels(ax2, spacing=3)
        plt.tight_layout()
        st.pyplot(fig2)

        # ================================================================
        # NEW CROSS-ANALYSIS SECTION
        # ================================================================
        st.markdown("---")
        st.markdown("## 🔄 Cross-Analysis Insights")

        # ================================================================
        # CHART 3: RECOMMENDATION SCORE BY ROLE
        # ================================================================
        st.markdown("### 📊 Recommendation Score by Role/Department")
        
        fig3, ax3 = plt.subplots(figsize=(16, 8))
        
        # Calculate average recommendation by role
        df_filtered['recommend_numeric'] = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
        role_recommend = df_filtered.groupby(role_col)['recommend_numeric'].agg(['mean', 'count']).reset_index()
        role_recommend = role_recommend[role_recommend['count'] >= 1]
        role_recommend = role_recommend.sort_values('mean', ascending=False)
        
        colors3 = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        bars = ax3.barh(role_recommend[role_col], role_recommend['mean'], 
                        color=colors3[:len(role_recommend)], edgecolor='white', linewidth=2)
        
        # Add value labels
        for i, (bar, val, count) in enumerate(zip(bars, role_recommend['mean'], role_recommend['count'])):
            ax3.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}/10 (n={int(count)})',
                    va='center', fontsize=10, fontweight='bold', color='#2c3e50')
        
        ax3.set_title('Average Recommendation Score by Role', 
                     fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
        ax3.set_xlabel('Average Score (0-10)', fontsize=13, fontweight='600')
        ax3.set_ylabel('')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_xlim(0, 11)
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
        
        fig7, ax7 = plt.subplots(figsize=(16, 8))
        apply_counts = df_filtered[apply_col].value_counts()
        
        colors7 = ['#667eea', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax7.bar(range(len(apply_counts)), apply_counts.values,
                      color=colors7[:len(apply_counts)], edgecolor='white', linewidth=2)
        
        wrapped_apply_labels = wrap_labels(apply_counts.index, max_width=30)
        ax7.set_xticks(range(len(apply_counts)))
        ax7.set_xticklabels(wrapped_apply_labels, rotation=0, ha='center', fontsize=10)
        
        for bar, val in zip(bars, apply_counts.values):
            ax7.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                   f'{int(val)}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color='#2c3e50')
        
        ax7.set_title('Application Methods', 
                     fontsize=18, fontweight='bold', pad=30, color='#2c3e50')
        ax7.set_ylabel('Number of Employees', fontsize=13, fontweight='600')
        ax7.set_xlabel('')
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        
        # Add extra space at bottom for labels
        plt.subplots_adjust(bottom=0.25)
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

else:
    st.info("👆 Please upload an Excel file to begin analyzing the survey data.")
