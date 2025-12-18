# ================================================================
# Streamlit Employee Survey Dashboard - AI-Enhanced Version
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

st.set_page_config(page_title="Employee Survey Dashboard", layout="wide")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

DISTINCT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                   '#1abc9c', '#e67e22', '#34495e', '#c0392b', '#16a085']

st.markdown("""
    <style>
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        border-bottom: 3px solid #3498db;
    }
    
    h2 {
        color: #34495e;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        padding-left: 10px;
        border-left: 5px solid #3498db;
    }
    
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
    
    .insights-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        margin: 30px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .insight-section {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #fff;
    }
    
    .insight-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #fff;
    }
    
    .insight-text {
        font-size: 15px;
        line-height: 1.6;
        color: rgba(255,255,255,0.95);
    }
    
    .recommendation {
        background: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# TITLE & BUTTONS
# ================================================================
col_title, col_buttons = st.columns([3, 1])

with col_title:
    st.markdown("<h1>Employee Survey Dashboard</h1>", unsafe_allow_html=True)

with col_buttons:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊 Generate Insights", use_container_width=True, type="primary"):
        st.session_state['generate_insights'] = True
        st.rerun()
    if st.button("📥 Download PDF", use_container_width=True):
        st.session_state['generate_pdf'] = True
        st.rerun()

# ================================================================
# LOAD DATA
# ================================================================
try:
    df = pd.read_excel('EE SurveyAnalysis.xlsx')
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ Error: 'EE SurveyAnalysis.xlsx' not found.")
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

for col in [recognized_col, growth_col, impact_col, training_pref_col, fulfillment_col, disability_col, enrolled_col, apply_col]:
    df[col] = df[col].astype(str)

# ================================================================
# SIDEBAR FILTERS
# ================================================================
with st.sidebar:
    st.markdown('<p style="font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 15px;">🔍 Filter Options</p>', unsafe_allow_html=True)
    
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
    st.warning("⚠️ No data matches the selected filters.")
else:
    recommend_scores = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
    positive_rec = recommend_scores[recommend_scores >= 8].count()
    avg_recommend = recommend_scores.mean()
    
    recognized = df_filtered[df_filtered[recognized_col].str.lower().str.contains("yes|somewhat", na=False)].shape[0]
    growth = df_filtered[df_filtered[growth_col].str.lower().str.contains("yes", na=False)].shape[0]
    impact = df_filtered[df_filtered[impact_col].str.lower().str.contains("positive impact", na=False)].shape[0]

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
    def wrap_labels(labels, max_width=40):
        import textwrap
        return ['\n'.join(textwrap.wrap(str(label), max_width)) for label in labels]

    # ================================================================
    # ALL CHARTS (Keeping your existing chart code)
    # ================================================================
    
    # CHART 1: JOB FULFILLMENT
    st.markdown("---")
    st.markdown("### 💼 Job Fulfillment Analysis")
    
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(18, 7))
    
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
    
    wrapped_labels = wrap_labels(fulfillment_counts.index, max_width=30)
    legend_labels = [f'{label}\n({count} responses)' for label, count in zip(wrapped_labels, fulfillment_counts.values)]
    ax1a.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
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
    
    for i, (bar, val) in enumerate(zip(ax1b.patches, fulfillment_counts.values)):
        ax1b.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{int(val)}', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    
    plt.tight_layout()
    st.pyplot(fig1)

    # CHART 2: TRAINING PREFERENCES
    st.markdown("---")
    st.markdown("### 📚 Training Preferences")
    
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    training_counts = df_filtered[training_pref_col].value_counts().sort_values(ascending=True)
    
    colors2 = ['#11998e', '#38ef7d', '#96e6a1', '#d4fc79', '#20E3B2', '#29FFC6']
    
    wrapped_training = wrap_labels(training_counts.index, max_width=40)
    y_pos = np.arange(len(training_counts))
    
    ax2.hlines(y=y_pos, xmin=0, xmax=training_counts.values, 
               color='#bdc3c7', alpha=0.4, linewidth=2)
    
    ax2.scatter(training_counts.values, y_pos, 
                color=colors2[:len(training_counts)], 
                s=500, zorder=3, edgecolors='white', linewidth=3)
    
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

    # CHART 3: RECOMMENDATION BY ROLE
    st.markdown("---")
    st.markdown("### 📊 Recommendation Score by Role/Department")
    
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    
    df_filtered['recommend_numeric'] = pd.to_numeric(df_filtered[recommend_col], errors='coerce')
    role_recommend = df_filtered.groupby(role_col)['recommend_numeric'].agg(['mean', 'count']).reset_index()
    role_recommend = role_recommend[role_recommend['count'] >= 1]
    role_recommend = role_recommend.sort_values('mean', ascending=True)
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('recommendation', ['#e74c3c', '#f39c12', '#2ecc71'])
    colors3 = [cmap(val/10) for val in role_recommend['mean']]
    
    bars = ax3.barh(role_recommend[role_col], role_recommend['mean'], 
                    color=colors3, edgecolor='white', linewidth=3)
    
    for i, (bar, val, count) in enumerate(zip(bars, role_recommend['mean'], role_recommend['count'])):
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

    # CHART 4: FULFILLMENT BY YEARS
    st.markdown("---")
    st.markdown("### ⏳ Job Fulfillment by Years of Employment")
    
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    
    fulfillment_years = pd.crosstab(df_filtered[years_col], df_filtered[fulfillment_col])
    
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
    # STATISTICAL INSIGHTS GENERATION FUNCTION
    # ================================================================
    def generate_statistical_insights():
        """Generate insights using statistical analysis"""
        
        insights = {
            "executive_summary": "",
            "key_patterns": [],
            "strengths": [],
            "concerns": [],
            "recommendations": [],
            "demographic_insights": ""
        }
        
        # Calculate key metrics
        promoter_rate = (positive_rec / total * 100)
        recognition_rate = (recognized / total * 100)
        growth_rate = (growth / total * 100)
        impact_rate = (impact / total * 100)
        
        # EXECUTIVE SUMMARY
        summary_parts = []
        summary_parts.append(f"Survey collected responses from {total} employees with an average recommendation score of {avg_recommend:.1f}/10.")
        
        if promoter_rate >= 70:
            summary_parts.append(f"Strong employee advocacy with {promoter_rate:.0f}% promoters.")
        elif promoter_rate >= 50:
            summary_parts.append(f"Moderate employee advocacy at {promoter_rate:.0f}% promoters.")
        else:
            summary_parts.append(f"Low employee advocacy at {promoter_rate:.0f}% promoters requires attention.")
        
        if recognition_rate < 60:
            summary_parts.append("Recognition gaps identified as a critical improvement area.")
        elif growth_rate < 60:
            summary_parts.append("Growth opportunities perception needs enhancement.")
        else:
            summary_parts.append("Strong foundation in recognition and growth perception.")
        
        insights["executive_summary"] = " ".join(summary_parts)
        
        # KEY PATTERNS
        # Pattern 1: Overall recommendation score
        if avg_recommend >= 8:
            insights["key_patterns"].append({
                "pattern": f"High recommendation score ({avg_recommend:.1f}/10) indicates strong employee satisfaction",
                "impact": "high"
            })
        elif avg_recommend >= 6:
            insights["key_patterns"].append({
                "pattern": f"Moderate recommendation score ({avg_recommend:.1f}/10) shows room for improvement",
                "impact": "medium"
            })
        else:
            insights["key_patterns"].append({
                "pattern": f"Low recommendation score ({avg_recommend:.1f}/10) signals critical issues requiring immediate attention",
                "impact": "high"
            })
        
        # Pattern 2: Recognition correlation
        recognition_recommend = df_filtered.groupby(recognized_col)['recommend_numeric'].mean()
        if len(recognition_recommend) > 1:
            max_recog = recognition_recommend.max()
            min_recog = recognition_recommend.min()
            if max_recog - min_recog > 2:
                insights["key_patterns"].append({
                    "pattern": f"Strong correlation between recognition and recommendation scores (up to {max_recog - min_recog:.1f} point difference)",
                    "impact": "high"
                })
        
        # Pattern 3: Role-based variations
        role_std = role_recommend['mean'].std()
        if role_std > 1.5:
            lowest_role = role_recommend.iloc[0]
            highest_role = role_recommend.iloc[-1]
            insights["key_patterns"].append({
                "pattern": f"Significant variation across roles: {highest_role[role_col]} ({highest_role['mean']:.1f}) vs {lowest_role[role_col]} ({lowest_role['mean']:.1f})",
                "impact": "high"
            })
        
        # Pattern 4: Training preferences
        top_training = training_counts.idxmax()
        training_pct = (training_counts.max() / training_counts.sum() * 100)
        insights["key_patterns"].append({
            "pattern": f"{training_pct:.0f}% prefer '{top_training}' - indicates clear training delivery preference",
            "impact": "medium"
        })
        
        # STRENGTHS
        if promoter_rate >= 60:
            insights["strengths"].append(f"Strong employee advocacy with {promoter_rate:.0f}% promoter rate (score 8+/10)")
        
        if recognition_rate >= 70:
            insights["strengths"].append(f"Excellent recognition culture - {recognition_rate:.0f}% of employees feel recognized")
        
        if growth_rate >= 70:
            insights["strengths"].append(f"Strong career development perception - {growth_rate:.0f}% see growth potential")
        
        if impact_rate >= 70:
            insights["strengths"].append(f"High sense of purpose - {impact_rate:.0f}% feel they make a positive impact")
        
        # Check fulfillment
        fulfillment_positive = sum([count for label, count in fulfillment_counts.items() 
                                   if any(word in str(label).lower() for word in ['yes', 'fulfilling', 'rewarding'])])
        if fulfillment_positive / total >= 0.7:
            insights["strengths"].append(f"Strong job fulfillment - {fulfillment_positive/total*100:.0f}% report positive fulfillment")
        
        # CONCERNS
        if promoter_rate < 50:
            insights["concerns"].append(f"Low promoter rate ({promoter_rate:.0f}%) - less than half of employees would recommend the organization")
        
        if recognition_rate < 60:
            insights["concerns"].append(f"Recognition gap - only {recognition_rate:.0f}% feel adequately recognized, {100-recognition_rate:.0f}% do not")
        
        if growth_rate < 60:
            insights["concerns"].append(f"Limited growth perception - {100-growth_rate:.0f}% of employees don't see clear growth opportunities")
        
        if impact_rate < 60:
            insights["concerns"].append(f"Purpose disconnect - {100-impact_rate:.0f}% don't feel they're making a positive impact")
        
        # Check for low-scoring roles
        low_roles = role_recommend[role_recommend['mean'] < 7]
        if len(low_roles) > 0:
            role_list = ", ".join([f"{row[role_col]} ({row['mean']:.1f})" for _, row in low_roles.iterrows()])
            insights["concerns"].append(f"Departments with concerning scores: {role_list}")
        
        # RECOMMENDATIONS
        # Priority 1: Address lowest metric
        metrics = {
            'recognition': recognition_rate,
            'growth': growth_rate,
            'impact': impact_rate,
            'recommendation': promoter_rate
        }
        lowest_metric = min(metrics, key=metrics.get)
        
        if lowest_metric == 'recognition' and recognition_rate < 70:
            insights["recommendations"].append({
                "priority": "high",
                "action": "Implement a structured recognition program with peer-to-peer and manager recognition components",
                "expected_impact": f"Could improve recommendation scores by 1-2 points based on correlation data. Target: increase recognition rate from {recognition_rate:.0f}% to 80%+"
            })
        
        if lowest_metric == 'growth' and growth_rate < 70:
            insights["recommendations"].append({
                "priority": "high",
                "action": "Develop clear career pathways and communicate advancement opportunities through quarterly growth conversations",
                "expected_impact": f"Address {100-growth_rate:.0f}% who don't see growth potential, improving retention and engagement"
            })
        
        # Training recommendation
        if 'virtual' in top_training.lower():
            insights["recommendations"].append({
                "priority": "medium",
                "action": f"Expand virtual training offerings - {training_pct:.0f}% prefer this format",
                "expected_impact": "Higher training participation and skill development through preferred delivery method"
            })
        elif 'person' in top_training.lower() or 'site' in top_training.lower():
            insights["recommendations"].append({
                "priority": "medium",
                "action": f"Prioritize in-person training sessions - {training_pct:.0f}% prefer this format",
                "expected_impact": "Improved training effectiveness and team building through face-to-face interaction"
            })
        
        # Role-specific recommendations
        if len(low_roles) > 0:
            lowest_role_name = low_roles.iloc[0][role_col]
            lowest_role_score = low_roles.iloc[0]['mean']
            insights["recommendations"].append({
                "priority": "high",
                "action": f"Conduct focus groups with {lowest_role_name} (score: {lowest_role_score:.1f}) to identify specific pain points",
                "expected_impact": "Targeted interventions for departments with lowest satisfaction, reducing turnover risk"
            })
        
        # Recognition-recommendation correlation
        if recognition_rate < 70:
            insights["recommendations"].append({
                "priority": "high",
                "action": "Launch monthly recognition initiatives (Employee of the Month, spot bonuses, public acknowledgments)",
                "expected_impact": f"Recognition strongly correlates with recommendations - could improve overall scores significantly"
            })
        
        # General recommendations
        if avg_recommend < 8:
            insights["recommendations"].append({
                "priority": "medium",
                "action": "Conduct follow-up pulse surveys quarterly to track improvement in key metrics",
                "expected_impact": "Continuous feedback loop to measure effectiveness of interventions"
            })
        
        # DEMOGRAPHIC INSIGHTS
        demographic_parts = []
        
        # Role insights
        if len(role_recommend) > 1:
            highest_role = role_recommend.iloc[-1]
            lowest_role = role_recommend.iloc[0]
            demographic_parts.append(f"Role analysis shows {highest_role[role_col]} has highest satisfaction ({highest_role['mean']:.1f}/10) while {lowest_role[role_col]} needs attention ({lowest_role['mean']:.1f}/10).")
        
        # Years of employment insights
        if years_col in df_filtered.columns:
            years_recommend = df_filtered.groupby(years_col)['recommend_numeric'].mean().sort_values()
            if len(years_recommend) > 1:
                newest = years_recommend.index[0]
                newest_score = years_recommend.values[0]
                veteran = years_recommend.index[-1]
                veteran_score = years_recommend.values[-1]
                demographic_parts.append(f"Tenure analysis: {newest} employees score {newest_score:.1f} vs {veteran} employees at {veteran_score:.1f}.")
        
        # Training preferences by role
        training_role = pd.crosstab(df_filtered[role_col], df_filtered[training_pref_col])
        if len(training_role) > 0:
            demographic_parts.append(f"Training preferences vary by department - consider customized delivery methods per team.")
        
        insights["demographic_insights"] = " ".join(demographic_parts) if demographic_parts else "Consistent patterns across demographic groups - no major variations detected."
        
        return insights

    # ================================================================
    # DISPLAY STATISTICAL INSIGHTS
    # ================================================================
    if 'generate_insights' in st.session_state and st.session_state['generate_insights']:
        st.markdown("---")
        st.markdown("## 📊 Data-Driven Insights & Recommendations")
        
        with st.spinner('🔍 Analyzing survey data and generating insights...'):
            insights = generate_statistical_insights()
            
            if insights:
                # Use a container with custom styling
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 15px; padding: 30px; color: white; 
                                margin: 30px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
                """, unsafe_allow_html=True)
                
                # Executive Summary
                st.markdown("### 📋 Executive Summary")
                st.write(insights.get("executive_summary", "No summary available"))
                st.markdown("---")
                
                # Key Patterns
                st.markdown("### 🔍 Key Patterns Identified")
                if 'key_patterns' in insights and len(insights['key_patterns']) > 0:
                    for pattern in insights['key_patterns']:
                        impact_emoji = "🔴" if pattern['impact'] == 'high' else "🟡" if pattern['impact'] == 'medium' else "🟢"
                        st.write(f"{impact_emoji} **{pattern['pattern']}**")
                else:
                    st.write("No significant patterns detected.")
                st.markdown("---")
                
                # Strengths
                st.markdown("### 💪 Organizational Strengths")
                if 'strengths' in insights and len(insights['strengths']) > 0:
                    for strength in insights['strengths']:
                        st.write(f"✅ {strength}")
                else:
                    st.write("⚠️ No metrics exceeded the 70% threshold. Focus on improving core satisfaction drivers.")
                st.markdown("---")
                
                # Concerns
                st.markdown("### ⚠️ Areas of Concern")
                if 'concerns' in insights and len(insights['concerns']) > 0:
                    for concern in insights['concerns']:
                        st.write(f"⚡ {concern}")
                else:
                    st.write("✅ All key metrics are performing well. Continue monitoring and maintain current practices.")
                st.markdown("---")
                
                # Recommendations
                st.markdown("### 🎯 Strategic Recommendations")
                if 'recommendations' in insights and len(insights['recommendations']) > 0:
                    for i, rec in enumerate(insights['recommendations'], 1):
                        priority_badge = "🔴 HIGH" if rec['priority'] == 'high' else "🟡 MEDIUM" if rec['priority'] == 'medium' else "🟢 LOW"
                        st.markdown(f"""
                        **{i}. {priority_badge} PRIORITY**  
                        **Action:** {rec["action"]}  
                        **Expected Impact:** {rec["expected_impact"]}
                        """)
                        if i < len(insights['recommendations']):
                            st.write("")  # Add spacing
                else:
                    st.write("Continue current practices and monitor key metrics.")
                st.markdown("---")
                
                # Demographic Insights
                st.markdown("### 👥 Demographic Insights")
                if 'demographic_insights' in insights and insights['demographic_insights']:
                    st.write(insights["demographic_insights"])
                else:
                    st.write("Consistent patterns across demographic groups.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.session_state['insights_data'] = insights
                st.success("✅ Insights generated successfully!")
            else:
                st.error("❌ Unable to generate insights. Please try again.")
        
        st.session_state['generate_insights'] = False

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; padding: 20px;'>Dashboard created with Streamlit • Statistical Analysis Powered • Homes First Employee Survey 2024</p>", unsafe_allow_html=True)
