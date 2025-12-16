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
import json
import requests

# ================================================================
# CONFIGURATION & STYLING
# ================================================================

st.set_page_config(page_title="Homes First Survey Dashboard", layout="wide")

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
    st.markdown("<h1>🏠 Homes First Employee Survey Dashboard</h1>", unsafe_allow_html=True)

with col_buttons:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🤖 Generate AI Insights", use_container_width=True, type="primary"):
        st.session_state['generate_insights'] = True
    if st.button("📥 Download PDF", use_container_width=True):
        st.session_state['generate_pdf'] = True

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
    # AI INSIGHTS GENERATION FUNCTION
    # ================================================================
    def generate_ai_insights():
        """Generate AI-powered insights from the survey data"""
        
        # Prepare data summary for AI
        data_summary = {
            "total_respondents": int(total),
            "recommendation_metrics": {
                "average_score": float(avg_recommend),
                "promoters": int(positive_rec),
                "promoter_percentage": float(positive_rec/total*100)
            },
            "recognition_metrics": {
                "feel_recognized": int(recognized),
                "recognition_percentage": float(recognized/total*100)
            },
            "growth_metrics": {
                "see_growth_potential": int(growth),
                "growth_percentage": float(growth/total*100)
            },
            "impact_metrics": {
                "feel_positive_impact": int(impact),
                "impact_percentage": float(impact/total*100)
            },
            "fulfillment_distribution": fulfillment_counts.to_dict(),
            "training_preferences": training_counts.to_dict(),
            "role_recommendations": {
                str(row[role_col]): {
                    "avg_score": float(row['mean']),
                    "count": int(row['count'])
                }
                for _, row in role_recommend.iterrows()
            }
        }
        
        prompt = f"""You are an expert HR data analyst. Analyze this employee survey data and provide actionable insights.

Survey Data Summary:
{json.dumps(data_summary, indent=2)}

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "executive_summary": "2-3 sentence overview of key findings",
    "key_patterns": [
        {{"pattern": "description", "impact": "high/medium/low"}}
    ],
    "strengths": [
        "identified strength with supporting data"
    ],
    "concerns": [
        "identified concern with supporting data"
    ],
    "recommendations": [
        {{"priority": "high/medium/low", "action": "specific recommendation", "expected_impact": "description"}}
    ],
    "demographic_insights": "Any notable patterns by role, age, etc."
}}

Focus on:
1. Correlation between recognition and recommendation scores
2. Training preference implications
3. Department-specific patterns
4. Growth perception trends
5. Actionable recommendations for leadership

Respond ONLY with valid JSON, no markdown or additional text."""

        try:
            # Get API key from Streamlit secrets or environment variable
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
            
            if not api_key:
                st.warning("⚠️ Anthropic API key not found. Please add it to your Streamlit secrets.")
                st.info("Add `ANTHROPIC_API_KEY = 'your-key-here'` to `.streamlit/secrets.toml`")
                return None
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            if 'content' in data and len(data['content']) > 0:
                text_content = data['content'][0].get('text', '')
                # Remove markdown code blocks if present
                text_content = text_content.replace('```json', '').replace('```', '').strip()
                return json.loads(text_content)
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"JSON Parse Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return None

    # ================================================================
    # DISPLAY AI INSIGHTS
    # ================================================================
    if 'generate_insights' in st.session_state and st.session_state['generate_insights']:
        st.markdown("---")
        st.markdown("## 🤖 AI-Generated Insights & Recommendations")
        
        with st.spinner('🔍 Analyzing data and generating insights... This may take a moment.'):
            insights = generate_ai_insights()
            
            if insights:
                st.markdown('<div class="insights-box">', unsafe_allow_html=True)
                
                # Executive Summary
                st.markdown(f"""
                <div class="insight-section">
                    <div class="insight-title">📋 Executive Summary</div>
                    <div class="insight-text">{insights.get('executive_summary', 'No summary available')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Key Patterns
                if 'key_patterns' in insights:
                    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">🔍 Key Patterns Identified</div>', unsafe_allow_html=True)
                    for pattern in insights['key_patterns']:
                        impact_emoji = "🔴" if pattern['impact'] == 'high' else "🟡" if pattern['impact'] == 'medium' else "🟢"
                        st.markdown(f"<div class='insight-text'>{impact_emoji} <strong>{pattern['pattern']}</strong></div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Strengths
                if 'strengths' in insights:
                    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">💪 Organizational Strengths</div>', unsafe_allow_html=True)
                    for strength in insights['strengths']:
                        st.markdown(f"<div class='insight-text'>✅ {strength}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Concerns
                if 'concerns' in insights:
                    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">⚠️ Areas of Concern</div>', unsafe_allow_html=True)
                    for concern in insights['concerns']:
                        st.markdown(f"<div class='insight-text'>⚡ {concern}</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendations
                if 'recommendations' in insights:
                    st.markdown('<div class="insight-section">', unsafe_allow_html=True)
                    st.markdown('<div class="insight-title">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)
                    for rec in insights['recommendations']:
                        priority_badge = "🔴 HIGH" if rec['priority'] == 'high' else "🟡 MEDIUM" if rec['priority'] == 'medium' else "🟢 LOW"
                        st.markdown(f"""
                        <div class='recommendation'>
                            <strong>{priority_badge} PRIORITY</strong><br>
                            <strong>Action:</strong> {rec['action']}<br>
                            <strong>Expected Impact:</strong> {rec['expected_impact']}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Demographic Insights
                if 'demographic_insights' in insights:
                    st.markdown(f"""
                    <div class="insight-section">
                        <div class="insight-title">👥 Demographic Insights</div>
                        <div class="insight-text">{insights['demographic_insights']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.session_state['insights_data'] = insights
                st.success("✅ Insights generated successfully!")
            else:
                st.error("❌ Unable to generate insights. Please try again.")
        
        st.session_state['generate_insights'] = False

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d; padding: 20px;'>Dashboard created with Streamlit • Powered by Claude AI • Homes First Employee Survey 2024</p>", unsafe_allow_html=True)
