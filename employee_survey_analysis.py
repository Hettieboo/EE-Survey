"""
Employee Survey Analysis Dashboard - Enhanced Streamlit App
Save this as: survey_app.py
Run with: streamlit run survey_app.py

Requirements: streamlit pandas openpyxl matplotlib seaborn reportlab
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Employee Survey Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
STOP_WORDS = {
    'and', 'the', 'to', 'of', 'a', 'i', 'my', 'in', 'for', 'on', 'it', 'is', 'with',
    'as', 'we', 'be', 'that', 'at', 'are', 'have', 'not', 'or', 'from', 'this', 'but',
    'they', 'you', 'all', 'can', 'more', 'when', 'there', 'so', 'up', 'out', 'if', 'about',
    'been', 'will', 'would', 'could', 'should', 'has', 'had', 'do', 'does', 'did', 'their'
}

THEMES = {
    "Workload": ["understaffed", "workload", "busy", "overwhelming", "paperwork", "time", "tasks", "overworked"],
    "Support": ["support", "help", "assist", "guidance", "resources", "tools", "backup"],
    "Team": ["team", "colleagues", "coworkers", "collaboration", "together", "staff", "teamwork"],
    "Training": ["training", "development", "learning", "skills", "education", "growth", "professional"],
    "Clients": ["client", "resident", "people", "helping", "care", "community", "impact"],
    "Management": ["management", "leadership", "supervisor", "manager", "direction", "communication"],
    "Recognition": ["recognition", "appreciation", "valued", "acknowledged", "feedback", "praise"],
    "Work-Life Balance": ["balance", "flexibility", "schedule", "hours", "time off", "burnout"]
}

POSITIVE_WORDS = [
    'fulfilling', 'great', 'excellent', 'positive', 'amazing', 'helpful', 'supportive',
    'good', 'love', 'enjoy', 'happy', 'appreciated', 'wonderful', 'rewarding', 'fantastic',
    'best', 'strong', 'passionate', 'caring', 'dedicated'
]

NEGATIVE_WORDS = [
    'challenging', 'difficult', 'poor', 'lack', 'never', 'inadequate', 'frustrating',
    'stress', 'overwhelmed', 'unhappy', 'concerned', 'disappointed', 'insufficient',
    'understaffed', 'burnout', 'worried', 'struggle', 'problem'
]


@st.cache_data
def load_data(file):
    """Load Excel file and return dataframe"""
    try:
        df = pd.read_excel(file)
        return df, None
    except Exception as e:
        return None, str(e)


def find_columns(df):
    """Identify key columns in the dataset"""
    cols = df.columns.tolist()
    
    # Find specific column types
    role_col = next((c for c in cols if 'role' in c.lower() or 'department' in c.lower()), None)
    age_col = next((c for c in cols if 'age' in c.lower()), None)
    gender_col = next((c for c in cols if 'gender' in c.lower()), None)
    recommend_col = next((c for c in cols if 'recommend' in c.lower()), None)
    years_col = next((c for c in cols if 'years' in c.lower() and 'employ' in c.lower()), None)
    
    # Find text columns (comments/open-ended responses)
    text_cols = []
    for col in cols:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_length = non_null.astype(str).str.len().mean()
                if avg_length > 30:  # Threshold for text responses
                    text_cols.append(col)
    
    return {
        'role': role_col,
        'age': age_col,
        'gender': gender_col,
        'recommend': recommend_col,
        'years': years_col,
        'text': text_cols
    }


def analyze_themes(df, text_cols):
    """Analyze themes across text responses"""
    theme_counts = {theme: 0 for theme in THEMES}
    
    for col in text_cols:
        for text in df[col].dropna().astype(str):
            text_lower = text.lower()
            for theme, keywords in THEMES.items():
                if any(keyword in text_lower for keyword in keywords):
                    theme_counts[theme] += 1
    
    return theme_counts


def get_word_frequency(df, text_cols, top_n=25):
    """Get most frequent words from text columns"""
    all_words = []
    
    for col in text_cols:
        for text in df[col].dropna().astype(str):
            words = text.lower().replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
            words = [w.strip('.,!?;:()[]{}"\'-') for w in words]
            words = [w for w in words if len(w) > 3 and w not in STOP_WORDS and w.isalpha()]
            all_words.extend(words)
    
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)


def calculate_sentiment(df, text_cols):
    """Calculate basic sentiment indicators"""
    positive_count = 0
    negative_count = 0
    
    for col in text_cols:
        for text in df[col].dropna().astype(str):
            text_lower = text.lower()
            positive_count += sum(1 for word in POSITIVE_WORDS if word in text_lower)
            negative_count += sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    return positive_count, negative_count


def calculate_nps(df, recommend_col):
    """Calculate Net Promoter Score"""
    if recommend_col not in df.columns:
        return None, None, None, None
    
    responses = df[recommend_col].dropna()
    total = len(responses)
    
    if total == 0:
        return None, None, None, None
    
    # Count promoters (top 2 categories) and detractors (bottom 2 categories)
    value_counts = responses.value_counts()
    sorted_categories = sorted(value_counts.index)
    
    if len(sorted_categories) < 2:
        return None, None, None, None
    
    # Assume last 2 are promoters, first 2 are detractors
    promoters = value_counts[sorted_categories[-2:]].sum() if len(sorted_categories) >= 2 else 0
    detractors = value_counts[sorted_categories[:2]].sum() if len(sorted_categories) >= 2 else 0
    
    nps = ((promoters - detractors) / total) * 100
    promoter_pct = (promoters / total) * 100
    detractor_pct = (detractors / total) * 100
    
    return nps, promoter_pct, detractor_pct, total


def generate_pdf_report(df, cols, positive_count, negative_count, nps_score):
    """Generate a comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3b82f6'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    title = Paragraph("📊 Employee Survey Analysis Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_text = Paragraph(f"<i>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>", styles['Normal'])
    story.append(date_text)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Responses', str(len(df))],
        ['Survey Questions', str(len(df.columns))],
        ['Positive Mentions', str(positive_count)],
        ['Negative Mentions', str(negative_count)],
        ['Text Response Columns', str(len(cols['text']) if cols['text'] else 0)],
    ]
    
    if nps_score is not None:
        summary_data.append(['Net Promoter Score (NPS)', f"{nps_score:.1f}"])
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Theme Analysis
    if cols['text']:
        story.append(PageBreak())
        story.append(Paragraph("Theme Analysis", heading_style))
        
        theme_counts = analyze_themes(df, cols['text'])
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions'])
        theme_df = theme_df.sort_values('Mentions', ascending=False)
        
        theme_table_data = [['Theme', 'Mentions']]
        for _, row in theme_df.iterrows():
            theme_table_data.append([str(row['Theme']), str(row['Mentions'])])
        
        theme_table = Table(theme_table_data, colWidths=[4*inch, 1.5*inch])
        theme_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        story.append(theme_table)
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    footer = Paragraph("<i>Generated by Employee Survey Analysis Tool • Built with Streamlit</i>", footer_style)
    story.append(footer)
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ================================================================
# MAIN APP
# ================================================================

# Header
st.markdown('<p class="main-header">📊 Employee Survey Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your Excel survey file",
    type=['xlsx', 'xls'],
    help="Upload an Excel file containing survey responses"
)

if not uploaded_file:
    st.info("👆 Please upload an Excel file to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📋 Overview
        - Total responses & demographics
        - Net Promoter Score (NPS)
        - Sentiment analysis
        - Key insights
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Deep Analysis
        - Cross-tabulations
        - Theme detection
        - Word frequency
        - Engagement metrics
        """)
    
    with col3:
        st.markdown("""
        ### 📄 Export & Share
        - Download full PDF report
        - Export raw data
        - Save insights
        - Share findings
        """)
    
    st.stop()

# Load data
df, error = load_data(uploaded_file)

if error:
    st.error(f"❌ Error loading file: {error}")
    st.stop()

if df is None or len(df) == 0:
    st.error("❌ No data found in the uploaded file")
    st.stop()

# Find column types
cols = find_columns(df)

# Calculate metrics
positive_count = 0
negative_count = 0
if cols['text']:
    positive_count, negative_count = calculate_sentiment(df, cols['text'])

# Calculate NPS
nps_score, promoter_pct, detractor_pct, nps_total = calculate_nps(df, cols['recommend'])

# ================================================================
# SUMMARY METRICS WITH PDF DOWNLOAD
# ================================================================
st.markdown("### 📈 Key Metrics")

# Add PDF download button in header
col_header1, col_header2 = st.columns([3, 1])
with col_header2:
    if st.button("📄 Download Full Report as PDF", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_buffer = generate_pdf_report(df, cols, positive_count, negative_count, nps_score)
                st.download_button(
                    "⬇️ Click to Download PDF",
                    pdf_buffer,
                    f"survey_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                    use_container_width=True
                )
                st.success("✅ PDF generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

# Metrics row
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Total Responses", len(df))

with metric_col2:
    st.metric("Survey Questions", len(df.columns))

with metric_col3:
    st.metric("Positive Mentions", positive_count, delta=f"+{positive_count - negative_count}" if positive_count > negative_count else None)

with metric_col4:
    st.metric("Concern Areas", negative_count)

with metric_col5:
    if nps_score is not None:
        st.metric("Net Promoter Score", f"{nps_score:.1f}", delta="Good" if nps_score > 0 else "Needs Improvement")
    else:
        st.metric("NPS", "N/A")

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview",
    "👥 Demographics", 
    "🔍 Cross-Analysis",
    "💬 Text Analysis",
    "🎯 Themes",
    "📄 Raw Data"
])

# ================================================================
# TAB 1: OVERVIEW
# ================================================================
with tab1:
    st.header("Survey Overview & Key Insights")
    
    # Quick insights
    st.markdown("### 🎯 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**📊 Response Rate**")
        response_rate = len(df)
        st.write(f"Total responses collected: **{response_rate}**")
        
        if cols['role']:
            unique_roles = df[cols['role']].nunique()
            st.write(f"Representing **{unique_roles}** different roles/departments")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**😊 Sentiment Overview**")
        if positive_count > 0 or negative_count > 0:
            sentiment_ratio = (positive_count / (positive_count + negative_count)) * 100
            st.write(f"Positive sentiment: **{sentiment_ratio:.1f}%**")
            st.write(f"Positive indicators: {positive_count} | Concerns: {negative_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # NPS Analysis
    if nps_score is not None:
        st.markdown("### 🎖️ Net Promoter Score (NPS) Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NPS Score", f"{nps_score:.1f}", help="Score ranges from -100 to +100")
        with col2:
            st.metric("Promoters", f"{promoter_pct:.1f}%", help="Employees who would recommend")
        with col3:
            st.metric("Detractors", f"{detractor_pct:.1f}%", help="Employees with concerns")
        
        # NPS Interpretation
        if nps_score > 50:
            st.success("🌟 **Excellent!** Your NPS indicates high employee satisfaction and loyalty.")
        elif nps_score > 0:
            st.info("👍 **Good!** Positive sentiment overall, with room for improvement.")
        else:
            st.warning("⚠️ **Needs Attention!** Focus on addressing employee concerns.")
    
    # Recommendation breakdown
    if cols['recommend']:
        st.markdown("### 📊 Recommendation Likelihood")
        
        rec_data = df[cols['recommend']].value_counts().reset_index()
        rec_data.columns = ['Response', 'Count']
        rec_data['Percentage'] = (rec_data['Count'] / rec_data['Count'].sum() * 100).round(1)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(rec_data['Response'], rec_data['Count'], color='#3b82f6')
        ax.set_xlabel('Number of Responses', fontsize=11)
        ax.set_title('How likely are you to recommend as a good place to work?', fontsize=14, fontweight='bold')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f"{int(width)} ({rec_data['Percentage'].iloc[i]}%)",
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ================================================================
# TAB 2: DEMOGRAPHICS
# ================================================================
with tab2:
    st.header("Demographics Analysis")
    
    # Role/Department
    if cols['role']:
        st.subheader("📍 Respondents by Role/Department")
        role_data = df[cols['role']].value_counts().reset_index()
        role_data.columns = ['Role', 'Count']
        role_data['Percentage'] = (role_data['Count'] / role_data['Count'].sum() * 100).round(1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(y='Role', x='Count', data=role_data, palette='viridis', ax=ax)
        ax.set_title('Distribution by Role/Department', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count', fontsize=11)
        
        for i, v in enumerate(role_data['Count']):
            ax.text(v, i, f" {v} ({role_data['Percentage'].iloc[i]}%)", 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        with st.expander("📊 Detailed breakdown"):
            st.dataframe(role_data, use_container_width=True)
    
    # Age, Gender, Years in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        if cols['age']:
            st.subheader("👤 Age Distribution")
            age_data = df[cols['age']].value_counts().reset_index()
            age_data.columns = ['Age Group', 'Count']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_palette = sns.color_palette("magma", len(age_data))
            wedges, texts, autotexts = ax.pie(
                age_data['Count'],
                labels=age_data['Age Group'],
                autopct='%1.1f%%',
                colors=colors_palette,
                startangle=90
            )
            ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Gender distribution
        if cols['gender']:
            st.subheader("⚧ Gender Distribution")
            gender_data = df[cols['gender']].value_counts().reset_index()
            gender_data.columns = ['Gender', 'Count']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Gender', y='Count', data=gender_data, palette='coolwarm', ax=ax)
            ax.set_title('Respondents by Gender', fontsize=14, fontweight='bold')
            
            for i, v in enumerate(gender_data['Count']):
                ax.text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        # Years employed
        if cols['years']:
            st.subheader("📅 Years Employed")
            years_data = df[cols['years']].value_counts().reset_index()
            years_data.columns = ['Years', 'Count']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Years', y='Count', data=years_data, palette='cividis', ax=ax)
            ax.set_title('Years Employed Distribution', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            
            for i, v in enumerate(years_data['Count']):
                ax.text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ================================================================
# TAB 3: CROSS-ANALYSIS
# ================================================================
with tab3:
    st.header("Cross-Analysis: Recommendation by Demographics")
    
    if cols['recommend']:
        # Recommendation by Role
        if cols['role']:
            st.subheader("📊 Recommendation by Role/Department")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            rec_by_role = pd.crosstab(df[cols['role']], df[cols['recommend']])
            rec_by_role.plot(kind='bar', ax=ax, colormap='Set2')
            ax.set_title('Recommendation Likelihood by Role', fontsize=14, fontweight='bold')
            ax.set_xlabel('Role/Department', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Recommendation', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Recommendation by Age
        if cols['age']:
            st.subheader("👥 Recommendation by Age Group")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            rec_by_age = pd.crosstab(df[cols['age']], df[cols['recommend']])
            rec_by_age.plot(kind='bar', ax=ax, colormap='Set3')
            ax.set_title('Recommendation Likelihood by Age Group', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age Group', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Recommendation', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Recommendation by Years Employed
        if cols['years']:
            st.subheader("📅 Recommendation by Years Employed")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            rec_by_years = pd.crosstab(df[cols['years']], df[cols['recommend']])
            rec_by_years.plot(kind='bar', ax=ax, colormap='viridis')
            ax.set_title('Recommendation Likelihood by Tenure', fontsize=14, fontweight='bold')
            ax.set_xlabel('Years Employed', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Recommendation', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No recommendation data available for cross-analysis")

# ================================================================
# TAB 4: TEXT ANALYSIS
# ================================================================
with tab4:
    st.header("Text Response Analysis")
    
    if cols['text']:
        # Sentiment overview
        st.subheader("😊 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**✅ Positive Indicators:** {positive_count}")
        with col2:
            st.warning(f"**⚠️ Concern Areas:** {negative_count}")
        
        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [positive_count, negative_count]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_sent = ['#10b981', '#ef4444']
        bars = ax.bar(sentiment_df['Sentiment'], sentiment_df['Count'], color=colors_sent)
        ax.set_title('Sentiment Distribution in Comments', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Word frequency
        st.subheader("📝 Most Frequent Words")
        
        word_freq = get_word_frequency(df, cols['text'], top_n=25)
        if word_freq:
            word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(10, 12))
            sns.barplot(y='Word', x='Frequency', data=word_df, palette='plasma', ax=ax)
            ax.set_title('Top 25 Most Frequent Words in Comments', fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency', fontsize=11)
            
            for i, v in enumerate(word_df['Frequency']):
                ax.text(v, i, f' {v}', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Download option
            csv = word_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Word Frequency Data",
                csv,
                "word_frequency.csv",
                "text/csv"
            )
        
        # Text columns analyzed
        st.subheader("📋 Text Response Columns Analyzed")
        for i, col in enumerate(cols['text'], 1):
            st.write(f"**{i}.** {col}")
            
            with st.expander(f"View sample responses"):
                samples = df[col].dropna().head(5)
                for idx, sample in enumerate(samples, 1):
                    st.write(f"**Response {idx}:** {sample}")
    else:
        st.info("No text response columns detected in your data")

# ================================================================
# TAB 5: THEMES
# ================================================================
with tab5:
    st.header("Theme Analysis")
    
    if cols['text']:
        theme_counts = analyze_themes(df, cols['text'])
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions'])
        theme_df = theme_df.sort_values('Mentions', ascending=False)
        
        # Main chart
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(y='Theme', x='Mentions', data=theme_df, palette='RdYlGn', ax=ax)
        ax.set_title('Key Themes in Survey Responses', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Mentions', fontsize=11)
        
        for i, v in enumerate(theme_df['Mentions']):
            ax.text(v, i, f' {v}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Theme details
        st.subheader("🔍 Theme Breakdown")
        
        col1, col2 = st.columns(2)
        
        for idx, (theme, keywords) in enumerate(THEMES.items()):
            with (col1 if idx % 2 == 0 else col2):
                mentions = theme_counts[theme]
                with st.expander(f"📌 {theme} ({mentions} mentions)"):
                    st.write(f"**Keywords tracked:** {', '.join(keywords)}")
                    if mentions > 0:
                        st.success(f"✅ Found in {mentions} responses")
                    else:
                        st.info("ℹ️ No mentions found")
        
        # Export theme data
        st.markdown("---")
        csv = theme_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Theme Analysis",
            csv,
            "theme_analysis.csv",
            "text/csv",
            type="primary"
        )
    else:
        st.info("No text columns available for theme analysis")

# ================================================================
# TAB 6: RAW DATA
# ================================================================
with tab6:
    st.header("Raw Survey Data")
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    # Search functionality
    search_term = st.text_input("🔍 Search in data", "", placeholder="Enter search term...")
    
    if search_term:
        mask = df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = df[mask]
        
        st.success(f"Found {len(filtered_df)} matching rows")
        st.dataframe(filtered_df, use_container_width=True, height=400)
    else:
        st.dataframe(df, use_container_width=True, height=400)
    
    # Download options
    st.markdown("---")
    st.subheader("📥 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download Full Dataset (CSV)",
            csv,
            "survey_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Summary CSV
        summary = {
            'Metric': ['Total Responses', 'Positive Mentions', 'Negative Mentions', 'NPS Score'],
            'Value': [len(df), positive_count, negative_count, f"{nps_score:.1f}" if nps_score else "N/A"]
        }
        summary_df = pd.DataFrame(summary)
        summary_csv = summary_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📊 Download Summary Metrics (CSV)",
            summary_csv,
            "summary_metrics.csv",
            "text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p><strong>Employee Survey Analysis Tool</strong> • Built with Streamlit</p>
    <p>📊 Comprehensive insights • 📄 PDF Reports • 📈 Data Visualization</p>
</div>
""", unsafe_allow_html=True)
