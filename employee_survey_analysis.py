"""
Employee Survey Analysis Dashboard - Streamlit App
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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Survey Analysis Dashboard",
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
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
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
    'good', 'love', 'enjoy', 'happy', 'appreciated', 'wonderful', 'rewarding', 'fantastic'
]

NEGATIVE_WORDS = [
    'challenging', 'difficult', 'poor', 'lack', 'never', 'inadequate', 'frustrating',
    'stress', 'overwhelmed', 'unhappy', 'concerned', 'disappointed', 'insufficient'
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
    recommend_col = next((c for c in cols if 'recommend' in c.lower()), None)
    
    # Find text columns (responses with longer average length)
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
        'recommend': recommend_col,
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
            # Clean and split text
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


def create_bar_chart(data, x_col, y_col, title, figsize=(10, 6), horizontal=False, color='steelblue'):
    """Create a matplotlib bar chart"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        bars = ax.barh(data[y_col], data[x_col], color=color)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
    else:
        bars = ax.bar(data[x_col], data[y_col], color=color)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    if horizontal:
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}', ha='left', va='center', fontsize=9)
    else:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_pie_chart(data, values_col, names_col, title, figsize=(8, 8)):
    """Create a matplotlib pie chart"""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("Set3", len(data))
    wedges, texts, autotexts = ax.pie(
        data[values_col], 
        labels=data[names_col],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    return fig


def generate_pdf_report(df, cols, positive_count, negative_count):
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
        ['Text Response Columns', str(len(cols['text']) if cols['text'] else 0)]
    ]
    
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
    
    # Sentiment Analysis
    if positive_count > 0 or negative_count > 0:
        story.append(Paragraph("Sentiment Analysis", heading_style))
        
        sentiment_ratio = positive_count / (positive_count + negative_count) * 100 if (positive_count + negative_count) > 0 else 0
        sentiment_text = f"The survey shows a {sentiment_ratio:.1f}% positive sentiment ratio, with {positive_count} positive indicators and {negative_count} areas of concern identified across all responses."
        story.append(Paragraph(sentiment_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Demographics
    if cols['role']:
        story.append(PageBreak())
        story.append(Paragraph("Demographics: Role Distribution", heading_style))
        
        role_data = df[cols['role']].value_counts().reset_index()
        role_data.columns = ['Role', 'Count']
        role_data['Percentage'] = (role_data['Count'] / role_data['Count'].sum() * 100).round(1)
        
        # Convert to table
        role_table_data = [['Role', 'Count', 'Percentage']]
        for _, row in role_data.iterrows():
            role_table_data.append([str(row['Role']), str(row['Count']), f"{row['Percentage']}%"])
        
        role_table = Table(role_table_data, colWidths=[3*inch, 1*inch, 1*inch])
        role_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
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
        
        story.append(role_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Theme Analysis
    if cols['text']:
        story.append(PageBreak())
        story.append(Paragraph("Theme Analysis", heading_style))
        
        theme_counts = analyze_themes(df, cols['text'])
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions'])
        theme_df = theme_df.sort_values('Mentions', ascending=False)
        
        # Convert to table
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
        story.append(Spacer(1, 0.2*inch))
        
        # Word Frequency
        story.append(PageBreak())
        story.append(Paragraph("Most Frequent Words", heading_style))
        
        word_freq = get_word_frequency(df, cols['text'], top_n=15)
        if word_freq:
            word_table_data = [['Word', 'Frequency']]
            for word, freq in word_freq:
                word_table_data.append([word, str(freq)])
            
            word_table = Table(word_table_data, colWidths=[3*inch, 2*inch])
            word_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ec4899')),
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
            
            story.append(word_table)
    
    # Recommendations section
    if cols['recommend']:
        story.append(PageBreak())
        story.append(Paragraph("Recommendation Likelihood", heading_style))
        
        rec_data = df[cols['recommend']].value_counts().reset_index()
        rec_data.columns = ['Response', 'Count']
        rec_data['Percentage'] = (rec_data['Count'] / rec_data['Count'].sum() * 100).round(1)
        
        rec_table_data = [['Response', 'Count', 'Percentage']]
        for _, row in rec_data.iterrows():
            rec_table_data.append([str(row['Response']), str(row['Count']), f"{row['Percentage']}%"])
        
        rec_table = Table(rec_table_data, colWidths=[3.5*inch, 1*inch, 1*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
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
        
        story.append(rec_table)
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    footer = Paragraph("<i>Generated by Employee Survey Analysis Tool • Built with Streamlit</i>", footer_style)
    story.append(footer)
    
    # Build PDF
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
    # Welcome screen
    st.info("👆 Please upload an Excel file to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📋 Overview
        - Total responses
        - Sentiment analysis
        - Recommendation scores
        - Key metrics at a glance
        """)
    
    with col2:
        st.markdown("""
        ### 👥 Demographics
        - Role distribution
        - Age breakdown
        - Department analysis
        - Visual charts
        """)
    
    with col3:
        st.markdown("""
        ### 💬 Text Analysis
        - Word frequency
        - Theme detection
        - Sentiment tracking
        - Searchable data
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    1. **Upload** your Excel file using the button above
    2. **Explore** different tabs for various insights
    3. **Export** findings as needed
    4. **Share** results with your team
    
    ### 📦 Requirements
    ```bash
    pip install streamlit pandas openpyxl matplotlib seaborn reportlab
    ```
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

# ================================================================
# SUMMARY METRICS
# ================================================================
st.markdown("### 📈 Key Metrics")

# Add PDF download button in header
col_header1, col_header2 = st.columns([3, 1])
with col_header2:
    if st.button("📄 Download Full Report as PDF", type="primary", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_buffer = generate_pdf_report(df, cols, positive_count, negative_count)
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

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Total Responses", len(df))

with metric_col2:
    st.metric("Survey Questions", len(df.columns))

with metric_col3:
    st.metric("Positive Mentions", positive_count)

with metric_col4:
    st.metric("Areas of Concern", negative_count)

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "👥 Demographics", 
    "💬 Text Analysis",
    "🎯 Themes",
    "📄 Raw Data"
])

# ================================================================
# TAB 1: OVERVIEW
# ================================================================
with tab1:
    st.header("Survey Overview")
    
    # Recommendation analysis
    if cols['recommend']:
        st.subheader("Likelihood to Recommend")
        
        rec_data = df[cols['recommend']].value_counts().reset_index()
        rec_data.columns = ['Response', 'Count']
        
        fig = create_bar_chart(rec_data, 'Response', 'Count', 
                              'How likely are you to recommend?',
                              figsize=(12, 6), color='#3b82f6')
        st.pyplot(fig)
        plt.close()
        
        total_responses = rec_data['Count'].sum()
        st.info(f"**Total Responses:** {total_responses}")
    
    # Sentiment overview
    if cols['text']:
        st.subheader("Sentiment Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**✅ Positive Indicators:** {positive_count}")
        with col2:
            st.warning(f"**⚠️ Negative Indicators:** {negative_count}")
        
        # Sentiment chart
        sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [positive_count, negative_count]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#10b981', '#ef4444']
        bars = ax.bar(sentiment_df['Sentiment'], sentiment_df['Count'], color=colors)
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ================================================================
# TAB 2: DEMOGRAPHICS
# ================================================================
with tab2:
    st.header("Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if cols['role']:
            st.subheader("Respondents by Role")
            
            role_data = df[cols['role']].value_counts().reset_index()
            role_data.columns = ['Role', 'Count']
            
            fig = create_bar_chart(role_data, 'Count', 'Role',
                                  'Distribution by Role/Department',
                                  figsize=(10, 8), horizontal=True, color='#8b5cf6')
            st.pyplot(fig)
            plt.close()
            
            # Show table
            with st.expander("📊 View detailed breakdown"):
                role_data['Percentage'] = (role_data['Count'] / role_data['Count'].sum() * 100).round(1)
                st.dataframe(role_data, use_container_width=True)
        else:
            st.info("No role/department column found")
    
    with col2:
        if cols['age']:
            st.subheader("Age Distribution")
            
            age_data = df[cols['age']].value_counts().reset_index()
            age_data.columns = ['Age Group', 'Count']
            
            fig = create_pie_chart(age_data, 'Count', 'Age Group', 'Age Distribution')
            st.pyplot(fig)
            plt.close()
            
            # Show table
            with st.expander("📊 View detailed breakdown"):
                age_data['Percentage'] = (age_data['Count'] / age_data['Count'].sum() * 100).round(1)
                st.dataframe(age_data, use_container_width=True)
        else:
            st.info("No age column found")

# ================================================================
# TAB 3: TEXT ANALYSIS
# ================================================================
with tab3:
    st.header("Text Response Analysis")
    
    if cols['text']:
        # Word frequency
        st.subheader("Most Frequent Words")
        
        word_freq = get_word_frequency(df, cols['text'], top_n=25)
        if word_freq:
            word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig = create_bar_chart(word_df, 'Frequency', 'Word',
                                  'Top 25 Most Frequent Words',
                                  figsize=(10, 12), horizontal=True, color='#ec4899')
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
        else:
            st.info("No words found to analyze")
        
        # Text columns list
        st.subheader("Text Response Columns Analyzed")
        for i, col in enumerate(cols['text'], 1):
            st.write(f"**{i}.** {col}")
            
            # Show sample responses
            with st.expander(f"View sample responses from: {col[:50]}..."):
                samples = df[col].dropna().head(5)
                for idx, sample in enumerate(samples, 1):
                    st.write(f"**Response {idx}:** {sample}")
    else:
        st.info("No text response columns detected in your data")

# ================================================================
# TAB 4: THEMES
# ================================================================
with tab4:
    st.header("Theme Analysis")
    
    if cols['text']:
        theme_counts = analyze_themes(df, cols['text'])
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions'])
        theme_df = theme_df.sort_values('Mentions', ascending=False)
        
        # Main chart
        fig = create_bar_chart(theme_df, 'Mentions', 'Theme',
                              'Key Themes in Survey Responses',
                              figsize=(10, 8), horizontal=True, color='#10b981')
        st.pyplot(fig)
        plt.close()
        
        # Theme details
        st.subheader("Theme Breakdown")
        
        col1, col2 = st.columns(2)
        
        for idx, (theme, keywords) in enumerate(THEMES.items()):
            with (col1 if idx % 2 == 0 else col2):
                mentions = theme_counts[theme]
                with st.expander(f"🔍 {theme} ({mentions} mentions)"):
                    st.write(f"**Keywords tracked:** {', '.join(keywords)}")
                    if mentions > 0:
                        st.success(f"Found in {mentions} responses")
                    else:
                        st.info("No mentions found")
        
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
# TAB 5: RAW DATA
# ================================================================
with tab5:
    st.header("Raw Survey Data")
    
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
    
    # Data info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Rows:** {df.shape[0]}")
    with col2:
        st.info(f"**Columns:** {df.shape[1]}")
    with col3:
        missing = df.isnull().sum().sum()
        st.info(f"**Missing Values:** {missing}")
    
    # Download options
    st.markdown("---")
    st.subheader("Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            "survey_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Generate PDF report
        try:
            pdf_buffer = generate_pdf_report(df, cols, positive_count, negative_count)
            st.download_button(
                "📄 Download PDF Report",
                pdf_buffer,
                f"survey_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                "application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            # Fallback to CSV
            summary = {
                'Total Responses': [len(df)],
                'Positive Mentions': [positive_count],
                'Negative Mentions': [negative_count],
                'Text Columns': [len(cols['text']) if cols['text'] else 0]
            }
            summary_df = pd.DataFrame(summary)
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📊 Download Summary CSV",
                summary_csv,
                "summary_report.csv",
                "text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p>Built with Streamlit • Employee Survey Analysis Tool</p>
    <p>Upload a new file to analyze different data</p>
</div>
""", unsafe_allow_html=True)
