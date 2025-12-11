"""
Employee Survey Analysis Dashboard - Streamlit App
Save this as: survey_app.py
Run with: streamlit run survey_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
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


def create_bar_chart(data, x, y, title, color=None, orientation='v'):
    """Create a plotly bar chart"""
    if orientation == 'h':
        fig = px.bar(data, x=x, y=y, orientation='h', title=title,
                    color=color, color_continuous_scale='Viridis')
    else:
        fig = px.bar(data, x=x, y=y, title=title,
                    color=color, color_continuous_scale='Blues')
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig


def create_pie_chart(data, values, names, title):
    """Create a plotly pie chart"""
    fig = px.pie(data, values=values, names=names, title=title)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig


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
                              color='Count')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        fig = px.bar(sentiment_df, x='Sentiment', y='Count',
                    color='Sentiment',
                    color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'},
                    title='Sentiment Distribution')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

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
                                  color='Count', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
            
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
            
            fig = px.bar(word_df, x='Frequency', y='Word', orientation='h',
                        title='Top 25 Most Frequent Words',
                        color='Frequency', color_continuous_scale='Plasma')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
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
        fig = px.bar(theme_df, x='Mentions', y='Theme', orientation='h',
                    title='Key Themes in Survey Responses',
                    color='Mentions', color_continuous_scale='RdYlGn')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
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
        if st.button("📥 Export Theme Analysis", type="primary"):
            csv = theme_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "theme_analysis.csv",
                "text/csv"
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
        # Create summary report
        summary = {
            'Total Responses': [len(df)],
            'Positive Mentions': [positive_count],
            'Negative Mentions': [negative_count],
            'Text Columns': [len(cols['text']) if cols['text'] else 0]
        }
        summary_df = pd.DataFrame(summary)
        summary_csv = summary_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📊 Download Summary Report",
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
