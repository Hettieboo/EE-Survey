#Employee Survey Dashboard

A comprehensive, interactive dashboard built with Streamlit to analyze and visualize employee survey data from Homes First.


## Overview

This dashboard provides real-time insights into employee satisfaction, engagement, and organizational culture at Homes First. It features interactive filters, dynamic visualizations, and key performance indicators to help leadership make data-driven decisions.

## Features

### Key Performance Indicators (KPIs)
- **Recommendation Rate**: Employee likelihood to recommend the organization (NPS-style)
- **Recognition Levels**: Percentage of employees feeling acknowledged
- **Growth Potential**: Employees who see career development opportunities
- **Positive Impact**: Staff who feel they make a meaningful difference

### Visual Analytics
1. **Job Fulfillment Analysis** - Dual-view donut and bar charts
2. **Training Preferences** - Lollipop chart comparing virtual vs in-person training
3. **Recommendation by Role** - Gradient-coded horizontal bars with threshold indicators
4. **Fulfillment by Tenure** - Stacked/grouped bars showing satisfaction over time
5. **Growth Perception by Age** - Stacked percentage bars across age groups
6. **Recognition vs Recommendation** - Correlation analysis with dual charts
7. **Application Methods** - Distribution of hiring channels with insight cards
8. **Disability Analysis** - Demographic breakdowns by age and gender

###  Interactive Filters
- **Role/Department**: Filter by specific teams or positions
- **Age Group**: Segment data by age demographics
- **Gender**: Analyze responses by gender identity

##  Getting Started

### Prerequisites

```bash
Python 3.8 or higher
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/homes-first-dashboard.git
cd homes-first-dashboard
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Ensure your data file is in place**
   - Place `EE SurveyAnalysis.xlsx` in the same directory as the script
   - The file should contain the survey response data

### Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
openpyxl>=3.1.0
```

Create a `requirements.txt` file with the above dependencies.

## 📁 Project Structure

```
homes-first-dashboard/
│
├── dashboard.py                 # Main Streamlit application
├── EE SurveyAnalysis.xlsx      # Survey data (Excel format)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 🎨 Dashboard Sections

### 1. Header & Overview
- Dynamic title with company branding
- Success message confirming data load
- Total respondent count in sidebar

### 2. Filters Panel (Sidebar)
- Multi-select dropdowns for Role, Age, and Gender
- Real-time filtering of all visualizations
- Dataset information display

### 3. KPI Metrics
Four gradient-styled metric cards displaying:
- Recommendation score and rate
- Employee recognition percentage
- Growth opportunity perception
- Positive impact sentiment

### 4. Detailed Analytics
Nine comprehensive visualization sections with:
- Modern chart types (donut, lollipop, gradient bars)
- Color-coded insights
- Statistical annotations
- Cross-analysis comparisons

## 🔧 Customization

### Updating Data
Simply replace `EE SurveyAnalysis.xlsx` with your new survey data. The dashboard automatically detects column names containing keywords like:
- "role" or "department"
- "age"
- "gender"
- "recommend"
- "years" and "employed"
- etc.

### Modifying Colors
Edit the color palettes in the script:
```python
DISTINCT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', ...]
```

### Adding New Visualizations
Follow the existing chart pattern:
```python
# ================================================================
# CHART X: YOUR CHART TITLE
# ================================================================
st.markdown("### 📊 Your Chart Title")
# Your matplotlib/seaborn code here
```

## 📊 Data Requirements

Your Excel file should contain columns for:
- Employee role/department
- Age group
- Gender identity
- Recommendation score (0-10)
- Years of employment
- Recognition/acknowledgment responses
- Growth potential perceptions
- Impact sentiment
- Training preferences
- Job fulfillment ratings
- Disability identification
- School enrollment status
- Application method

## 🚢 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py"]
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Author

- **Henrietta Atsenokhai** - *Initial work* -(https://github.com/Hettieboo)
 

## Contact

For questions or support, please contact:
- Email: henrietta.atsenokhai@gmail.com

## Future Enhancements

- [ ] Export functionality for filtered data
- [ ] PDF report generation
- [ ] Trend analysis across multiple survey periods
- [ ] Predictive analytics for employee retention
- [ ] Mobile-responsive design improvements
- [ ] Multi-language support
- [ ] Email alert system for critical metrics

---

**Built with ❤️ using Streamlit**
