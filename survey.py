# ================================================================
# Employee Survey Analysis Script
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

def main():
    # 1️⃣ Load the dataset
    df = pd.read_excel('EE Survey-Analysis.xlsx')  # Replace with your file path

    # 2️⃣ Quick overview
    print("Dataset shape:", df.shape)
    print(df.head(3))

    # 3️⃣ Identify key columns
    role_col = [c for c in df.columns if "role" in c.lower() or "department" in c.lower()][0]
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    gender_col = [c for c in df.columns if "gender" in c.lower()][0]
    recommend_col = [c for c in df.columns if "recommend" in c.lower()][0]
    years_col = [c for c in df.columns if "years" in c.lower() and "employed" in c.lower()][0]

    # 4️⃣ Demographics Analysis
    plt.figure()
    sns.countplot(y=df[role_col], order=df[role_col].value_counts().index, palette='viridis')
    plt.title("Respondents by Role/Department")
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.countplot(x=df[age_col], order=df[age_col].value_counts().index, palette='magma')
    plt.title("Respondents by Age Group")
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.countplot(x=df[gender_col], order=df[gender_col].value_counts().index, palette='coolwarm')
    plt.title("Respondents by Gender")
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.countplot(x=df[recommend_col], order=df[recommend_col].value_counts().index, palette='plasma')
    plt.title("Likelihood to Recommend Homes First")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.countplot(x=df[years_col], order=df[years_col].value_counts().index, palette='cividis')
    plt.title("Years Employed at Homes First")
    plt.tight_layout()
    plt.show()

    # 5️⃣ Text Analysis - Comments
    text_cols = [c for c in df.columns if "comment" in c.lower()]
    STOP_WORDS = {'and','the','to','of','a','i','my','in','for','on','it','is','with','as','we','be'}

    # Combine all comments
    all_text = " ".join(df[col].dropna().astype(str).sum() for col in text_cols)

    # Top 20 words
    words = [w.lower().strip('.,!?;:()[]{}') for w in all_text.split() if w.lower() not in STOP_WORDS and len(w)>3]
    word_freq = Counter(words).most_common(20)
    word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    print("Top 20 Words in Comments:")
    print(word_df)

    # 6️⃣ Sentiment Analysis
    POSITIVE_WORDS = ['fulfilling', 'great', 'excellent', 'positive', 'amazing', 'helpful', 'supportive', 'good', 'love', 'enjoy', 'happy', 'appreciated', 'wonderful']
    NEGATIVE_WORDS = ['challenging', 'difficult', 'poor', 'lack', 'not', 'never', 'inadequate', 'frustrating', 'stress', 'overwhelmed', 'unhappy', 'concerned']

    positive_count = sum(sum(1 for w in POSITIVE_WORDS if w in str(text).lower()) for col in text_cols for text in df[col].dropna())
    negative_count = sum(sum(1 for w in NEGATIVE_WORDS if w in str(text).lower()) for col in text_cols for text in df[col].dropna())

    print(f"Positive mentions: {positive_count}")
    print(f"Negative mentions: {negative_count}")

    # 7️⃣ Themes Analysis
    THEMES = {
        "Workload": ["understaffed", "workload", "busy", "overwhelming", "paperwork", "time", "tasks"],
        "Support": ["support", "help", "assist", "guidance", "resources", "tools"],
        "Team": ["team", "colleagues", "coworkers", "collaboration", "together", "staff"],
        "Training": ["training", "development", "learning", "skills", "education", "growth"],
        "Clients": ["client", "resident", "people", "helping", "care", "community"],
        "Management": ["management", "leadership", "supervisor", "manager", "direction"],
        "Recognition": ["recognition", "appreciation", "valued", "acknowledged", "feedback"],
        "Work-Life Balance": ["balance", "flexibility", "schedule", "hours", "time off"]
    }

    theme_counts = {theme: 0 for theme in THEMES}
    for col in text_cols:
        for text in df[col].dropna().astype(str):
            t = text.lower()
            for theme, keywords in THEMES.items():
                if any(kw in t for kw in keywords):
                    theme_counts[theme] += 1

    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions']).sort_values('Mentions', ascending=False)
    plt.figure()
    sns.barplot(x='Mentions', y='Theme', data=theme_df, palette='RdYlGn')
    plt.title("Key Themes in Survey Comments")
    plt.tight_layout()
    plt.show()
    print(theme_df)

    # 8️⃣ Cross-analysis examples
    plt.figure()
    sns.countplot(x=df[recommend_col], hue=df[role_col], data=df, palette='Set2')
    plt.title("Recommendation by Role/Department")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.countplot(x=df[recommend_col], hue=df[age_col], data=df, palette='Set3')
    plt.title("Recommendation by Age Group")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
