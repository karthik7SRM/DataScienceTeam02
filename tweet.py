import pandas as pd

# Load the CSV
df = pd.read_csv("tweets_cleaned_tokenized.csv")

# Convert text to lowercase for consistent matching
df['Text_cleaned'] = df['Text_cleaned'].str.lower()

# Define category keywords
category_keywords = {
    'sports': ['football', 'cricket', 'hockey', 'tennis', 'olympics', 'goal', 'match', 'score', 'tournament', 'messi'],
    'cinema': ['movie', 'film', 'cinema', 'trailer', 'actor', 'actress', 'shahrukh', 'box office', 'marvel', 'avengers'],
    'politics': ['president', 'prime minister', 'election', 'policy', 'reform', 'bjp', 'congress', 'modi'],
    'technology': ['iphone', 'android', 'ai', 'chatgpt', 'gpt', 'technology', 'tech', 'app', 'nasa'],
    'education': ['education', 'exam', 'university', 'school', 'college', 'syllabus'],
    'science': ['experiment', 'mars', 'satellite', 'research', 'lab', 'scientist'],
    'finance': ['budget', 'stock', 'share', 'market', 'finance', 'investment', 'bank'],
    'environment': ['climate', 'global warming', 'pollution', 'recycle', 'green', 'eco'],
    'health': ['vaccine', 'health', 'doctor', 'covid', 'hospital', 'cancer', 'treatment']
}

# Categorization function using text
def categorize_text(text):
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'others'

# Apply categorization to 'Text' column
df['Category'] = df['Text_cleaned'].apply(categorize_text)

# Save or preview result
df.to_csv("tweets_categorized_from_text1.csv", index=False)
print(df[['Text_cleaned', 'Category']].head(10))
