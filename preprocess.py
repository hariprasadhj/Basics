import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove all special characters (leave only alphabets and spaces)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Get a set of English stopwords from NLTK
    stopwords_set = set(stopwords.words("english"))

    # Tokenization
    tokens = word_tokenize(text)

    # Apply Lemmatization to each token and filter out stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]
    print(tokens)
    # Rejoin tokens into a single string
    clean_text = ' '.join(tokens)
    return clean_text

# Load dataset from CSV
file_path = '/Users/hari/Documents/Visual Studio/Basics-1/covidtweet.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path,encoding='cp1252')

print(df.describe())

# Display first few rows before preprocessing
print("Before Preprocessing:")
print(df.head())

# Apply preprocessing to the tweet_text column
df['tweet_text'] = df['tweet_text'].astype(str)
df['cleaned_tweet_text'] = df['tweet_text'].apply(preprocess_text)

# Display first few rows after preprocessing
print("\nAfter Preprocessing:")
print(df[['cleaned_tweet_text']].head())

# Save the cleaned data to a new CSV file
output_path = 'cleaned_tweets.csv'  # Replace with your desired output path
df.to_csv(output_path, index=False)
print(f"\nCleaned data saved to {output_path}")