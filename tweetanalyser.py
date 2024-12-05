import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

file_path = '/Users/hari/Documents/Visual Studio/Basics-1/cleaned_tweets.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)
X = df['cleaned_tweet_text']
y = df['label']

print("Class Distribution:\n", y.value_counts())

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data, and transform the testing data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ConfusionMatrixDisplay.from_estimator(nb_model, X_test_vec, y_test, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()