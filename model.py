# resume_classifier.py

import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load CSV
try:
    df = pd.read_csv(r'C:\Users\saadk\Desktop\Project\UpdatedResumeDataSet.csv')
    print(" File loaded successfully!")
    print(" Rows loaded:", len(df))
    print(" Columns:", list(df.columns))
except Exception as e:
    print(" Failed to load CSV:", e)
    exit()

# Step 2: Check columns
if 'Resume' not in df.columns or 'Category' not in df.columns:
    print(" CSV must contain 'Resume' and 'Category' columns.")
    exit()

# Step 3: Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['Cleaned_Text'] = df['Resume'].apply(clean_text)

# Step 4: Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Category']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Save model and vectorizer
try:
    joblib.dump(model, 'resume_classifier_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print(" Model and vectorizer saved successfully!")
except Exception as e:
    print(" Failed to save model/vectorizer:", e)
