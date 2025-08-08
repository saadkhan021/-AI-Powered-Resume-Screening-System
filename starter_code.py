# resume_classifier.py (updated for your CSV)

import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load your labeled dataset
try:
    df = pd.read_csv(r'C:\Users\saadk\Desktop\Project\UpdatedResumeDataSet.csv')
except FileNotFoundError:
    print(" Error: File 'labeled_resumes.csv' not found. Make sure it's in the same directory.")
    exit()

# Step 2: Check for actual column names
print(" Loaded CSV. Found columns:", list(df.columns))

# Step 3: Preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#  Use your actual column names: 'Resume' and 'Category'
df['Cleaned_Text'] = df['Resume'].apply(clean_text)

# Step 4: Vectorize the cleaned text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Category']

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Step 8: Save the model and vectorizer
joblib.dump(model, 'resume_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n Model and vectorizer saved as 'resume_classifier_model.pkl' and 'tfidf_vectorizer.pkl'")
