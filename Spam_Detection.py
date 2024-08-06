import pandas as pd
import zipfile
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# URL for the dataset ZIP file
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# Download and extracting the dataset
zip_filename = 'smsspamcollection.zip'
dataset_filename = 'SMSSpamCollection'

# Download the ZIP file
response = requests.get(zip_url)
with open(zip_filename, 'wb') as file:
    file.write(response.content)

# Extract the ZIP file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset
df = pd.read_csv(dataset_filename, sep='\t', header=None, names=['label', 'message'])

# Encode labels: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check class distribution
print("Class distribution:")
print(df['label'].value_counts())

# Split the data into features and labels
X = df['message']
y = df['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines TF-IDF vectorization and SVM classifier
pipeline = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear')
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

def classify_message(message):
    """
    Classify a new message as 'spam' or 'ham'.
    """
    prediction = pipeline.predict([message])
    return 'spam' if prediction[0] == 1 else 'ham'

# Test the classifier with user input
def main():
    while True:
        user_input = input("\nEnter a message to classify (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = classify_message(user_input)
        print(f"The message is classified as: {result}")

if __name__ == "__main__":
    main()

# Clean up extracted files
os.remove(zip_filename)
os.remove(dataset_filename)
