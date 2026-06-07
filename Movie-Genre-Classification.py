# ============================================================
# Movie Genre Classification using Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# Sample Dataset
# ------------------------------------------------------------

data = {
    "plot": [
        "A superhero saves the world from dangerous villains",
        "Two people fall in love despite family conflicts",
        "A detective investigates a mysterious murder case",
        "Aliens invade earth and humans fight back",
        "A young couple experiences a romantic journey",
        "A police officer tracks a serial killer",
        "An astronaut travels through space to save humanity",
        "A family struggles with emotional relationships",
        "A warrior battles enemies in an epic war",
        "Friends share funny experiences in college"
    ],

    "genre": [
        "Action",
        "Romance",
        "Thriller",
        "Sci-Fi",
        "Romance",
        "Thriller",
        "Sci-Fi",
        "Drama",
        "Action",
        "Comedy"
    ]
}

df = pd.DataFrame(data)

print("\nDataset Preview:")
print(df.head())

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["plot"],
    df["genre"],
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# ML Pipeline
# ------------------------------------------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("classifier", MultinomialNB())
])

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

print("\nModel Performance")
print("-" * 40)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")
plt.title("Movie Genre Classification")
plt.show()

# ------------------------------------------------------------
# User Prediction
# ------------------------------------------------------------

print("\nMovie Genre Predictor")
print("-" * 40)

new_plot = input("Enter movie plot: ")

prediction = model.predict([new_plot])[0]

print("\nPredicted Genre:", prediction)
