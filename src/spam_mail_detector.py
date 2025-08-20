"""
Spam Mail Detector (Pro+ Version)
---------------------------------

- Loads SMS Spam Collection dataset
- Cleans and preprocesses text (stopwords, stemming, TF-IDF)
- Trains multiple models (Naive Bayes, Logistic Regression, SVM, Random Forest)
- Performs cross-validation & hyperparameter tuning (GridSearchCV)
- Evaluates models with classification reports, confusion matrices, ROC curves
- Saves plots to figures/
- Saves best models to models/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import joblib
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure folders exist
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/spam.csv", sep='\t', names=["label", "message"], encoding="latin-1")
df = df[["label", "message"]]

# Preprocessing function
ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

df["clean_message"] = df["message"].apply(preprocess_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["clean_message"]).toarray()
y = df["label"].map({"ham": 0, "spam": 1}).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel='linear', probability=True),
    "RandomForest": RandomForestClassifier(random_state=42)
}

results = {}
best_params = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\n{name} - CV Accuracy: {cv_scores.mean():.4f}")
    
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"figures/{name}_confusion_matrix.png")
    plt.close()

    # ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(f"figures/{name}_roc.png")
    plt.close()

# 🔍 Optimized Hyperparameter tuning (GridSearchCV for consistent results)
print("\n🔍 Hyperparameter Tuning (GridSearchCV)...")

# --- SVM ---
svm = SVC(probability=True)
svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
svm_search = GridSearchCV(
    svm,
    svm_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)
svm_search.fit(X_train, y_train)
best_params["SVM"] = svm_search.best_params_
print("✅ Best SVM params:", svm_search.best_params_)

# Save best SVM model
joblib.dump(svm_search.best_estimator_, "models/best_svm_model.pkl")

# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
rf_search = GridSearchCV(
    rf,
    rf_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_params["RandomForest"] = rf_search.best_params_
print("✅ Best RandomForest params:", rf_search.best_params_)

# Save best RF model
joblib.dump(rf_search.best_estimator_, "models/best_rf_model.pkl")

# WordClouds
spam_words = " ".join(df[df.label=="spam"]["clean_message"])
ham_words = " ".join(df[df.label=="ham"]["clean_message"])

spam_wc = WordCloud(width=600, height=400).generate(spam_words)
ham_wc = WordCloud(width=600, height=400).generate(ham_words)

plt.figure(figsize=(10,5))
plt.imshow(spam_wc)
plt.axis("off")
plt.title("Spam WordCloud")
plt.savefig("figures/spam_wordcloud.png")
plt.close()

plt.figure(figsize=(10,5))
plt.imshow(ham_wc)
plt.axis("off")
plt.title("Ham WordCloud")
plt.savefig("figures/ham_wordcloud.png")
plt.close()

# Summary of results
print("\n📊 Final Results:")
for k,v in results.items():
    print(f"{k}: {v:.4f}")

print("\nBest Hyperparameters:")
print(best_params)

print("\n💾 Best models saved in 'models/' folder (SVM & RandomForest).")
