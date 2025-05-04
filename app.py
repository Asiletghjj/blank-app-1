
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')
st.title("Student Performance ML App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_.csv")
    return df

df = load_data()

# Data exploration
st.subheader("Dataset Preview")
st.write(df.head())

# Data preprocessing
X = df.drop(columns=["timestamp", "exam_score"])
y = df["exam_score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for supervised models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Supervised Learning: Random Forest Regressor ---
st.subheader("1. Supervised: Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

st.write("**Random Forest Evaluation**")
st.write("MSE:", mean_squared_error(y_test, y_pred_rf))
st.write("R2 Score:", r2_score(y_test, y_pred_rf))

# Learning curve
st.write("Learning Curve")
train_sizes, train_scores, test_scores = learning_curve(rf, X_scaled, y, cv=5, scoring='r2')
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label="Train Score")
plt.plot(train_sizes, test_mean, label="Test Score")
plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.title("Random Forest Learning Curve")
plt.legend()
st.pyplot(plt)

# --- Unsupervised Learning: KMeans ---
st.subheader("2. Unsupervised: KMeans Clustering")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters
st.write("Cluster Distribution")
st.bar_chart(df["cluster"].value_counts())

# Visualizing clusters with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
scatter = ax.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis')
ax.set_title("KMeans Clusters (PCA Projection)")
st.pyplot(fig)

# --- Classification Model: Logistic Regression ---
st.subheader("3. Classification: Logistic Regression (High vs Low Exam Score)")
df["target_class"] = (df["exam_score"] > df["exam_score"].median()).astype(int)
y_class = df["target_class"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train_clf, y_train_clf)
y_pred_clf = clf.predict(X_test_clf)

st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_clf, y_pred_clf))

st.write("Classification Report:")
st.text(classification_report(y_test_clf, y_pred_clf))

# Learning curve for classifier
st.write("Learning Curve (Classifier)")
train_sizes, train_scores, test_scores = learning_curve(clf, X_scaled, y_class, cv=5, scoring='accuracy')
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label="Train Accuracy")
plt.plot(train_sizes, test_mean, label="Test Accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Learning Curve")
plt.legend()
st.pyplot(plt)
