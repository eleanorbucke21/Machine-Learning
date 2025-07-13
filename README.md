# 🤖 Machine Learning Coursework – Classification, NLP, PCA & AI Applications

This repository presents my final machine learning coursework project submitted for the **Diploma in Data Analytics for Business** at **CCT College Dublin**. The notebook demonstrates key machine learning concepts through four distinct real-world use cases involving classification, natural language processing, dimensionality reduction, and AI integration.

---

## 📘 Project Summary

The notebook `ML_CA2_SBA23145.ipynb` includes:

### 1. 🧠 Glass Type Classification
- Neural network model to predict glass types based on chemical composition
- Use of `StandardScaler` and Keras' `Sequential` model

### 2. 📚 Text Preprocessing – Kindle Reviews
- Natural Language Processing (NLP) on Amazon Kindle reviews
- Tokenization, stopword removal, lemmatization
- Cleaned dataset prepared for sentiment analysis

### 3. 🌟 Sentiment Classification – Kindle Ratings
- Neural network classification of review sentiment (1–5 stars)
- Model building, training, testing, and optimization
- Use of Keras and Scikit-learn's `GridSearchCV` for tuning

### 4. 💬 AI Use Case – Customer Support Automation
- Analysis of how Open API-based LLMs (e.g., ChatGPT) can improve customer support
- Real-world comparisons with tools like Netflix and YouTube recommendation engines
- Discussion of LLM benefits and implementation strategies in e-commerce

---

## 🧠 What I Learned

Through this multifaceted project, I developed skills in:

- Cleaning and analyzing both structured and unstructured datasets
- Performing NLP text preprocessing (tokenizing, lemmatizing, vectorizing)
- Building and optimizing neural network classifiers with Keras
- Applying PCA to reduce dimensionality while retaining 99.5% variance
- Evaluating classification models using accuracy, confusion matrix, and classification report
- Exploring real-world AI use cases using LLMs for customer service

---

## 🧪 Techniques and Topics Covered

- Exploratory Data Analysis (EDA)
- Data cleaning for both numeric and text data
- Neural network architecture with Keras
- Hyperparameter tuning with `GridSearchCV`
- Principal Component Analysis (PCA)
- NLP preprocessing: `nltk`, `wordnet`, stopwords
- Use case analysis for AI in customer support

---

## 📁 Project Structure

```
MACHINE-LEARNING/
├── all_kindle_review.csv
├── glass_data.csv
├── spambase.csv
├── ML_CA2_SBA23145.ipynb 
├── images/ 
├── LICENSE
└── README.md
```

---

### 📦 Datasets

- **`glass_data.csv`** – Main dataset used for PCA and classification  
- **`all_kindle_review.csv`** – Supplementary dataset used for NLP practice

---

### 🧾 Jupyter Notebook

- **`ML_CA2_SBA23145.ipynb`** – Fully documented notebook with:
  - Code  
  - Visualizations  
  - PCA analysis  
  - Confusion matrix  
  - Markdown explanations  
  - Final reflections

---

## 🧰 Tools and Technologies Used

### 📦 Libraries and Frameworks
- `numpy`, `pandas` – Data manipulation and analysis  
- `matplotlib`, `seaborn` – Visualizations  
- `scikit-learn` – Preprocessing, PCA, classification, evaluation  
- `tensorflow.keras` – Deep learning model architecture  
- `scikeras.wrappers` – Integration of Keras with scikit-learn  
- `google.colab` – Optional file upload utility

### 🧪 Machine Learning Techniques
- Train/test split and cross-validation  
- StandardScaler for feature scaling  
- PCA for reducing dimensionality  
- Confusion matrix and classification report for evaluation

---

### 💻 Imported Libraries

```python
# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing & Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from scikeras.wrappers import KerasClassifier

# Model Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Google Colab-Specific Utilities
from google.colab import files
```

---

## 📚 References

Ahmed, T. (2024). *Exploratory Data Analysis*. Moodle, CCT College Dublin  
Ahmed, T. (2024). *PCA and Dimensionality Reduction*. Moodle  
McQuaid, D. & Ahmed, T. (2024). *Jupyter Notebook Tutorial*. Moodle

---

> 📝 This notebook reflects my ability to apply core machine learning methods, dimensionality reduction, and academic reporting within a practical context using real datasets.
