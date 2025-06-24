# News Article Classification with NLP and Machine Learning

This project uses natural language processing (NLP) and machine learning to classify news articles into distinct categories based on their headline and short description. Built using Python in JupyterLab, it explores multiple classification models and NLP pipelines, and includes topic modeling for unsupervised insights.

---

## Dataset

- **Source**: [News Category Dataset on Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) by Rishab Misra  
- The dataset contains over 200,000 news articles, each with a headline, short description, and labeled category.

---

## Project Workflow

### 🔹 1. Data Cleaning & Preparation
- Combined headlines and short descriptions for better context
- Removed special characters, lowercased text
- Mapped similar categories into a consolidated set (e.g., merging *ARTS* and *ARTS & CULTURE*)
- Balanced the dataset by resampling to ensure equal class distribution across 10 selected categories

### 🔹 2. Baseline Modeling
- Implemented a random classifier using `DummyClassifier` for performance benchmarking

### 🔹 3. NLP Preprocessing
- Used **spaCy** for tokenization
- Applied traditional vectorization methods (TF-IDF, CountVectorizer)
- Integrated **fastText** embeddings for semantic-aware representation

### 🔹 4. Model Training
- Trained and evaluated multiple supervised models:
  - **SGDClassifier**
  - **Multinomial Naive Bayes**
  - **Logistic Regression**
  - **SVM**
- Used `RandomizedSearchCV` for hyperparameter tuning
- Visualized performance with **confusion matrices** and reported metrics like:
  - Accuracy
  - F1 Macro
  - F1 Weighted

### 🔹 5. Topic Modeling (Unsupervised Learning)
- Explored latent themes using **Latent Dirichlet Allocation (LDA)** via `tomotopy`
- Generated interpretable topic-word distributions and visualizations using t-SNE

---

## Results

- Achieved over **70% classification accuracy** on the test set
- Found strong alignment between fastText + logistic models and topic-specific categories
- LDA revealed underlying themes not directly tied to labels, offering content-based insights

---
