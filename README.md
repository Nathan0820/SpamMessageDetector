# Spam Message Detector

A binary text classifier that detects spam messages using NLP and scikit-learn.

## Overview

This project takes a dataset of 5,572 SMS messages labeled as **ham** (legitimate) or **spam**, cleans and vectorizes the text, then trains and compares five different ML models to find the best performer.

## Pipeline

1. **Text Preprocessing** — lowercase, remove punctuation, strip stop words, apply Porter stemming
2. **Feature Extraction** — TF-IDF vectorization (up to 5,000 features, unigrams + bigrams)
3. **Model Training** — train five classifiers on a 90/10 stratified split
4. **Evaluation** — compare accuracy, precision, recall, and F1 score
5. **Hyperparameter Tuning** — GridSearchCV on the best model (SVM)

## Models Compared

| Model               | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Naive Bayes          | 0.9731   | 1.0000    | 0.8000 | 0.8889   |
| Logistic Regression  | 0.9695   | 1.0000    | 0.7733 | 0.8722   |
| **SVM**              | **0.9785** | **1.0000** | **0.8400** | **0.9130** |
| Random Forest        | 0.9767   | 1.0000    | 0.8267 | 0.9051   |
| Decision Tree        | 0.9642   | 0.8873    | 0.8400 | 0.8630   |

SVM achieved the highest F1 score and was further tuned with GridSearchCV.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 messages (4,825 ham / 747 spam), stored in `data/spamMessage.csv`.

## Tech Stack

- Python 3.9
- pandas, NumPy
- NLTK (stopwords, stemming)
- scikit-learn (TF-IDF, SVM, Naive Bayes, Logistic Regression, Random Forest, Decision Tree, GridSearchCV)
- matplotlib, seaborn

## Getting Started

```bash
git clone https://github.com/Nathan0820/SpamMessageDetector.git
cd SpamMessageDetector
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

Then open `SpamEmailDetection.ipynb` and run all cells.
