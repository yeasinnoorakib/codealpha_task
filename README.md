# 🚀 CodeAlpha Machine Learning Projects

**Author:** Mohammad Yasin Nur Akib  
**Internship Program:** CodeAlpha – Machine Learning Internship  

This repository contains **two machine learning projects** completed as part of the CodeAlpha internship.  
The projects focus on real-world applications of **machine learning and deep learning** using audio and medical data.

---

## 📌 Projects Overview

1. Emotion Recognition from Speech  
2. Disease Prediction from Medical Data  

Each project follows a complete machine learning pipeline including preprocessing, model training, evaluation, and visualization.

---

# 🧠 Project 1: Emotion Recognition from Speech

## 🔹 Problem Statement
The goal of this project is to **recognize human emotions from speech audio signals**.  
Emotion recognition is widely used in applications such as sentiment analysis, voice assistants, and human–computer interaction.

---

## 🔹 Dataset
- **Dataset Name:** Toronto Emotional Speech Set (TESS)
- **Source:** Kaggle  
- **Data Type:** Audio (`.wav` files)
- **Speakers:** Female speakers
- **Emotion Classes (7):**
  - Neutral
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprise

---

## 🔹 Feature Extraction
- **MFCC (Mel-Frequency Cepstral Coefficients)** are extracted from audio signals.
- Audio preprocessing steps:
  - Resampling
  - Padding/trimming to fixed duration
  - Normalization

---

## 🔹 Model Used
- **Convolutional Neural Network (CNN)**
- Implemented using **PyTorch**
- MFCC features treated as 2D inputs

---

## 🔹 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🔹 How to Run
pip install -r requirements.txt
python -m src.train
python -m src.evaluate


---

# 🧠 Project 2: Disease Prediction from Medical Data


## 📌 Overview
This project predicts the presence of heart disease using patient medical data and machine learning models.

---

## 📂 Dataset
- **Name:** Heart Disease Dataset
- **Source:** Kaggle / UCI Machine Learning Repository
- **Data Type:** CSV (tabular data)
- **Target:** Presence or absence of heart disease

---

## 🔍 Data Preprocessing
- Handling missing values
- Feature scaling
- Encoding categorical features
- Train-test split

---

## 🧠 Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## ▶️ How to Run
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.interpret
