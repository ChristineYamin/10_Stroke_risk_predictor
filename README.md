# Project 10 - Stroke Risk Prediction System (MLOps)

This repository contains the complete MLOps pipeline for a **Stroke Risk Prediction System**, built as part of the **"23 projects at 23"** series. The system uses machine learning to predict the likelihood of a stroke based on patient health metrics, with a strong emphasis on handling severe class imbalance and ensuring deployment readiness through automated testing.

## Project Overview

Medical datasets for stroke prediction are highly imbalanced, with positive cases (strokes) making up a very small percentage of the data. This project focuses on building a model that prioritizes **high recall** to minimize false negatives, ensuring that at-risk patients are accurately identified.

### Key Highlights:
* **Model Performance:** Achieved **82% recall** on the minority class using XGBoost.
* **Production Threshold Tuning:** Lowered the prediction threshold to 0.18 to ensure high-risk cases are caught.
* **MLOps / CI/CD:** Automated testing pipeline via GitHub Actions.
* **Streamlit Web Application:** A user-friendly UI deployed on the cloud.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** XGBoost, Scikit-learn, Imbalanced-learn
* **Deployment:** Streamlit Community Cloud
* **CI/CD:** GitHub Actions
* **Version Control:** Git & GitHub

---

## Model Evoluation and Experiment
I conducted 6 iterations to find the optimal balance between accuracy and the critical need for high recall in a medical context:

Iteration,Method,Focus,Key Outcome
Model 1,Baseline Random Forest,Default parameters,"High accuracy, but low recall for strokes."
Model 2,Balanced Random Forest,class_weight='balanced',Slight improvement in identifying stroke cases.
Model 3,SMOTE + Random Forest,Synthetic oversampling,Better minority class representation.
Model 4,Tuned RF (Threshold 0.20),Probability adjustment,Significant recall boost.
Model 5,XGBoost (Weighted),scale_pos_weight,Superior gradient boosting for imbalanced data.
Model 6,Final Pipeline,XGBoost + 0.18 Threshold,The Winner: Maximized recall to 82%.

## Why Model 6?
In stroke prediction, a False Negative (failing to identify a patient at risk) is far more dangerous than a False Positive (a healthy patient being flagged for further testing). By moving from a standard Random Forest to a weighted XGBoost and lowering the decision threshold to 0.18, I optimized the system to prioritize patient safety, ensuring the majority of high-risk cases are caught by the model.


## Model Evaluation Metrics
 Accuracy - 0.67
 Recall(Class1 - Stroke ) - 0.82
 Precision(class 1 - Stroke) - 0.11
 ROC-AUC - 0.77

 ## CI/CD Pipeline
 The GitHub Actions pipeline automatically tests the integrity of the serialized model on every push to the main branch. It simulates an inference of a high-risk patient and asserts the prediction threshold to avoid regression.


 ## Live Demo
https://10strokeriskpredictor-iwrrbcuqnrmsgwo4zhr6sh.streamlit.app/