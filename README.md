# 📉 Customer Churn Predictor

An AI-powered customer churn prediction system that uses machine learning to identify customers who are likely to leave a telecom service. The application also includes an **AI Business Advisor** that explains the prediction, highlights risk factors, and suggests customer-retention actions.

## 🚀 Live Demo

**Try the application:**
https://churnpredictor678dk.streamlit.app/

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies. This project uses customer information such as tenure, monthly charges, contract type, payment method, and subscribed services to predict whether a customer is likely to churn.

The machine learning model makes the churn prediction, while an LLM-based AI Business Advisor provides an explanation and practical retention suggestions.

## ✨ Features

* 📊 Customer churn prediction
* 🤖 AI-powered Business Advisor
* 📈 Churn probability and prediction confidence
* 🔍 Key customer risk factors
* 💡 Customer-retention suggestions
* 📋 Customer information summary
* 🧠 Multiple machine learning models
* 🎨 Interactive Streamlit interface

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **Streamlit**
* **Groq LLM API**
* **Joblib**
* **Matplotlib**
* **Seaborn**

## 📂 Dataset

This project uses the **IBM Telco Customer Churn Dataset**.

The dataset contains customer information including:

* Tenure
* Monthly Charges
* Contract
* Payment Method
* Internet Service
* Online Security
* Tech Support
* Streaming Services
* Churn status

## ⚙️ Machine Learning Workflow

```text
Customer Information
        ↓
Data Preprocessing
        ↓
Feature Transformation
        ↓
Machine Learning Model
        ↓
Churn Prediction
        ↓
Probability & Confidence
        ↓
AI Business Advisor
        ↓
Explanation + Risk Factors + Retention Suggestions
```

## 🧠 Machine Learning Models

The project explores and compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Special attention is given to **recall**, since identifying customers who are actually at risk of churn is important for customer-retention use cases.

## 🤖 AI Business Advisor

The application sends the customer information and the ML prediction to an LLM.

The LLM does **not** make the churn prediction. Instead, it explains the machine learning result and generates:

* Explanation of the prediction
* Key risk factors
* Possible reasons for churn
* Three practical retention suggestions

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/dineshkumar678rb/Churn_Predictor.git
cd Churn_Predictor
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

## 🔐 Environment Variables

The Groq API key is stored in `.env` and is excluded from GitHub using `.gitignore`.

**Never commit your API key to the repository.**

## 📁 Project Structure

```text
Churn_Predictor/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.joblib
│   ├── metrics.json
│   ├── cm_decision_tree.png
│   ├── cm_logistic_regression.png
│   └── cm_random_forest.png
│
├── app.py
├── eda.py
├── train.py
├── churn_distribution.png
├── requirements.txt
└── .gitignore
```

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Binary classification
* Data preprocessing
* Categorical feature handling
* Logistic Regression
* Decision Trees
* Random Forest
* Model evaluation
* Precision and recall
* Confusion matrices
* Streamlit application development
* LLM API integration
* Prompt engineering
* Combining machine learning with LLM-based explanations

## 👨‍💻 Author

**Dinesh Kumar R B**

BCA Student | Machine Learning & Full-Stack Development Enthusiast
