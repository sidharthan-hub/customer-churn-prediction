# Customer Churn Prediction

## Project Overview

Customer churn occurs when customers stop using a company's service.
In highly competitive industries like telecommunications, retaining existing customers is more cost-effective than acquiring new ones.

This project uses **Machine Learning (Logistic Regression)** to predict whether a telecom customer is likely to churn based on their service usage and billing information.

The model generates **churn probability scores** so companies can identify high-risk customers and take preventive actions such as offering discounts or improving services.

---

## Dataset

**IBM Telco Customer Churn Dataset**

Total customers: **7,043**

Target variable: **Churn (Yes / No)**

The dataset contains information such as:

* Gender
* SeniorCitizen
* Partner
* Dependents
* Tenure
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod
* MonthlyCharges
* TotalCharges

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression

---

## Project Workflow

1. Load the telecom churn dataset
2. Data cleaning and preprocessing
3. Convert TotalCharges to numeric values
4. Handle missing values
5. Encode categorical variables
6. Split dataset into training and testing data
7. Scale numerical features
8. Train Logistic Regression model
9. Evaluate model performance
10. Generate churn probability scores
11. Save trained model (`model.pkl`)

---

## Model Evaluation

The model performance is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

These metrics help determine how well the model predicts customer churn.

---

## Risk Categories

Based on churn probability:

| Probability Score | Risk Level    |
| ----------------- | ------------- |
| 0.8 – 1.0         | Critical Risk |
| 0.5 – 0.8         | Moderate Risk |
| < 0.5             | Low Risk      |

This helps businesses take appropriate actions to retain customers.

---

## Output Files

* **model.pkl** → Trained machine learning model
* **churn_predictions.csv** → Predicted churn risk scores

---

## How to Run the Project

Install required packages:

```
py -m pip install -r requirements.txt
```

Run the model:

```
py main.py
```

---

## Business Impact

By identifying customers at risk of leaving, companies can:

* Provide targeted discounts
* Improve customer support
* Increase customer retention
* Improve overall revenue

Even retaining **5% of high-risk customers** can significantly increase company profit.

---

## Author

**Sidharthan M R**
