# Telco Customer Churn Analysis

This repository contains an R-based statistical analysis of customer churn in the telecommunications sector, using the publicly available Telco Customer Churn dataset. The project is part of a data science coursework by **Luigi Mascolo**.

## 📊 Project Overview

The main objective of this project is to understand the drivers behind customer churn and develop predictive models to help businesses proactively retain their customers.

### Key Steps:

- **Data Preprocessing**: Cleaning and transforming the dataset for modeling.
- **Exploratory Data Analysis (EDA)**: Identifying key trends and potential churn patterns.
- **Modeling**:
  - Classical Logistic Regression
  - Penalized Logistic Regression (Lasso, Ridge, Elastic Net)
  - Distributed Logistic Regression using Spark
- **Model Evaluation**: Using metrics such as Accuracy and AUC to assess model performance.
- **Business Insights**: Interpreting model outputs to suggest data-driven strategies.

## 📁 Files

- `Project-Work-Telco Dataset.pdf` — Final report of the analysis (in PDF).
- `Project Work Telco Dataset.Rmd` — R Markdown file containing the code and narrative used to generate the report.

## 🛠️ Technologies Used

- **R**, **RMarkdown**
- **tidyverse**, **glm**, **glmnet**, **sparklyr**, **ggplot2**
- **Apache Spark** (via R interface)

## 🧠 Insights

- Customers with long-term contracts, people to care for (dependents), or who use security/backup services are less likely to churn.
- Offer A and E, and month-to-month contracts are associated with higher churn.
- Credit card payments correlate with increased customer retention.

## 📄 License

This project is released under the MIT License.

---

© 2025 Luigi Mascolo
