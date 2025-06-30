# Analyzing Customer Churn in Telecommunications 📊 | GLM-Telco-Dataset

![GitHub release](https://img.shields.io/badge/releases-v1.0-blue.svg) [![Download](https://img.shields.io/badge/download-latest%20release-brightgreen.svg)](https://github.com/JuanQuimbayo/GLM-Telco-Dataset/releases)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project focuses on analyzing customer churn in a telecommunications company. Using the "Telco Customer Churn" dataset, we aim to identify patterns and factors contributing to customer attrition. By employing various statistical methods and machine learning techniques, we provide insights that can help in reducing churn rates.

## Dataset Description

The "Telco Customer Churn" dataset contains information about customers, including demographic details, account information, and services used. Key features include:

- Customer ID
- Gender
- Age
- Tenure (months)
- Services (e.g., phone, internet)
- Payment method
- Monthly charges
- Total charges
- Churn status (yes/no)

The dataset is available for download in the [Releases section](https://github.com/JuanQuimbayo/GLM-Telco-Dataset/releases). 

## Technologies Used

This project utilizes a variety of technologies and libraries, including:

- **R**: The primary programming language for data analysis.
- **RMarkdown**: For creating dynamic reports.
- **GLM**: Generalized Linear Models for statistical modeling.
- **Elastic Net**: A regularization technique that combines Lasso and Ridge regression.
- **Logistic Regression**: For predicting binary outcomes (churn vs. no churn).
- **Spark & Sparklyr**: For handling large datasets and distributed computing.

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JuanQuimbayo/GLM-Telco-Dataset.git
   ```

2. **Install R and RStudio** if you haven't already. You can download them from [CRAN](https://cran.r-project.org/) and [RStudio](https://www.rstudio.com/products/rstudio/download/).

3. **Install required R packages**. Open RStudio and run:

   ```R
   install.packages(c("dplyr", "ggplot2", "caret", "glmnet", "sparklyr"))
   ```

4. **Download the dataset** from the [Releases section](https://github.com/JuanQuimbayo/GLM-Telco-Dataset/releases) and place it in the project directory.

## Usage

After installation, you can start analyzing the dataset. Open the RMarkdown file in RStudio and knit it to generate the report. The analysis includes:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Model building and evaluation

## Exploratory Data Analysis (EDA)

EDA is a crucial step in understanding the dataset. In this project, we perform various analyses to visualize and summarize the data. Key steps include:

- **Data Cleaning**: Handling missing values and data types.
- **Visualizations**: Using `ggplot2` to create plots that illustrate customer demographics, service usage, and churn rates.

Here are some example visualizations:

- **Churn Rate by Gender**:
  
  ```R
  ggplot(data, aes(x = gender, fill = churn)) +
    geom_bar(position = "fill") +
    labs(title = "Churn Rate by Gender", y = "Proportion")
  ```

- **Monthly Charges vs. Churn**:

  ```R
  ggplot(data, aes(x = monthly_charges, fill = churn)) +
    geom_histogram(bins = 30, alpha = 0.5) +
    labs(title = "Monthly Charges Distribution by Churn Status")
  ```

## Modeling Techniques

We employ various modeling techniques to predict customer churn:

1. **Logistic Regression**: This model helps in understanding the relationship between customer characteristics and the likelihood of churn.

   ```R
   model <- glm(churn ~ age + tenure + monthly_charges, data = data, family = "binomial")
   ```

2. **Elastic Net**: Combines Lasso and Ridge regression for better performance on high-dimensional data.

   ```R
   cv_model <- cv.glmnet(x, y, alpha = 0.5)
   ```

3. **Random Forest**: A robust method that handles non-linear relationships and interactions between variables.

   ```R
   library(randomForest)
   rf_model <- randomForest(churn ~ ., data = data)
   ```

## Results

The results of our analysis provide insights into customer churn:

- **Key Factors**: Identified factors that significantly impact churn include monthly charges, tenure, and payment method.
- **Model Performance**: Evaluated model accuracy using metrics such as precision, recall, and F1-score.

Visual representations of model performance can be generated using confusion matrices and ROC curves.

## Contributing

We welcome contributions to improve this project. If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your branch and create a pull request.

Please ensure that your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For further updates, visit the [Releases section](https://github.com/JuanQuimbayo/GLM-Telco-Dataset/releases).