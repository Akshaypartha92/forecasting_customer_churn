## Energy - Forecasting Customer Churn
We have been given Synthetic datasets representing customer data for an energy company over a period of 12 months. Each month's dataset contains various features related to customer usage, billing, interactions, demographics, and contract information. The goal is to use these datasets to build a model that predicts customer churn for the next month.

### Data Description
We have 12 csv files, each representing one month of data. Each file includes the following features:

CustomerID: Unique identifier for each customer.
UsageData: Monthly usage data in kWh.
BillingAmount: Monthly billing amount in dollars.
PaymentHistory: Payment status (0: late payment, 1: on-time payment).
CustomerServiceInteractions: Number of interactions with customer service.
Complaints: Number of complaints in the last year.
PromotionsReceived: Number of promotions received in the last year.
ContractType: Type of contract (0: fixed, 1: variable).
ContractDuration: Duration of the contract in months.
Age: Age of the customer.
Gender: Gender of the customer (Male, Female).
Income: Annual income of the customer.
HouseholdSize: Number of people in the customer's household.
Churned_MonthX: Binary target indicating whether the customer churned in the current month (0: stayed, 1: churned).

Customer Churn Prediction Report for Energy Sector

Customer Churn Prediction Report for Energy Sector

## Introduction

This report outlines the methodology and findings from a customer churn prediction analysis conducted on energy custiomer churn data. The primary objective was to identify patterns and predictors of customer churn to inform retention strategies. The dataset comprised 12 CSV files, each representing a month's data, without explicit temporal identifiers.

## Data Processing

Data Consolidation and Cleaning
Data Integration: Combined 12 monthly CSV files into a single dataset due to the absence of explicit temporal markers, ensuring a unified structure for analysis.​

Variable Consistency Check: Performed a thorough examination of variable consistency across all files to ensure uniformity in data representation.​

Anomaly Handling: Identified and corrected anomalies, such as negative values in the CustomerServiceInteractions variable, which were logically inconsistent. These values were set to zero to maintain data integrity.​
Feature Engineering

Categorical Encoding: Applied one-hot encoding to categorical variables, including binned income levels and contract types, to facilitate their use in machine learning models.​

Numerical Imputation: Addressed missing values in UsageData and BillingAmount by imputing the median values grouped by ContractType. This approach was chosen over KNN-based imputation due to better performance in preliminary tests.​

Demographic Binning: Created bins for demographic variables such as age, income, household size, and contract duration to capture non-linear relationships and reduce model complexity.​
Handling Class Imbalance

Given the dataset's imbalance (10% churners vs. 90% non-churners), several techniques were employed to address this issue:​

Oversampling: Utilized Synthetic Minority Over-sampling Technique (SMOTE) and SMOTE combined with Tomek links to generate synthetic examples of the minority class, enhancing the model's ability to learn from limited churn data.​

Undersampling: Implemented Cluster-Based UnderSampling (CUBE) to reduce the majority class size, thereby balancing the class distribution and mitigating potential biases.​
These methods aimed to provide the model with a more balanced perspective, improving its predictive performance on the minority class.​

## Model Development

Multiple machine learning models were trained and evaluated, including:​

Logistic Regression
Decision Trees
XGBoost
LightGBM
Support Vector Machines (SVM)

The models were primarily evaluated based on F1 score and recall, metrics particularly relevant in imbalanced classification scenarios. A lower classification threshold of 0.3 (as opposed to the standard 0.5) was adopted to prioritize the identification of potential churners, aligning with the business objective of customer retention.​

## Evaluation Metrics and Model Performance

Given the synthetic nature of the dataset and its inherent biases, traditional accuracy metrics were deemed insufficient for evaluating model performance. Models exhibited high accuracy by predominantly predicting the majority class (non-churners), failing to identify actual churners effectively. This phenomenon, known as the accuracy paradox, underscores the limitations of accuracy in imbalanced datasets.​

To address this, the evaluation focused on:​

Recall: Measuring the model's ability to correctly identify actual churners, ensuring that at-risk customers are not overlooked.​

Lift Factor: Assessing the model's effectiveness in identifying churners compared to random selection, providing insights into its practical utility.​
By emphasizing these metrics, the evaluation aligned more closely with the business objective of minimizing customer churn.​

## Future Recommendations

To enhance the model's predictive capabilities and business applicability, the following recommendations are proposed:

Churn Risk Scoring: Develop a churn risk probability score ranging from 0% to 100%, enabling the categorization of customers into risk tiers (e.g., red, yellow, green) for targeted interventions.​

Feature Selection Techniques: Explore forward and backward stepwise regression methods to identify the most significant predictors of churn, potentially improving model interpretability and performance.​

Temporal Data Integration: Incorporate temporal variables, such as contract start dates, to capture seasonality and tenure effects, which may influence churn behavior.​

Data Integrity: Ensure the uniqueness of customer identifiers to prevent data duplication and maintain the accuracy of customer-level analyses.​

## Conclusion

The analysis successfully identified key factors associated with customer churn in the energy sector, utilizing a combination of data preprocessing, feature engineering, and advanced modeling techniques. Addressing class imbalance and optimizing model thresholds were crucial steps in enhancing predictive performance. Implementing the recommended strategies could further improve model accuracy and provide actionable insights for customer retention initiatives.​