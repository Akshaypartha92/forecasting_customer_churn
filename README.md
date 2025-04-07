# Energy - Forecasting Customer Churn
We have been given Synthetic datasets representing customer data for an energy company over a period of 12 months. Each month's dataset contains various features related to customer usage, billing, interactions, demographics, and contract information. The goal is to use these datasets to build a model that predicts customer churn for the next month.

## Data Description
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
