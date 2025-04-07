# Energy - Forecasting Customer Churn
We have been given Synthetic datasets representing customer data for an energy company over a period of 12 months. Each month's dataset contains various features related to customer usage, billing, interactions, demographics, and contract information. The goal is to use these datasets to build a model that predicts customer churn for the next month.

## Data Description
We have 12 csv files, each representing one month of data. Each file includes the following features:

1. CustomerID: Unique identifier for each customer.
2. UsageData: Monthly usage data in kWh.
3. BillingAmount: Monthly billing amount in dollars.
4. PaymentHistory: Payment status (0: late payment, 1: on-time payment).
5. CustomerServiceInteractions: Number of interactions with customer service.
6. Complaints: Number of complaints in the last year.
7. PromotionsReceived: Number of promotions received in the last year.
8. ContractType: Type of contract (0: fixed, 1: variable).
9. ContractDuration: Duration of the contract in months.
10. Age: Age of the customer.
11. Gender: Gender of the customer (Male, Female).
12. Income: Annual income of the customer.
13. HouseholdSize: Number of people in the customer's household.
14. Churned_MonthX: Binary target indicating whether the customer churned in the current month (0: stayed, 1: churned).
