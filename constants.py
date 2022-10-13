'''
This module contains the constant values used in the churn library.

Author: Ben Duncan
Date: 10/07/2022
'''
# Data paths
DATA_PTH = r"./data/bank_data.csv"

# Image paths
CHURN_HIST_PTH = './images/eda/churn_distribution.jpg'
CUSTOMER_AGE_HIST_PTH = './images/eda/customer_age_distribution.jpg'
MARITAL_STATUS_HIST_PTH = './images/eda/marital_status_distribution.jpg'
TRANSACTION_HIST_PTH = './images/eda/total_transaction_distribution.jpg'
CORRELATION_HEATMAP_PTH = './images/eda/heatmap.jpg'
ROC_CURVE_RESULTS_PTH = './images/results/roc_curve_result.jpg'
LOGISTIC_RESULTS_PTH = './images/results/logistic_results.jpg'
RF_RESULTS_PTH = './images/results/rf_results.jpg'
FEATURE_IMPORTANCES_PTH = './images/results/rf_results.jpg'
SHAP_PTH = './images/results/shap_summary.jpg'
TEST_PLOT_PTH = './images/test.jpg'

# Model paths:
RFC_MODEL_PTH = './models/rfc_model.pkl'
LOGISTIC_MODEL_PTH = './models/logistic_model.pkl'

# Feature constants
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

# Model params
PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

TEST_SIZE = 0.3
RANDOM_STATE = 42
SOLVER = 'lbfgs'
MAX_ITER = 3000
CV = 5
