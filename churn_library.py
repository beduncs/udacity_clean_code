'''
The churn library provides functions for performing churn analysis.

Author: Ben Duncan
Date: 10/07/2022
'''

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import constants

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    '''
    # Read DataFrame from csv file
    data_df = pd.read_csv(pth)
    # Add Churn column to DataFrame
    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_df


def save_plot(figure, pth):
    '''
    get figure from subplot and save to a specified path
    input:
            figure: matplotlib figure
            pth: string to destination path

    output:
            None
    '''
    figure.savefig(pth)
    figure.clf()


def perform_eda(data_df):
    '''
    perform eda on data_df and save figures to images folder
    input:
            data_df: pandas dataframe

    output:
            None
    '''
    # Plot and save churn distribution
    churn_hist = data_df['Churn'].hist()
    save_plot(churn_hist.get_figure(), constants.CHURN_HIST_PTH)

    # Plot and save customer age distribution
    customer_age_hist = data_df['Customer_Age'].hist()
    save_plot(customer_age_hist.get_figure(), constants.CUSTOMER_AGE_HIST_PTH)

    # Plot and save marital status distribution
    marital_status_hist = data_df.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    save_plot(marital_status_hist.get_figure(),
              constants.MARITAL_STATUS_HIST_PTH)

    # Plot and save Total Transaction Distribution
    transaction_hist = sns.histplot(
        data_df['Total_Trans_Ct'], stat='density', kde=True)
    save_plot(transaction_hist.get_figure(), constants.TRANSACTION_HIST_PTH)

    # Plot and save correlation heatmap
    correlation_heatmap = sns.heatmap(
        data_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    save_plot(correlation_heatmap.get_figure(),
              constants.CORRELATION_HEATMAP_PTH)


def encoder_helper(data_df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            data_df: pandas dataframe with new columns for encoded variables
            encoded_columns: List containing the new encoded columns names
    '''
    encoded_columns = []
    for category in category_lst:
        # Encode categorical columns
        cat_lst = []
        cat_groups = data_df.groupby(category).mean()['Churn']

        for val in data_df[category]:
            cat_lst.append(cat_groups.loc[val])

        cat_name = f'{category}_Churn'
        # append categorical column name to the response list
        encoded_columns.append(cat_name)

        # add categorical columns to the dataframe
        data_df[cat_name] = cat_lst

    return data_df, encoded_columns


def perform_feature_engineering(data_df, encoded_columns):
    '''
    input:
              data_df: pandas dataframe
              encoded_columns: List of columns names of the encoded features

    output:
            x_data: x data
            y_data: y data
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    keep_cols = constants.QUANT_COLUMNS + encoded_columns
    x_data = data_df[keep_cols]
    y_data = data_df['Churn']
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)
    return x_data, y_data, x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    save_plot(plt, constants.LOGISTIC_RESULTS_PTH)
    plt.clf()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    save_plot(plt, constants.RF_RESULTS_PTH)
    plt.clf()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    save_plot(plt, output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    '''

    # unwrap splits

    # grid search
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # Build out model architecture
    lrc = LogisticRegression(
        solver=constants.SOLVER,
        max_iter=constants.MAX_ITER)

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=constants.PARAM_GRID,
        cv=constants.CV)

    # Fit the models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Evaluate predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Plot roc curve results
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    roc_ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=roc_ax,
        alpha=0.8)
    lrc_plot.plot(ax=roc_ax, alpha=0.8)
    save_plot(plt, constants.ROC_CURVE_RESULTS_PTH)

    # Save model off in pickle files
    joblib.dump(cv_rfc.best_estimator_, constants.RFC_MODEL_PTH)
    joblib.dump(lrc, constants.LOGISTIC_MODEL_PTH)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr
