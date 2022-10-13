'''
The churn script logging and test module provides functions
for testing the churn library analysis functions and logging the results.

Author: Ben Duncan
Date: 10/09/2022
'''

import os
from os.path import exists
import logging

import joblib
import matplotlib.pyplot as plt

import constants
import churn_library as cl

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data, pth):
    '''
    tests the data import function and logs results
    input:
            import_data: function from library
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    '''
    try:
        imported_df = import_data(pth)
        logging.info("Imported DataFrame: \n %s", imported_df.head())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        # Ensure the dataframe has rows and columns
        assert imported_df.shape[0] > 0
        assert imported_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return imported_df


def test_save_plot(save_plot, pth):
    '''
    tests the save plot function and logs results
    input:
            save_plot: function from library
            pth: a path to the save location
    output:
            None
    '''
    try:
        fig = plt.figure()
        save_plot(fig, pth)

        # Ensure the file was written
        assert exists(pth)
        logging.info("Testing save_plot: SUCCESS")

        # Clean up
        os.remove(pth)
    except AssertionError as err:
        logging.error(
            "Testing save_plot: The file doesn't appear to have been saveds")
        raise err


def test_eda(perform_eda, eda_data_df):
    '''
    test perform eda function and logs results
    input:
            perform_eda: function from library
            eda_data_df: pandas dataframe containing imported data
    output:
            None
    '''
    try:
        perform_eda(eda_data_df)

        # Ensure each of the expected plots were saved
        assert exists(constants.CHURN_HIST_PTH)
        assert exists(constants.CUSTOMER_AGE_HIST_PTH)
        assert exists(constants.MARITAL_STATUS_HIST_PTH)
        assert exists(constants.TRANSACTION_HIST_PTH)
        assert exists(constants.CORRELATION_HEATMAP_PTH)
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: An output file wasn't found")
        raise err


def test_encoder_helper(encoder_helper, encoder_data_df, category_lst):
    '''
    test encoder helper function and logs results
    input:
            encoder_helper: function from library
            encoder_data_df: pandas dataframe containing imported data
            category_lst: list of columns that contain categorical features
    output:
            eh_encoded_df: pandas dataframe with new columns for encoded variables
            eh_encoded_columns: List containing the new encoded columns names
    '''
    try:
        eh_encoded_df, eh_encoded_columns = encoder_helper(
            encoder_data_df, category_lst)

        # Ensure that the return dataframe is not empty
        assert not eh_encoded_df.empty
        logging.info("Encoded df: %s", eh_encoded_df.head())

        # Ensure that the return list of columns is not empty
        assert len(eh_encoded_columns) > 0
        logging.info("Encoded columns: %s", eh_encoded_columns)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: A return element appears to be emptyy")
        raise err

    return eh_encoded_df, eh_encoded_columns


def test_perform_feature_engineering(
        perform_feature_engineering,
        fe_encoded_df,
        fe_encoded_columns):
    '''
    test perform feature engineering function and logs results
    input:
            perform_feature_engineering: function from library
            fe_encoded_df: pandas dataframe containing imported data
            fe_encoded_columns: list of columns that contain categorical features
    output:
            fe_x_data: x data
            fe_y_data: y data
            fe_x_train: X training data
            fe_x_test: X testing data
            fe_y_train: y training data
            fe_y_test: y testing data
    '''
    try:
        fe_x_data, fe_y_data, fe_x_train, fe_x_test, fe_y_train, fe_y_test = perform_feature_engineering(
            fe_encoded_df, fe_encoded_columns)

        # Ensure that the returned arrays are not empty
        assert len(fe_x_train) > 0
        logging.info("x_train: %s", fe_x_train.head())
        assert len(fe_x_test) > 0
        logging.info("x_test: %s", fe_x_test.head())
        assert len(fe_y_train) > 0
        logging.info("y_train: %s", fe_y_train.head())
        assert len(fe_y_test) > 0
        logging.info("y_test: %s", fe_y_test.head())
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: A return element appears to be empty")
        raise err

    return fe_x_data, fe_y_data, fe_x_train, fe_x_test, fe_y_train, fe_y_test


def test_train_models(
        train_models,
        tm_x_train,
        tm_x_test,
        tm_y_train,
        tm_y_test):
    '''
    test train_models function and logs results
    input:
            train_models: function from library
            tm_x_train: X training data
            tm_x_test: X testing data
            tm_y_train: y training data
            tm_y_test: y testing data
    output:
            tm_y_train_preds_lr: training predictions from logistic regression
            tm_y_train_preds_rf: training predictions from random forest
            tm_y_test_preds_lr: test predictions from logistic regression
            tm_y_test_preds_rf: test predictions from random forest
    '''
    try:
        tm_y_train_preds_rf, tm_y_test_preds_rf, tm_y_train_preds_lr, tm_y_test_preds_lr = train_models(
            tm_x_train, tm_x_test, tm_y_train, tm_y_test)

        # Ensure the models were saved
        assert exists(constants.RFC_MODEL_PTH)
        assert exists(constants.LOGISTIC_MODEL_PTH)
        logging.info(
            "tm_y_train_preds_rf: %s, tm_y_test_preds_rf: %s, tm_y_train_preds_lr: %s, tm_y_test_preds_lr: %s, ",
            tm_y_train_preds_rf,
            tm_y_test_preds_rf,
            tm_y_train_preds_lr,
            tm_y_test_preds_lr)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: A return model appears to be missing")
        raise err

    return tm_y_train_preds_rf, tm_y_test_preds_rf, tm_y_train_preds_lr, tm_y_test_preds_lr


def test_classification_report_image(
        classification_report_image,
        cr_y_train,
        cr_y_test,
        cr_y_train_preds_lr,
        cr_y_train_preds_rf,
        cr_y_test_preds_lr,
        cr_y_test_preds_rf):
    '''
    test classification report image function and logs results
    input:
            classification_report_image: function from library
            cr_y_train: y training data
            cr_y_test: y testing data
            cr_y_train_preds_lr: training predictions from logistic regression
            cr_y_train_preds_rf: training predictions from random forest
            cr_y_test_preds_lr: test predictions from logistic regression
            cr_y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    try:
        classification_report_image(
            cr_y_train,
            cr_y_test,
            cr_y_train_preds_lr,
            cr_y_train_preds_rf,
            cr_y_test_preds_lr,
            cr_y_test_preds_rf)

        # Ensure the models were saved
        assert exists(constants.RF_RESULTS_PTH)
        logging.info("Results image at: %s", constants.RF_RESULTS_PTH)
        logging.info("Testing classification_report_image: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: The return path appears to be empty")
        raise err


def test_feature_importance_plot(
        feature_importance_plot,
        model,
        fi_x_data,
        output_pth):
    '''
    test feature_importance_plot function and logs results
    input:
            feature_importance_plot: function from library
            model: model object containing feature_importances_
            fi_x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    try:
        feature_importance_plot(model, fi_x_data, output_pth)

        # Ensure the models were saved
        assert exists(constants.FEATURE_IMPORTANCES_PTH)
        logging.info("Results image at: %s", constants.FEATURE_IMPORTANCES_PTH)
        logging.info("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: The return path appears to be empty")
        raise err


if __name__ == "__main__":
    # Test import data function
    data_df = test_import(cl.import_data, constants.DATA_PTH)

    # Test save plot function
    test_save_plot(cl.save_plot, constants.TEST_PLOT_PTH)

    # Test perform eda function
    test_eda(cl.perform_eda, data_df)

    # Test encoder helper function
    encoded_df, encoded_columns = test_encoder_helper(
        cl.encoder_helper, data_df, constants.CAT_COLUMNS)

    # Test perform feature engineering function
    x_data, y_data, x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, encoded_df, encoded_columns)

    # Test train models function
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = test_train_models(
        cl.train_models, x_train, x_test, y_train, y_test)

    # Test classification report image function
    test_classification_report_image(
        cl.classification_report_image,
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Test feature importance plot
    rfc_model = joblib.load(constants.RFC_MODEL_PTH)
    test_feature_importance_plot(
        cl.feature_importance_plot,
        rfc_model,
        x_data,
        constants.FEATURE_IMPORTANCES_PTH)
