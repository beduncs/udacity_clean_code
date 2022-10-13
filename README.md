# Predict Customer Churn
Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is used to demonstrate a customer churn analysis using Clean Code best practices. This project is intended to demonstrate composing, testing, and logging an analysis in modular functions using PEP 8 coding standards.

## Files and data description

The directory structure for the projects is as follows:

```
.
├── Guide.ipynb          # Provided by Udacity: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Provided by Udacity: Contains the code to be refactored
├── churn_library.py     # Functions defined for the churn analysis
├── churn_script_logging_and_tests.py # Testing and logs for executing the churn analysis
├── constants.py # Constant values used in the script
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Data provided for the analysis
│   └── bank_data.csv
├── images               
│   ├── eda              # Store EDA image results
│   └── results          # Store model training/testing results
├── logs                 # Store logs
└── models               # Store models
```

The directory contains two primary submission artifacts from the project, the `churn_library.py` and `churn_script_logging_and_tests.py` files.
- The `churn_library.py` module contains functions composed to complete the churn analysis. 
- The `churn_script_logging_and_tests.py` file is provided to demonstrate an example execution of the churn library functions, as well as a demonstration of logging and unit testings for the functions.

## Dependencies
This project relies on the following dependencies
- [scikit-learn](https://scikit-learn.org)
- [joblib](https://joblib.readthedocs.io)
- [pandas](https://pandas.pydata.org)
- [numpy](https://www.numpy.org)
- [matplotlib](https://matplotlib.org)
- [seaborn](https://seaborn.pydata.org)

## Running Files
In order to execute these files, an environment of python dependencies must be defined. The `churn_script_logging_and_tests.py` can be executed from the command line while the `churn_library.py` files provides a library of functions to be imported.

- The python environment can be resolved by executing `python -m pip install -r requirements_py3.6.txt`
- An example of logging/testing with this analysis library can be executed using `ipython churn_script_logging_and_tests.py`

Executing the `churn_script_logging_and_tests.py` files will write a log in to the `./logs` directory and produced images to the `./images/`.

The image locations, model parameters and other constants can be modified using the `constants.py` file.
