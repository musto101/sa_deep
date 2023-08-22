import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import GridSearchCV

# from sklearn import datasets
import optuna
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from lifelines.utils import concordance_index
from sklearn.impute import KNNImputer

# from lifelines.datasets import load_rossi

#
# X = load_rossi().drop('week', axis=1)  # keep as a dataframe
# Y = load_rossi().pop('week')
#
# model = sklearn_adapter(CoxPHFitter, event_col='arrest')
# EL = model()
#
# clf = GridSearchCV(EL, {
#    "penalizer": 10.0 ** np.arange(-2, 3),
#    "l1_ratio": [0, 1/3, 2/3]
# }, cv=4)
# clf.fit(X, Y)
#
# print(clf.best_estimator_)

# iris = datasets.load_iris()
# x = iris.data
# y = iris.target


def remove_one_sum_columns(df):
    # Calculate the column sums
    column_sums = df.sum(axis=0)

    # Identify columns with a sum of zero
    one_sum_columns = column_sums[column_sums <= 1].index

    # Remove the zero-sum columns
    df = df.drop(one_sum_columns, axis=1)

    return df


# Load your dataset into a Pandas DataFrame
# The dataset should include columns for time-to-event, event indicator, and covariates
# train = pd.read_csv('data/train.csv')
# valid = pd.read_csv('data/valid.csv')
#
# # train = remove_one_sum_columns(train)
# common_columns = train.columns.intersection(valid.columns)
# valid = valid.reindex(columns=common_columns)
#
# # Extract the time-to-event, event indicator, and covariates from the data
# # time_to_event = valid['last_visit']
# # event_indicator = valid['last_DX']
# # covariates = valid.drop(['last_visit', 'last_DX'], axis=1)
#
# time_to_event = train['last_visit']
# event_indicator = train['last_DX']
# # combine the two dataframes
# # y = pd.concat([time_to_event, event_indicator], axis=1)
# y = event_indicator
# train_cov = train.drop(['last_visit', 'last_DX'], axis=1)
# # change train_cov to float
# train_cov = train_cov.astype(float)
# y = y.astype(float)
#
# Define the objective function for hyperparameter tuning
# def objective(params):
#     alpha = params['alpha']
#     l1_ratio = params['l1_ratio']
#
#     # Fit the Cox proportional hazards model using the selected features and specified hyperparameters
#     cph = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
#     cph.fit(train, duration_col='last_visit', event_col='last_DX')
#
#     # Calculate the concordance index as the metric to optimize
#     c_index = concordance_index(event_indicator, cph.predict_partial_hazard(covariates))
#
#     return -c_index  # Minimize negative concordance index

# combining bayesian optimization with cross validation
# def objective(alpha, l1_ratio):
#
#     # Fit the Cox proportional hazards model using the selected features and specified hyperparameters
#     cph = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
#     kf = KFold(n_splits=5, shuffle=True)
#     # cph.fit(train, duration_col='last_visit', event_col='last_DX')
#
#     # Calculate the concordance index as the metric to optimize
#     scores = []
#     for train_index, test_index in kf.split(train):
#         X_train, X_test = train_cov.iloc[train_index], train_cov.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         cph.fit(X_train, y_train)
#         predicted_survival = cph.predict_survival_function(X_test)
#         c_index = concordance_index(y_test, predicted_survival)
#         scores.append(c_index)
#     return sum(scores) / len(scores)

#
# def objective(alpha, l1_ratio):
#     # Fit the Cox proportional hazards model using the selected features and specified hyperparameters
#     cph = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
#     kf = KFold(n_splits=5, shuffle=True)
#     # cph.fit(train, duration_col='last_visit', event_col='last_DX')
#
#     # Calculate the negative log-likelihood as the metric to optimize
#     log_likelihoods = []
#     for train_index, test_index in kf.split(train):
#         # X_train, X_test = train_cov.iloc[train_index], train_cov.iloc[test_index]
#         # y_train, y_test = y[train_index], y[test_index]
#         X_train, X_test = train_cov.iloc[train_index].values, train_cov.iloc[test_index].values
#         # y_train, y_test = y[train_index].values, y[test_index].values
#         y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()
#         cph.fit(X_train, y_train, duration_col='last_visit', event_col='last_DX')
#
#         # Compute survival table for test data
#         test_survival_table = survival_table_from_events(y_test, X_test)
#
#         # Calculate negative log-likelihood
#         neg_log_likelihood = -cph._log_likelihood(test_survival_table)
#         log_likelihoods.append(neg_log_likelihood)
#     return np.mean(log_likelihoods)


# Define the search space for hyperparameters
# space = {
#     'alpha': hp.loguniform('alpha', -5, 5),
#     'l1_ratio': hp.uniform('l1_ratio', 0, 1)
# }
#
# space = {
#     'alpha': [0., 1.],
#     'l1_ratio': [0, 1]
# }


np.random.seed(1234)

codes = {"CN_MCI": 0, "Dementia": 1}

dat = pd.read_csv("data/mci_lipids.csv")
dat.drop(dat.columns[0], axis=1, inplace=True)
dat.drop(dat.columns[0], axis=1, inplace=True)
dat["last_DX"].replace(codes, inplace=True)

# num_durations = 60
# labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)

best_params = []
best_c_index = []

# define 5-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

c_indices = []

model = sklearn_adapter(CoxPHFitter, event_col="last_DX")
EL = model()

for i in range(10):

    # split off test set
    dat_train, dat_test = train_test_split(dat, test_size=0.2, random_state=1234)

    get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)

    best_params = []
    best_c_index = []

    # define 10-fold cross validation test harness
    # kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

    # Preprocessing steps
    col_standardize = [
        col for col in dat_train.columns if col not in ["last_visit", "last_DX"]
    ]
    col_leave = ["last_DX", "last_visit"]
    standardize = [([col], StandardScaler()) for col in col_standardize]
    col_knn = [([col], KNNImputer(n_neighbors=5)) for col in col_standardize]
    leave = [(col, None) for col in col_leave]
    relief = ReliefF(n_neighbors=50, n_features_to_select=100)
    # col_relief = [([col], relief) for col in col_standardize]

    x_mapper = DataFrameMapper(leave + standardize)
    knn_mapper = DataFrameMapper(leave + col_knn)
    # relief_mapper = DataFrameMapper(leave + col_relief)

    # Fit preprocessing steps on train set and transform both train and test sets
    X_train_scale = pd.DataFrame(
        x_mapper.fit_transform(dat_train).astype("float32"), columns=dat_train.columns
    )
    # X_train = knn_mapper.fit_transform(X_train_scale).astype('float32')
    X_train = pd.DataFrame(
        knn_mapper.fit_transform(X_train_scale).astype("float32"),
        columns=dat_train.columns,
    )

    Y_train = dat_train.iloc[:, :2].to_numpy()
    Y_test = dat_test.iloc[:, :2].to_numpy()

    # X_train = relief_mapper.fit_transform(X_train, X_train['last_DX']).astype('float32')

    X_test = pd.DataFrame(
        x_mapper.transform(dat_test).astype("float32"), columns=dat_test.columns
    )
    X_test = knn_mapper.transform(X_test).astype("float32")

    X_test_relief = X_test[:, 2:]

    # Y_train = dat_train.iloc[:, :2].to_numpy()
    # Y_test = dat_test.iloc[:, :2].to_numpy()

    # reliefF = relief.fit(X_train_relief, Y_train[:,0])
    #
    # reliefF.top_features_

    # Y_train = pd.DataFrame(Y_train, columns=['last_DX', 'last_visit'])
    # Y_test = pd.DataFrame(Y_test, columns=['last_DX', 'last_visit'])

    X_train_relief = X_train.iloc[:, 2:].values

    X_train_relief = relief.fit_transform(X_train_relief, Y_train[:, 0]).astype(
        "float32"
    )

    # extract the top 100 features
    names = X_train.columns[2:]

    name = []
    for i in range(100):
        name.append(names[relief.top_features_[i]])

    Y_train = pd.DataFrame(Y_train, columns=["last_DX", "last_visit"])
    Y_test = pd.DataFrame(Y_test, columns=["last_DX", "last_visit"])

    # create a dataframe with the top 100 features
    X_train_relief = pd.DataFrame(X_train_relief, columns=name)
    train_df = pd.concat([Y_train, X_train_relief], axis=1)

    X_test_relief = relief.transform(X_test_relief)
    X_test_relief = pd.DataFrame(X_test_relief, columns=name)
    test_df = pd.concat([Y_test, X_test_relief], axis=1)
    # change to dataframe
    # X_test = pd.DataFrame(X_test)

    def objective(trial):
        # define the hyperparameters to tune
        params = {
            "penalizer": [trial.suggest_float("penalizer", 1, 4)],
            "l1_ratio": [trial.suggest_float("l1_ratio", 0.0, 1.0)],
        }

        clf = GridSearchCV(EL, params, cv=10)

        clf.fit(train_df, Y_train["last_visit"])
        # fetch the best parameters
        # best_params.append(clf.best_score_)
        # return the negative log-likelihood
        cv_results = clf.best_score_
        return -cv_results

    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(objective, n_trials=100, n_jobs=-1)
    best_params = study.best_params

    # Fit the model to the entire training set with the best parameters for survival xgboost
    model = EL.set_params(**best_params)
    model.fit(train_df, Y_train["last_visit"])
    surv_el = model.predict(test_df)

    # Get the concordance index for the test set
    c_index = 1 - concordance_index(Y_test["last_DX"], surv_el, Y_test["last_visit"])
    c_indices.append(c_index)

pd.DataFrame(c_indices).to_csv("data/elastic_net_c_indices.csv", index=False)

# 0.77 test set
