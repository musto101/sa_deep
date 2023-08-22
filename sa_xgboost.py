import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import KNNImputer
import optuna
from sklearn.model_selection import KFold, train_test_split
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from lifelines.utils import concordance_index
import pandas as pd


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
get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)

for i in range(100):

    # split off test set
    dat_train, dat_test = train_test_split(dat, test_size=0.2, random_state=1234)
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

    x_mapper = DataFrameMapper(leave + standardize)
    knn_mapper = DataFrameMapper(leave + col_knn)

    # Fit preprocessing steps on train set and transform both train and test sets
    X_train = pd.DataFrame(
        x_mapper.fit_transform(dat_train).astype("float32"), columns=dat_train.columns
    )
    X_train = knn_mapper.fit_transform(X_train).astype("float32")

    X_train = X_train[:, 2:]
    X_test = pd.DataFrame(
        x_mapper.transform(dat_test).astype("float32"), columns=dat_test.columns
    )
    X_test = knn_mapper.transform(X_test).astype("float32")

    X_test = X_test[:, 2:]

    Y_train = get_target(dat_train)
    Y_test = get_target(dat_test)

    X_train = relief.fit_transform(X_train, Y_train[1])
    # apply relief to test set
    X_test = relief.transform(X_test)

    # define the objective function for XGBoost for Optuna
    def objective(trial):
        # define the hyperparameters to tune
        param = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "booster": "gbtree",
            "nthread": -1,
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "eta": trial.suggest_loguniform("eta", 1e-8, 1e-5),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-8, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.0, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.0, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-5),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1e-5),
        }

        # define the number of boosting rounds
        num_round = 50

        # define the number of folds for cross validation
        nfold = 10

        dtrain = xgb.DMatrix(X_train, label=Y_train[1])
        # fit the model
        cv_results = xgb.cv(param, dtrain, num_round, nfold=nfold, seed=1234)
        # fetch the best parameters
        # best_params.append(cv_results['test-cox-nloglik-mean'].iloc[-1])
        # return the negative log-likelihood
        return cv_results["test-cox-nloglik-mean"].iloc[-1]

    # Create the Optuna study
    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(objective, n_trials=100, n_jobs=-1)

    # Fetch the best parameters
    best_params = study.best_params

    # Fit the model to the entire training set with the best parameters for survival xgboost
    dtrain = xgb.DMatrix(X_train, label=Y_train[1], weight=Y_train[0])
    dtest = xgb.DMatrix(X_test, label=Y_test[1], weight=Y_test[0])
    model = xgb.train(best_params, dtrain, num_boost_round=1000, verbose_eval=100)

    # Predict on test set
    surv_xgb = model.predict(dtest)

    # Get the concordance index for the test set
    c_index = concordance_index(Y_test[1], surv_xgb, Y_test[0])
    c_indices.append(c_index)

pd.DataFrame(c_indices).to_csv("data/xgb_c_indices.csv", index=False)

# 0.81 c-index test set

# import xgboost as xgb
# import pandas as pd
# import numpy as np
# from bayes_opt import BayesianOptimization
# from lifelines.utils import concordance_index
# # For preprocessing
# from sklearn.preprocessing import StandardScaler
# from sklearn_pandas import DataFrameMapper
# from sklearn.impute import KNNImputer
# import optuna
# from sklearn.model_selection import KFold
# np.random.seed(1234)
#
# train = pd.read_csv('data/dat_train.csv')
# train['time'] = train['time'].round(0)
#
# test = pd.read_csv('data/dat_test.csv')
# test['time'] = test['time'].round(0)
#
# train = train.dropna()
# test = test.dropna()
#
# train['time'] = train['time'].astype(int)
# test['time'] = test['time'].astype(int)
#
# X_train = train.drop(['dem_hse_w8', 'time'], axis=1)
# y_train = train[['dem_hse_w8']]
# event_train = train[['time']]
#
# X_test = test.drop(['dem_hse_w8', 'time'], axis=1)
# y_test = test[['dem_hse_w8']]
# event_test = test[['time']]
# # Convert the data into DMatrix format for XGBoost
# dtrain = xgb.DMatrix(X_train, label=y_train, weight=event_train)
# dtest = xgb.DMatrix(X_test)
#
# # Set the XGBoost parameters
# def xgb_cox_hyperopt(eta, max_depth, subsample, colsample_bytree):
#     params = {
#         'objective': 'survival:cox',
#         'eval_metric': 'cox-nloglik',
#         'booster': 'gbtree',
#         'eta': max(min(eta, 1), 0),
#         'max_depth': int(max_depth),
#         'subsample': max(min(subsample, 1), 0),
#         'colsample_bytree': max(min(colsample_bytree, 1), 0),
#         'gpu_id': 0,  # Specify the GPU ID to use
#         'tree_method': 'gpu_hist',  # Use GPU histogram-based method
#     }
#
#     dtrain = xgb.DMatrix(X_train, label=y_train, weight=event_train)
#
#     cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, stratified=False, seed=42)
#     return -cv_results['test-cox-nloglik-mean'].iloc[-1]
#
#
# # Define the hyperparameter search space
# pbounds = {'eta': (0.01, 0.3),
#            'max_depth': (3, 10),
#            'subsample': (0.5, 1),
#            'colsample_bytree': (0.5, 1)}
#
# # Create the Bayesian optimization object
# optimizer = BayesianOptimization(f=xgb_cox_hyperopt, pbounds=pbounds, random_state=42)
#
# # Perform the hyperparameter search
# optimizer.maximize(init_points=5, n_iter=20)
#
# # Print the best hyperparameters and the corresponding objective value
# best_params = optimizer.max['params']
# best_objective = optimizer.max['target']
# print("Best Hyperparameters:")
# print(best_params)
# print("Best Objective Value: {:.4f}".format(best_objective))
#
# # Set the XGBoost parameters
# # params = {
# #     'objective': 'survival:cox',
# #     'eval_metric': 'cox-nloglik',
# #     'booster': 'gbtree',
# #     'eta': 0.1,
# #     'max_depth': 3,
# #     'subsample': 0.8,
# #     'colsample_bytree': 0.8
# # }
#
# # Train the Cox survival-based XGBoost model
# model = xgb.train(best_params, dtrain)
#
# # Make predictions on the test set
# predictions = model.predict(dtest)
#
# # Evaluate the model using concordance index
# c_index = concordance_index(y_test, -predictions, event_test)
# print("Concordance Index: {:.4f}".format(c_index))
