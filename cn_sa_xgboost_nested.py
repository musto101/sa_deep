import numpy as np
import xgboost as xgb
import optuna
from concordance import concordance_index
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from sklearn.pipeline import Pipeline


get_target = lambda df: (df['last_visit'].values, df['last_DX'].values)

np.random.seed(1234)
data = pd.read_csv('data/cn_preprocessed_wo_csf.csv')

codes = {'CN': 0, 'MCI_AD': 1}

# drop first column
data = data.drop(['Unnamed: 0'], axis=1)

# drop DX columns
data = data.drop(['DX.CN'], axis=1)
data = data.drop(['DX.MCI'], axis=1)
data = data.drop(['DX.AD'], axis=1)
# data.columns

data['last_DX'].replace(codes, inplace=True)


# create an empty list to store the test concordance indices
preds = []

# create folds for cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=1234)



for i in kf.split(data):
    # print(i)
    train = data.iloc[i[0]]
    # print(train.head())
    test = data.iloc[i[1]]
    # print(test.head())

    # split train into train and validation
    val = train.sample(frac=0.2, random_state=0)
    # print(val.head())
    train = train.drop(val.index)

    # split train and test into X and y
    train_X = train.drop(['last_visit', 'last_DX'], axis=1)
    # print('train_x', train_X.head())
    train_y = get_target(train)
    # print('train_y', train_y)

    val_X = val.drop(['last_visit', 'last_DX'], axis=1)
    val_y = get_target(val)

    test_X = test.drop(['last_visit', 'last_DX'], axis=1)
    test_y = get_target(test)
    # print('test_y', test_y)

    # create the preprocessing pipeline
    preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=5)),
                          ('scaler', StandardScaler())])

    preprocessing.fit(train_X)

    train_X = preprocessing.transform(train_X)
    # print('train_X', train_X)
    val_X = preprocessing.transform(val_X)
    test_X = preprocessing.transform(test_X)

    # run the ReliefF algorithm

    fs = ReliefF(n_neighbors=50, n_features_to_select=100)

    train_X = fs.fit_transform(train_X, train_y[1])
    # print('train_X', train_X)
    val_X = fs.transform(val_X)
    test_X = fs.transform(test_X)

    # convert the train data to a dataframe
    train_X = pd.DataFrame(train_X)

    # add the last_visit and last_DX columns to the dataframe
    train_X['last_visit'] = train_y[0]
    train_X['last_DX'] = train_y[1]

    val_X = pd.DataFrame(val_X)
    val_X['last_visit'] = val_y[0]
    val_X['last_DX'] = val_y[1]

    test_X = pd.DataFrame(test_X)
    test_X['last_visit'] = test_y[0]
    test_X['last_DX'] = test_y[1]

    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(train_X.drop(['last_visit', 'last_DX'], axis=1), label=train_y[1], weight=train_y[0])
    dval = xgb.DMatrix(val_X.drop(['last_visit', 'last_DX'], axis=1), label=val_y[1], weight=val_y[0])
    dtest = xgb.DMatrix(test_X.drop(['last_visit', 'last_DX'], axis=1), label=test_y[1], weight=test_y[0])

# run the base model
# params = {
#     "objective": "survival:cox",
#     "eval_metric": "cox-nloglik",
#     "verbosity": 0,
#     "booster": "gbtree",
#     "nthread": 4,
#     "seed": 1234,
# }
#
# model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, "eval")], verbose_eval=100)
#
# # Predict on test set
# surv_xgb = model.predict(dtest)
#
# # Get the concordance index for the test set
# c_index = 1 - concordance_index(test_y['1'], surv_xgb, test_y['0'])

# run bayesian optimization

# fake params
# params = {
#     "objective": "survival:cox",
#     "eval_metric": "cox-nloglik",
#     "verbosity": 0,
#     "booster": "gbtree",
#     "nthread": -1,
#     "seed": 1234,
#     "eta": 0.00005,
#     "max_depth": 2,
#     "subsample": 0.2,
#     "colsample_bytree": 0.2,
#     'gamma': 0,
#     'min_child_weight': 0,
#     'reg_alpha': 0,
#     'reg_lambda': 16,
# }

    def objective(trial):
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "verbosity": 0,
            "booster": "gbtree",
            "nthread": -1,
            "seed": 1234,
            "eta": trial.suggest_float("eta", 1e-8, 0.00005),
            "max_depth": trial.suggest_int("max_depth", 1, 2),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            # 'num_boost_round': trial.suggest_int('num_boost_round', 10, 1000),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.1, 0.5),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 0.0001),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 16, 20),
        }

        print(params)

        model = xgb.train(params, dtrain, evals=[(dval, "eval")], verbose_eval=100, num_boost_round=3000,
                          early_stopping_rounds=10)

        trial.set_user_attr(key="best_xgb", value=model)

        # Predict on test set
        surv_xgb = model.predict(dval)

        # remove any nan and inf from the prediction
        surv_xgb = np.nan_to_num(surv_xgb, nan=-1, posinf=1, neginf=1)

        # check prediction for nan
        # print('number of nan in prediction', np.isnan(surv_xgb).sum())
        # drop any nan in the prediction
        # surv_xgb = surv_xgb[~np.isnan(surv_xgb)]

        # check number of nan in test_y
        # print('number of nan in val_y', val_y[1].isnull().sum())
        # print('number of nan in val_y', val_y[1].isnull().sum())

        # Get the concordance index for the test set
        c_index = 1 - concordance_index(dval.get_weight(), surv_xgb, dval.get_label())

        return -c_index


    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_xgb", value=trial.user_attrs["best_xgb"])


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10000, n_jobs=-1, callbacks=[callback])

    best_model = study.user_attrs["best_xgb"]

    # Fetch the best parameters
    best_params = study.best_params

    # print the best score
    #     print("Best score:", study.best_value)

    # use best model to predict on test set
    surv_xgb = best_model.predict(dtest)

    # remove any nan and inf from the prediction
    surv_xgb = np.nan_to_num(surv_xgb, nan=-1, posinf=1, neginf=1)

    # store the predictions for the test set


    preds.append(surv_xgb)

    # print("Test Concordance Index:", c_index)

# extract the outcome for the full dataset
full_y = get_target(data)

# calculate the concordance index for the whole dataset
c_index = 1 - concordance_index(full_y[1], np.concatenate(preds), full_y[0])

print("Concordance Index:", c_index)
#
# # create params including the best parameters
# best_params = {
#     "objective": "survival:cox",
#     "eval_metric": "cox-nloglik",
#     "verbosity": 0,
#     "booster": "gbtree",
#     "nthread": -1,
#     "seed": 1234,
#     "eta": best_params['eta'],
#     "max_depth": best_params['max_depth'],
#     "subsample": best_params['subsample'],
#     "colsample_bytree": best_params['colsample_bytree'],
#     'gamma': best_params['gamma'],
#     'min_child_weight': best_params['min_child_weight'],
#     'reg_alpha': best_params['reg_alpha'],
#     'reg_lambda': best_params['reg_lambda'],
# }



#
# # Fit the model to the entire training set with the best parameters for survival xgboost
# model = xgb.train(best_params, dtrain, verbose_eval=100, num_boost_round=1000)
#
# # get the concordance index for the training set
# surv_xgb = model.predict(dtrain)
# c_index = 1- concordance_index(train_y[1], surv_xgb, train_y[0])
#
# print('Train Concordance Index:', c_index)
#
# # Predict on test set
# surv_xgb = model.predict(dtest)
#
# # Get the concordance index for the test set
# c_index = 1 - concordance_index(test_y['1'], surv_xgb, test_y['0'])
#
# print("Test Concordance Index:", c_index)
