import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from concordance import concordance_index

np.random.seed(1234)

train_X = pd.read_csv('data/cn_train_features.csv')
train_y = pd.read_csv('data/cn_train_labels.csv')
val_X = pd.read_csv('data/cn_val_features.csv')
val_y = pd.read_csv('data/cn_val_labels.csv')
test_X = pd.read_csv('data/cn_test_features.csv')
test_y = pd.read_csv('data/cn_test_labels.csv')

# check data for missing values
print(train_X.isnull().sum().sum())
print(val_X.isnull().sum().sum())
print(test_X.isnull().sum().sum())

print(train_y.isnull().sum().sum())
print(val_y.isnull().sum().sum())
print(test_y.isnull().sum().sum())

# Convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(train_X, label=train_y['1'], weight=train_y['0'])
dval = xgb.DMatrix(val_X, label=val_y['1'], weight=val_y['0'])
dtest = xgb.DMatrix(test_X, label=test_y['1'], weight=test_y['0'])

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

def objective(trial):
    def objective(trial):
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "verbosity": 0,
            "booster": "gbtree",
            "nthread": -1,
            "seed": 1234,
            "eta": trial.suggest_float("eta", 1e-8, 0.05),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 0.5),
            # 'num_boost_round': trial.suggest_int('num_boost_round', 10, 1000),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
            'gamma': trial.suggest_float('gamma', 0, 0.01),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 0.001),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 10, 20),
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
    print('number of nan in prediction', np.isnan(surv_xgb).sum())
    # drop any nan in the prediction
    # surv_xgb = surv_xgb[~np.isnan(surv_xgb)]

    # check number of nan in test_y
    # print('number of nan in val_y', val_y['1'].isnull().sum())
    # print('number of nan in val_y', val_y['1'].isnull().sum())

    # Get the concordance index for the test set
    c_index = 1 - concordance_index(dval.get_weight(), surv_xgb, dval.get_label())

    return -c_index

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_xgb", value=trial.user_attrs["best_xgb"])

test_c = []


for i in range(1000):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=-1, callbacks=[callback])
    test_c.append(study.best_value)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=-1, callbacks=[callback])

    best_model = study.user_attrs["best_xgb"]

# Fetch the best parameters
    best_params = study.best_params

# print the best score
#     print("Best score:", study.best_value)

# use best model to predict on test set
    surv_xgb = best_model.predict(dtest)

# remove any nan and inf from the prediction
    surv_xgb = np.nan_to_num(surv_xgb, nan=-1, posinf=1, neginf=1)

# calculate the concordance index for the test set
    c_index = 1 - concordance_index(test_y['1'], surv_xgb, test_y['0'])

    # print("Test Concordance Index:", c_index)

    test_c.append(c_index)

# get the absolute value of the test concordance indices
test_c = np.abs(test_c)

#summary of the test concordance indices
print('median test', np.median(test_c))
print('mean test', np.mean(test_c))
print('std test', np.std(test_c))

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
