import numpy as np
import xgboost as xgb
import optuna
from concordance import concordance_index
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
from sklearn.model_selection import KFold

get_target = lambda df: (df['last_visit'].values, df['last_DX'].values)

np.random.seed(1234)
data = pd.read_csv('data/mci_preprocessed_wo_csf2.csv')

codes = {'CN_MCI': 0, 'Dementia': 1}

# drop first column
data = data.drop(['Unnamed: 0'], axis=1)

# drop DX columns
data = data.drop(['DX.CN'], axis=1)
data = data.drop(['DX.MCI'], axis=1)
data = data.drop(['DX.AD'], axis=1)
# data.columns

data['last_DX'].replace(codes, inplace=True)
# def callback(study, trial):
#     if study.best_trial.number == trial.number:
#         study.set_user_attr(key="best_transformer", value=trial.user_attrs["best_transformer"])


def objective(trial):
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "verbosity": 0,
        "booster": "gbtree",
        "nthread": -1,
        "seed": 1234,
        "eta": trial.suggest_float("eta", 1e-5, 0.01),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        # 'num_boost_round': trial.suggest_int('num_boost_round', 10, 1000),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.2),
        'gamma': trial.suggest_float('gamma', 0, 0.001),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 0.0001),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 20),
    }

    print(params)

    model = xgb.train(params, dtrain, evals=[(dval, "eval")], num_boost_round=1000,
                      early_stopping_rounds=1000)

    trial.set_user_attr(key="best_xgb", value=model)

    # Predict on test set
    surv_xgb = model.predict(dval)

    # predict on the training set
    surv_xgb_train = model.predict(dtrain)

    # remove any nan and inf from the prediction
    # surv_xgb = np.nan_to_num(surv_xgb, nan=-1, posinf=1, neginf=1)
    #
    # # check prediction for nan
    # print('number of nan in prediction', np.isnan(surv_xgb).sum())
    # # drop any nan in the prediction
    # # surv_xgb = surv_xgb[~np.isnan(surv_xgb)]
    #
    # # check number of nan in test_y
    # print('number of nan in val_y', val_y['1'].isnull().sum())
    # print('number of nan in val_y', val_y['1'].isnull().sum())

    # get the concordance index for the training set
    c_index_train = concordance_index(train_y['last_visit'], surv_xgb_train, train_y['last_DX'])

    # Get the concordance index for the test set
    c_index = concordance_index(dval.get_weight(), surv_xgb, dval.get_label())

    print("Train Concordance Index:", c_index_train)

    return c_index

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_xgb", value=trial.user_attrs["best_xgb"])


val_score = []
test_score = []

mc_test = []


# create folds for cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

for i in range(100):
    print(i)
    for train_index, test_index in kf.split(data):
        training = data.iloc[train_index]
        # print(train.head())
        test = data.iloc[test_index]
        # print(test.head())

        # split train into train and validation
        val = training.sample(frac=0.2)
        # print(val.head())
        training = training.drop(val.index)

        # split train and test into X and y
        train_X = training.drop(['last_visit', 'last_DX'], axis=1)
        # print('train_x', train_X.head())
        train_y = get_target(training)
        # print('train_y', train_y)

        val_X = val.drop(['last_visit', 'last_DX'], axis=1)
        val_y = get_target(val)

        test_X = test.drop(['last_visit', 'last_DX'], axis=1)
        test_y = get_target(test)
        # print('test_y', test_y)

        # create the preprocessing pipeline
        preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=5)),
                                  ('scaler', StandardScaler())])

        # pipeline = ImbPipeline(steps=[
        # ('imputer', KNNImputer(n_neighbors=5)),
        # ('scaler', StandardScaler()),
        # ('smote', SMOTE(random_state=42))
        # # Add your classifier here, for example: ('clf', RandomForestClassifier())
        # ])

        preprocessing.fit(train_X)

        # train_X, train_y = pipeline.fit_resample(train_X, train_y)
        # val_X = pipeline.fit_resample(val_X, val_y[1])
        # test_X = pipeline.fit_resample(test_X, test_y[1])

        train_X = preprocessing.transform(train_X)
        # print('train_X', train_X)
        val_X = preprocessing.transform(val_X)
        test_X = preprocessing.transform(test_X)

        # run the ReliefF algorithm

        fs = ReliefF(n_neighbors=50, n_features_to_select=200)

        train_X = fs.fit_transform(train_X, train_y[0])

        # find the top 200 features and their reliefF scores ordered from highest to lowest
        # selected_scores = fs.feature_importances_
        # selected_features = fs.top_features_
        #
        # # concatenate the scores and features into a single array and sort them by score descending
        # selected = np.column_stack((selected_features, selected_scores))
        # selected = selected[selected[:, 1].argsort()[::-1]]
        #
        # # extract the top 200 feature indices
        # top_feature_indices = selected[:200, 0].astype(int)
        #
        # # filter the column names of the original dataframe
        # selected = data.columns[top_feature_indices]

        # print('train_X', train_X)
        val_X = fs.transform(val_X)
        test_X = fs.transform(test_X)

        # convert the train data to a dataframe
        # train_X = pd.DataFrame(train_X, columns=selected)
        train_X = pd.DataFrame(train_X)

        # add the last_visit and last_DX columns to the dataframe
        # train_X['last_visit'] = train_y[0]
        # train_X['last_DX'] = train_y[1]

        # val_X = pd.DataFrame(val_X, columns=selected)
        val_X = pd.DataFrame(val_X)
        # val_X['last_visit'] = val_y[0]
        # val_X['last_DX'] = val_y[1]

        # test_X = pd.DataFrame(test_X, columns=selected)
        test_X = pd.DataFrame(test_X)
        # test_X['last_visit'] = test_y[0]
        # test_X['last_DX'] = test_y[1]

        print(train_X.isnull().sum().sum())
        print(val_X.isnull().sum().sum())
        print(test_X.isnull().sum().sum())

        train_y = pd.DataFrame(train_y).T
        #order the rows of the dataframe by last_DX
        train_y = train_y.sort_values(by=1)
        train_y = train_y.rename(columns={0: 'last_visit', 1: 'last_DX'})
        val_y = pd.DataFrame(val_y).T
        val_y = val_y.rename(columns={0: 'last_visit', 1: 'last_DX'})
        test_y = pd.DataFrame(test_y).T
        test_y = test_y.rename(columns={0: 'last_visit', 1: 'last_DX'})


        # Convert the data into DMatrix format for XGBoost
        dtrain = xgb.DMatrix(train_X, label=train_y['last_DX'], weight=train_y['last_visit'])
        dval = xgb.DMatrix(val_X, label=val_y['last_DX'], weight=val_y['last_visit'])
        dtest = xgb.DMatrix(test_X, label=test_y['last_DX'], weight=test_y['last_visit'])

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

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, n_jobs=-1, callbacks=[callback])

        best_model = study.user_attrs["best_xgb"]

        # save the best model



    # Fetch the best parameters
        best_params = study.best_params

    # print the best score
    #     print("Best score:", study.best_value)

    # use best model to predict on test set
        surv_xgb = best_model.predict(dtest)

    # remove any nan and inf from the prediction
        surv_xgb = np.nan_to_num(surv_xgb, nan=-1, posinf=1, neginf=1)

    # calculate the concordance index for the test set
        c_index = concordance_index(test_y['last_visit'], surv_xgb, test_y['last_DX'])


        # print("Test Concordance Index:", c_index)

        test_score.append(c_index)
        # save test concordance indices to a csv
        # pd.DataFrame(test_score).to_csv("data/test_c_indices_xgb.csv", index=False)
        # val_score.append(study.best_value)
        # save validation concordance indices to a csv
        # pd.DataFrame(val_score).to_csv("data/val_c_indices_xgb.csv", index=False)

    # get the absolute value of the test concordance indices
    test_c = np.abs(test_score)

    mean_test = np.mean(test_c)

    mc_test.append(mean_test)

    joblib.dump(best_model, 'data/best_xgb_model.pkl')
    joblib.dump(best_model, 'data/best_xgb_model.joblib')

    #summary of the test concordance indices
    print('median test', np.median(test_c))
    print('mean test', np.mean(test_c))
    print('std test', np.std(test_c))

    # save the test concordance indices to a csv
    pd.DataFrame(mc_test).to_csv("data/test_c_indices_xgb2.csv", index=False)

    # get the absolute value of the validation concordance indices
    # val_c = np.abs(val_score)
    #
    # # save the validation concordance indices to a csv
    # pd.DataFrame(val_c).to_csv("data/val_c_indices_xgb.csv", index=False)

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
