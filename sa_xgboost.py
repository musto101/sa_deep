import xgboost as xgb
import pandas as pd
from bayes_opt import BayesianOptimization
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

train = pd.read_csv('data/dat_train.csv')
train['time'] = train['time'].round(0)

test = pd.read_csv('data/dat_test.csv')
test['time'] = test['time'].round(0)

train = train.dropna()
test = test.dropna()

train['time'] = train['time'].astype(int)
test['time'] = test['time'].astype(int)

X_train = train.drop(['dem_hse_w8', 'time'], axis=1)
y_train = train[['dem_hse_w8']]
event_train = train[['time']]

X_test = test.drop(['dem_hse_w8', 'time'], axis=1)
y_test = test[['dem_hse_w8']]
event_test = test[['time']]
# Convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, weight=event_train)
dtest = xgb.DMatrix(X_test)

# Set the XGBoost parameters
def xgb_cox_hyperopt(eta, max_depth, subsample, colsample_bytree):
    params = {
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'booster': 'gbtree',
        'eta': max(min(eta, 1), 0),
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'gpu_id': 0,  # Specify the GPU ID to use
        'tree_method': 'gpu_hist',  # Use GPU histogram-based method
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=event_train)

    cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, stratified=False, seed=42)
    return -cv_results['test-cox-nloglik-mean'].iloc[-1]


# Define the hyperparameter search space
pbounds = {'eta': (0.01, 0.3),
           'max_depth': (3, 10),
           'subsample': (0.5, 1),
           'colsample_bytree': (0.5, 1)}

# Create the Bayesian optimization object
optimizer = BayesianOptimization(f=xgb_cox_hyperopt, pbounds=pbounds, random_state=42)

# Perform the hyperparameter search
optimizer.maximize(init_points=5, n_iter=20)

# Print the best hyperparameters and the corresponding objective value
best_params = optimizer.max['params']
best_objective = optimizer.max['target']
print("Best Hyperparameters:")
print(best_params)
print("Best Objective Value: {:.4f}".format(best_objective))

# Set the XGBoost parameters
# params = {
#     'objective': 'survival:cox',
#     'eval_metric': 'cox-nloglik',
#     'booster': 'gbtree',
#     'eta': 0.1,
#     'max_depth': 3,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8
# }

# Train the Cox survival-based XGBoost model
model = xgb.train(best_params, dtrain)

# Make predictions on the test set
predictions = model.predict(dtest)

# Evaluate the model using concordance index
c_index = concordance_index(y_test, -predictions, event_test)
print("Concordance Index: {:.4f}".format(c_index))
