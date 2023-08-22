import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from skrebate import ReliefF

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch  # For building the networks
import torchtuples as tt  # Some useful functions
from sklearn.model_selection import KFold

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import pandas as pd
import optuna

np.random.seed(1234)
_ = torch.manual_seed(123)

best_params = []
best_loss = []

kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

df = pd.read_csv("data/mci_lipids.csv")
# df_train.dropna(inplace=True)

codes = {"CN_MCI": 0, "Dementia": 1}
df["last_DX"].replace(codes, inplace=True)

df_test = df.sample(frac=0.2)
df = df.drop(df_test.index)
# df_val_test = df.sample(frac=0.2)
# df = df.drop(df_val_test.index)


for i, (train_index, test_index) in enumerate(kfold.split(df)):
    print(f"Fold {i}:")
    df_train, df_val = (
        df.iloc[train_index],
        df.iloc[test_index],
    )

    df_val_test = df_train.sample(frac=0.1)
    df_train = df_train.drop(df_val_test.index)

    cols_standardize = df_train.columns[3:]
    col_leave = df_train.columns[:3]

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in col_leave]

    col_knn = [([col], KNNImputer(n_neighbors=5)) for col in cols_standardize]
    relief = ReliefF(n_neighbors=50, n_features_to_select=100)

    x_mapper = DataFrameMapper(standardize + leave)
    knn_mapper = DataFrameMapper(leave + col_knn)

    x_train = pd.DataFrame(
        x_mapper.fit_transform(df_train), columns=df_train.columns
    ).astype("float32")
    x_val = pd.DataFrame(x_mapper.transform(df_val), columns=df_val.columns).astype(
        "float32"
    )
    # x_test = pd.DataFrame(x_mapper.transform(df_test), columns=df_test.columns).astype(
    #     "float32"
    # )
    x_val_test = pd.DataFrame(
        x_mapper.transform(df_val_test), columns=df_val_test.columns
    ).astype("float32")

    x_train = knn_mapper.fit_transform(x_train).astype("float32")
    x_val = knn_mapper.transform(x_val).astype("float32")
    # x_test = knn_mapper.transform(x_test).astype("float32")
    x_val_test = knn_mapper.transform(x_val_test).astype("float32")

    num_durations = 60
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    y_val_test = labtrans.transform(*get_target(df_val_test))

    x_train = relief.fit_transform(x_train[:, 2:], y_train[1]).astype("float32")
    x_val = relief.transform(x_val[:, 2:]).astype("float32")
    # x_test = relief.transform(x_test[:, 2:]).astype("float32")
    x_val_test = relief.transform(x_val_test[:, 2:]).astype("float32")

    train = (x_train, y_train)
    val = (x_val, y_val)

    # We don't need to transform the test labels
    # durations_test, events_test = get_target(df_test)

    def objective(trial):
        # Define the hyperparameters to tune

        in_features = x_train.shape[1]
        # suggest number of nodes and layers

        num_nodes = [
            trial.suggest_int("n_units_l{}".format(i), 2, 200)
            for i in range(trial.suggest_int("n_layers", 1, 100))
        ]
        # num_nodes = trial.suggest_int('num_nodes', 2, 256)
        out_features = labtrans.out_features

        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        dropout = trial.suggest_float("dropout", 0.4, 0.9)
        dropout = np.float32(dropout)
        # activation = trial.suggest_categorical(
        #     "activation",
        #     [
        #         "ReLU",
        #         "LeakyReLU",
        #         "PReLU",
        #         "tanh",
        #         "SELU",
        #         "ELU",
        #         "CELU",
        #         "GLU",
        #         "hardshrink",
        #         "LogSigmoid",
        #         "Softplus",
        #     ],
        # )
        net = tt.practical.MLPVanilla(
            in_features, num_nodes, out_features, batch_norm, dropout
        )

        sigma = trial.suggest_float("sigma", 0.5, 1)
        alpha = trial.suggest_float("alpha", 0.5, 1)

        # sigma = np.float32(sigma)
        # alpha = np.float32(alpha)

        learning_rate = trial.suggest_loguniform("learning_rate", 1e-10, 1e-5)
        batch_size = 256
        epochs = trial.suggest_int("epochs", 400, 800)
        callbacks = [
            tt.callbacks.EarlyStopping(metric="loss", patience=50, min_delta=0.0001)
        ]

        print("in_features:", in_features)
        print("num_nodes:", num_nodes)
        print("out_features:", out_features)
        print("batch_norm:", batch_norm)
        print("dropout:", dropout)

        # Create and train the DeepHit model with the given hyperparameters
        model = DeepHitSingle(
            net, tt.optim.Adam, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts
        )
        model.optimizer.set_lr(learning_rate)
        model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
        # x_train_relief = np.array(x_train_relief)

        # Evaluate the model on the validation set
        surv = model.predict_surv_df(x_val_test)
        eval_surv = EvalSurv(surv, y_val_test[0], y_val_test[1], censor_surv="km")
        time_grid = np.linspace(y_val_test[0].min(), y_val_test[0].max(), 100)
        loglike = eval_surv.integrated_nbll(time_grid)

        # Return the loss as the objective value
        return loglike

    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(objective, n_trials=100, n_jobs=-1)  # 50 trials was 0.68 on test set

    # Print optimization results
    print("Number of finished trials:", len(study.trials))
    print("Best trial parameters:", study.best_trial.params)
    print("Best score:", study.best_value)

    best_params.append(study.best_params)
    best_loss.append(abs(study.best_value))
    deephit = pd.DataFrame(best_params)
    deephit["loss"] = best_loss
    deephit.to_csv("data/deephit_optuna.csv")


# run on the whole dataset
df_val = df.sample(frac=0.2)
df = df.drop(df_val.index)

cols_standardize = df.columns[3:]
col_leave = df.columns[:3]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in col_leave]

col_knn = [([col], KNNImputer(n_neighbors=5)) for col in cols_standardize]
relief = ReliefF(n_neighbors=50, n_features_to_select=100)

x_mapper = DataFrameMapper(standardize + leave)
knn_mapper = DataFrameMapper(leave + col_knn)

x_train = pd.DataFrame(x_mapper.fit_transform(df), columns=df.columns).astype("float32")
x_val = pd.DataFrame(x_mapper.transform(df_val), columns=df_val.columns).astype(
    "float32"
)
x_test = pd.DataFrame(x_mapper.transform(df_test), columns=df_test.columns).astype(
    "float32"
)

x_train = knn_mapper.fit_transform(x_train).astype("float32")
x_val = knn_mapper.transform(x_val).astype("float32")
x_test = knn_mapper.transform(x_test).astype("float32")

num_durations = 60
labtrans = DeepHitSingle.label_transform(num_durations)
get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)
y_train = labtrans.fit_transform(*get_target(df))
y_val = labtrans.transform(*get_target(df_val))

x_train = relief.fit_transform(x_train[:, 2:], y_train[1]).astype("float32")
x_val = relief.transform(x_val[:, 2:]).astype("float32")
x_test = relief.transform(x_test[:, 2:]).astype("float32")

val = (x_val, y_val)
durations_test, events_test = get_target(df_test)

params = pd.read_csv("data/deephit_optuna.csv")

# find the best params using a marjority vote
params["n_params"] = params["n_layers"] * params["n_units_l1"]
params["weighted"] = params["loss"] * params["n_params"]

best_params = params.sort_values("weighted", ascending=False).iloc[0].to_dict()
# drop c_index value
best_params.pop("loss")
# drop weighted value
best_params.pop("weighted")
# drop n_params value
best_params.pop("n_params")
# drop index value
best_params.pop("Unnamed: 0")

# drop values with nan
best_params = {k: v for k, v in best_params.items() if v == v}


in_features = x_train.shape[1]
num_nodes = [
    best_params["n_units_l{}".format(i)] for i in range(best_params["n_layers"])
]
# convert to int
num_nodes = [int(i) for i in num_nodes]
out_features = labtrans.out_features
batch_norm = best_params["batch_norm"]
dropout = best_params["dropout"]
learning_rate = best_params["learning_rate"]
epochs = best_params["epochs"]

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
model = DeepHitSingle(net, tt.optim.Adam, duration_index=labtrans.cuts)
batch_size = 256

model.optimizer.set_lr(lr=learning_rate)

callbacks = [tt.callbacks.EarlyStopping(metric="loss", patience=50)]
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)

# _ = log.plot()

# Evaluate the model on the train set
surv = model.predict_surv_df(x_train)
eval_surv = EvalSurv(surv, y_train[1], y_train[0], censor_surv="km")
time_grid = np.linspace(y_train[0].min(), y_train[0].max(), 100)
c_index_train = eval_surv.concordance_td()
print("C-index train: {:.4f}".format(c_index_train))

# Evaluate the model on the test set
surv = model.predict_surv_df(x_test)
eval_surv = EvalSurv(surv, durations_test, events_test, censor_surv="km")
c_index_test = eval_surv.concordance_td()
print("C-index test: {:.4f}".format(c_index_test))
