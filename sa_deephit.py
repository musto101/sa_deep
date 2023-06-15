from bayes_opt import BayesianOptimization
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
import numpy as np

if __name__ == '__main__':
    train_features = np.load('data/train_features.npy')
    val_features = np.load('data/valid_features.npy')
    test_features = np.load('data/test_features.npy')
    features = [train_features, val_features, test_features]

    train_labels = np.load('data/train_labels.npy').astype(np.int32)
    val_labels = np.load('data/valid_labels.npy').astype(np.int32)
    test_labels = np.load('data/test_labels.npy').astype(np.int32)
    labels = [train_labels, val_labels, test_labels]


# Define the objective function for hyperparameter tuning
def deephit_hyperopt(alpha, sigma_1, sigma_2, num_nodes_shared, num_nodes_indiv):
    num_nodes = [int(num_nodes_shared)] * 3 + [int(num_nodes_indiv)] * 3
    alpha = np.exp(alpha)
    sigma_1 = np.exp(sigma_1)
    sigma_2 = np.exp(sigma_2)

    model = DeepHit(num_durations=5, alpha=alpha, sigma_1=sigma_1, sigma_2=sigma_2, num_nodes=num_nodes)
    model.fit(train_features, train_labels[1], train_labels[0], batch_size=256, epochs=20, verbose=False)

    surv = model.predict_surv_df(X)
    ev = EvalSurv(surv, T, E, censor_surv='km')
    c_index = ev.concordance_td()
    return c_index

# Define the hyperparameter search space
pbounds = {'alpha': (-5, 5),
           'sigma_1': (-5, 5),
           'sigma_2': (-5, 5),
           'num_nodes_shared': (32, 256),
           'num_nodes_indiv': (32, 256)}

# Create the Bayesian optimization object
optimizer = BayesianOptimization(f=deephit_hyperopt, pbounds=pbounds, random_state=42)

# Perform the hyperparameter search
optimizer.maximize(init_points=5, n_iter=20)

# Print the best hyperparameters and the corresponding objective value
best_params = optimizer.max['params']
best_objective = optimizer.max['target']
print("Best Hyperparameters:")
print(best_params)
print("Best Objective Value: {:.4f}".format(best_objective))
