import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline
from lifelines.utils.sklearn_adapter import sklearn_adapter


def remove_one_sum_columns(df):
    # Calculate the column sums
    column_sums = df.sum(axis=0)

    # Identify columns with a sum of zero
    one_sum_columns = column_sums[column_sums <= 1].index

    # Remove the zero-sum columns
    df = df.drop(one_sum_columns, axis=1)

    return df

get_target = lambda df: (df['last_visit'].values, df['last_DX'].values)


# Load your dataset into a Pandas DataFrame
# The dataset should include columns for time-to-event, event indicator, and covariates
data = pd.read_csv('data/mci_lipids.csv')
codes = {'CN_MCI': 0, 'Dementia': 1}

# drop first two columns
data = data.drop(['Unnamed: 0', 'X'], axis=1)
#data.columns

data['last_DX'].replace(codes, inplace=True)
# split the data into train and test
train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)

# split train and test into X and y
train_X = train.drop(['last_visit', 'last_DX'], axis=1)
#train.columns
train_y = get_target(train)

test_X = test.drop(['last_visit', 'last_DX'], axis=1)
test_y = get_target(test)

# Create a new instance of CoxPHFitter
cph = CoxPHFitter()


# create the preprocessing pipeline
preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=5)),
                            ('scaler', StandardScaler())])

preprocessing.fit(train_X)

train_X = preprocessing.transform(train_X)
test_X = preprocessing.transform(test_X)

# run the ReliefF algorithm

fs = ReliefF(n_neighbors=50, n_features_to_select=100)

train_X = fs.fit_transform(train_X, train_y[1])

test_X = fs.transform(test_X)

# convert the train data to a dataframe
train_X = pd.DataFrame(train_X)
train_X['last_visit'] = train_y[0]
train_X['last_DX'] = train_y[1]

# Fit the Cox proportional hazards model
cph.fit(train_X, duration_col='last_visit', event_col='last_DX', show_progress=True)

# Print the summary of the model
cph.print_summary()

# Access the estimated coefficients and hazard ratios
coefficients = cph.params_
hazard_ratios = np.exp(coefficients)

# Print the estimated coefficients and hazard ratios
print("Estimated Coefficients:")
print(coefficients)
print("Hazard Ratios:")
print(hazard_ratios)

# calculate the concordance index for the training data
c_index = concordance_index(train_X['last_DX'], cph.predict_partial_hazard(train_X)) # train c-index is 0.82

# calculate the concordance index for the test data
# test_data = remove_one_sum_columns(test_data)
# drop columns that are not in the training data

# convert the train data to a dataframe
test_X = pd.DataFrame(test_X)
test_X['last_visit'] = test_X[0]
test_X['last_DX'] = test_X[1]

c_index_test = concordance_index(test_X['last_DX'], cph.predict_partial_hazard(test_X))

# c_index for test is 0.77 and for train is 0.82 so thats the one to beat
