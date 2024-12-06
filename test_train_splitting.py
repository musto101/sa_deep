import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from sklearn.pipeline import Pipeline
import imblearn

get_target = lambda df: (df['last_visit'].values, df['last_DX'].values)

# Load your dataset into a Pandas DataFrame
# The dataset should include columns for time-to-event, event indicator, and covariates
data = pd.read_csv('data/mci_preprocessed_wo_csf.csv')
data.shape
codes = {'CN_MCI': 0, 'Dementia': 1}

# drop first column
data = data.drop(['Unnamed: 0'], axis=1)

# drop DX columns
data = data.drop(['DX.CN'], axis=1)
data = data.drop(['DX.MCI'], axis=1)
data = data.drop(['DX.AD'], axis=1)
# data.columns

data['last_DX'].replace(codes, inplace=True)

# # run smote for imbalanced data
# smote = imblearn.over_sampling.SMOTE()
#
# X, y = smote.fit_resample(data.drop(['last_visit', 'last_DX'], axis=1), data['last_DX'])

# split the data into train and test
train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)

# split train into train and validation
val = train.sample(frac=0.2, random_state=0)
train = train.drop(val.index)

# split train and test into X and y
train_X = train.drop(['last_visit', 'last_DX'], axis=1)
#train.columns
train_y = get_target(train)

val_X = val.drop(['last_visit', 'last_DX'], axis=1)
val_y = get_target(val)

test_X = test.drop(['last_visit', 'last_DX'], axis=1)
test_y = get_target(test)

# create the preprocessing pipeline
preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=5)),
                            ('scaler', StandardScaler())])

preprocessing.fit(train_X)

train_X = preprocessing.transform(train_X)
val_X = preprocessing.transform(val_X)
test_X = preprocessing.transform(test_X)

# run the ReliefF algorithm

fs = ReliefF(n_neighbors=50, n_features_to_select=100)

train_X = fs.fit_transform(train_X, train_y[1])
val_X = fs.transform(val_X)
test_X = fs.transform(test_X)

# convert the train data to a dataframe
train_X = pd.DataFrame(train_X)

# add the last_visit and last_DX columns to the dataframe
# train_X['last_visit'] = train_y[0]
# train_X['last_DX'] = train_y[1]
#
val_X = pd.DataFrame(val_X)
# val_X['last_visit'] = val_y[0]
# val_X['last_DX'] = val_y[1]
#
test_X = pd.DataFrame(test_X)
# test_X['last_visit'] = test_y[0]
# test_X['last_DX'] = test_y[1]

# save the data
train_X.to_csv('data/train_features.csv', index=False)
val_X.to_csv('data/val_features.csv', index=False)
test_X.to_csv('data/test_features.csv', index=False)

# save the labels
train_y = pd.DataFrame(train_y).T
val_y = pd.DataFrame(val_y).T
test_y = pd.DataFrame(test_y).T

train_y.to_csv('data/train_labels.csv', index=False)
val_y.to_csv('data/val_labels.csv', index=False)
test_y.to_csv('data/test_labels.csv', index=False)
