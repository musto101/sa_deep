"""
    The original Transformer code is from: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    The modification for survival analysis is done by: https://github.com/shihux/sa_transformer/blob/main/sa_transformer.py
    I have modified it for use on the ADNI dataset.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from lifelines import *
import pandas as pd
import argparse
from operator import itemgetter
from concordance import concordance_index
import optuna
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

parser = argparse.ArgumentParser(description='Survival analysis')
parser.add_argument('--max_time', type=int, default=500, help='Max number of months')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--N', type=int, default=6, help='Number of modules')
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--drop_prob', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--coeff', type=float, default=1.0)
parser.add_argument('--coeff2', type=float, default=1.0)
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints')
parser.add_argument('--report_interval', type=int, default=5)
parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
parser.add_argument('--mode', default='client')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--host', type=str, default='localhost')
opt = parser.parse_args()
# print(opt)

# Create the checkpoint directory
if not os.path.exists(opt.save_ckpt_dir):
    os.makedirs(opt.save_ckpt_dir)


class TranDataset(Dataset):
    def __init__(self, features, labels, is_train=True):
        self.is_train = is_train
        self.data = []

        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            duration, is_observed = label[0], label[1]
            temp.append([duration, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, is_observed, feature in new_temp:
            if is_observed:
                mask = opt.max_time * [1.]
                label = duration * [1.] + (opt.max_time - duration) * [0.]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(),
                     torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.] + (opt.max_time - (duration + 1)) * [0.]
                label = opt.max_time * [1.]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(),
                     torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a + 1, len(self.data))
            return [[self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a]))]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


# Final layer for the Transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        return torch.sigmoid(x.squeeze(-1))


class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)

    def forward(self, x, mask=None):
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def evaluate(encoder, test_loader):
    encoder.eval()
    with torch.no_grad():
        pred_durations, true_durations, is_observed = [], [], []
        pred_obs_durations, true_obs_durations = [], []

        total_surv_probs = []

        # NOTE batch size is 1
        for features, durations, mask, label, is_observed_single in test_loader:
            # print('durations', durations)
            # print('mask', mask)
            # print('label', label)
            # print('is_observed_single', is_observed_single)
            # print('features', features.shape, features.dtype)
            is_observed.append(is_observed_single)
            # print('is_observed', is_observed)
            sigmoid_preds = encoder.forward(features)
            # print('sigmoid_preds', sigmoid_preds)
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            # print('surv_probs', surv_probs)
            total_surv_probs.append(surv_probs)
            # print('total_surv_probs', total_surv_probs)

            if opt.pred_method == 'mean':
                pred_duration = torch.sum(surv_probs).item()
                # print('pred_duration', pred_duration)
            elif opt.pred_method == 'median':
                pred_duration = 0
                # print('pred_duration_median', pred_duration)
                while True:
                    if surv_probs[pred_duration] < 0.5:
                        break
                    else:
                        pred_duration += 1
                        if pred_duration == len(surv_probs):
                            break

            true_duration = durations.squeeze().item()
            # print('true_duration', true_duration)
            pred_durations.append(pred_duration)
            # print('pred_durations', pred_durations)
            true_durations.append(true_duration)
            # print('true_durations', true_durations)

            if is_observed_single:
                pred_obs_durations.append(pred_duration)
                # print('pred_obs_durations', pred_obs_durations)
                true_obs_durations.append(true_duration)
                # print('true_obs_durations', true_obs_durations)

        total_surv_probs = torch.stack(total_surv_probs)
        # print('total_surv_probs', total_surv_probs.shape)

        pred_obs_durations = np.asarray(pred_obs_durations)
        # print('pred_obs_durations', pred_obs_durations)
        true_obs_durations = np.asarray(true_obs_durations)
        # print('true_obs_durations', true_obs_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))
        # print('mae_obs', mae_obs)

        pred_durations = np.asarray(pred_durations)
        # print('pred durations', pred_durations)
        true_durations = np.asarray(true_durations)
        # print('true durations', true_durations)

        # print is_observed
        is_observed = torch.stack(is_observed)
        # change is_observed to a numpy array to use it as a mask
        is_observed = is_observed.cpu().data.numpy()
        # change is_observed to a boolean array
        is_observed = is_observed.astype(bool)
        # change is_observed to a 1D array
        is_observed = is_observed.squeeze()

        print('pred durations OBS', pred_durations[is_observed].round())
        print('true durations OBS', true_durations[is_observed].round())

        print('pred durations CRS', pred_durations[~is_observed].round())
        print('true durations CRS', true_durations[~is_observed].round())

        test_cindex = concordance_index(true_durations, pred_durations, is_observed)
        print('c index', test_cindex, 'mean abs error (OBS)', mae_obs)

    return test_cindex, mae_obs, total_surv_probs


def checkpoint(model, total_surv_probs):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}/best_model_trainset_{}.pth".format(opt.save_ckpt_dir, datadir)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # Save npy array
    surv_probs_out_path = "{}/best_surv_probs_test_{}.pth".format(opt.save_ckpt_dir, datadir)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())


def train(features, labels, encoder):
    train_loader = DataLoader(TranDataset(features[0], labels[0], is_train=True), batch_size=opt.train_batch_size,
                              shuffle=True)
    # NOTE VAL batch size is 1
    val_loader = DataLoader(TranDataset(features[1], labels[1], is_train=False), batch_size=1, shuffle=False)
    # NOTE TEST batch size is 1
    test_loader = DataLoader(TranDataset(features[2], labels[2], is_train=False), batch_size=1, shuffle=False)
    # NOTE using C index as early stopping criterion
    best_val_cindex, best_test_cindex, best_test_mae, best_epoch = -1, -1, 9999999, 0

    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    for t in range(opt.num_epochs):
        print('epoch', t)
        encoder.train()

        tot_loss = 0.
        for features, true_durations, mask, label, is_observed in train_loader:
            optimizer.zero_grad()

            is_observed_a = is_observed[0]
            mask_a = mask[0]
            mask_b = mask[1]
            label_a = label[0]
            label_b = label[1]
            true_durations_a = true_durations[0]
            true_durations_b = true_durations[1]

            sigmoid_a = encoder.forward(features[0])
            surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
            loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)

            sigmoid_b = encoder.forward(features[1])
            surv_probs_b = torch.cumprod(sigmoid_b, dim=1)

            cond_a = is_observed_a & (true_durations_a < true_durations_b)
            if torch.sum(cond_a) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
                diff = mean_lifetimes_b[cond_a] - mean_lifetimes_a[cond_a]
                true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
                loss += opt.coeff * torch.mean(nn.ReLU()(true_diff - diff))

            cond_a2 = is_observed_a
            if torch.sum(cond_a2) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                loss += opt.coeff2 * F.l1_loss(mean_lifetimes_a[cond_a2], true_durations_a[cond_a2])

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('train total loss', tot_loss)

        # Evaluate
        if t > 0 and t % opt.report_interval == 0:
            print('VAL')
            val_cindex, val_mae, val_total_surv_probs = evaluate(encoder, val_loader)
            # print('TEST')
            test_cindex, test_mae, test_total_surv_probs = evaluate(encoder, test_loader)

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_val_mae = val_mae
                best_test_cindex = test_cindex
                best_test_mae = test_mae
                best_epoch = t
                checkpoint(encoder, test_total_surv_probs)

            print('current val cindex', val_cindex, 'val mae', val_mae)
            print('BEST val cindex', best_val_cindex, 'mae', best_val_mae, 'at epoch', best_epoch)
            # print('best test cindex', best_test_cindex, 'mae', best_test_mae)
    return best_val_cindex, best_test_cindex


def main(features, labels, num_features):
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = Encoder(encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features).cuda()
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
    score, test = train(features, labels, encoder)
    return score, test


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

# implement smote for imbalanced data
smote = SMOTE()

# create an empty list to store the test concordance indices
preds = []

# create folds for cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

for i in kf.split(data):
    # print(i)
    training = data.iloc[i[0]]
    # print(train.head())
    test = data.iloc[i[1]]
    # print(test.head())

    # split train into train and validation
    val = training.sample(frac=0.2, random_state=0)
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
    print('train_X', train_X)
    val_X = preprocessing.transform(val_X)
    test_X = preprocessing.transform(test_X)

    # run the ReliefF algorithm

    fs = ReliefF(n_neighbors=50, n_features_to_select=200)

    train_X = fs.fit_transform(train_X, train_y[0])

    # find the top 200 features and their reliefF scores ordered from highest to lowest
    selected_scores = fs.feature_importances_
    selected_features = fs.top_features_

    # concatenate the scores and features into a single array and sort them by score descending
    selected = np.column_stack((selected_features, selected_scores))
    selected = selected[selected[:, 1].argsort()[::-1]]

    # extract the top 200 feature indices
    top_feature_indices = selected[:200, 0].astype(int)

    # filter the column names of the original dataframe
    selected = data.columns[top_feature_indices]

    # print('train_X', train_X)
    val_X = fs.transform(val_X)
    test_X = fs.transform(test_X)

    # convert the train data to a dataframe
    train_X = pd.DataFrame(train_X, columns=selected)

    # add the last_visit and last_DX columns to the dataframe
    train_X['last_visit'] = train_y[0]
    train_X['last_DX'] = train_y[1]

    val_X = pd.DataFrame(val_X, columns=selected)
    val_X['last_visit'] = val_y[0]
    val_X['last_DX'] = val_y[1]

    test_X = pd.DataFrame(test_X, columns=selected)
    test_X['last_visit'] = test_y[0]
    test_X['last_DX'] = test_y[1]

    # restore the column names


    # run smote on the training data
    train_X, train_y = smote.fit_resample(train_X.drop(['last_DX'], axis=1), train_X['last_DX'])

    # add last_DX back to train_X
    train_X['last_DX'] = train_y

    smote_val = SMOTE(k_neighbors=2)

    # run smoote on the validation data
    val_X, val_y = smote_val.fit_resample(val_X.drop(['last_DX'], axis=1), val_X['last_DX'])

    # add last_DX back to val_X
    val_X['last_DX'] = val_y

    # add las

    # train_features = train_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    #
    # val_features = val_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    # # print dtype of val_features
    # print(val_features.dtype)
    # test_features = test_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    # # print dtype of test_features
    # print(test_features.dtype)
    # features = [train_features, val_features, test_features]
    #
    # train_labels = train_X[['last_visit', 'last_DX']].to_numpy()
    # # print dtype of train_labels
    # print(train_labels[0].dtype)
    # val_labels = val_X[['last_visit', 'last_DX']].to_numpy()
    # # print dtype of val_labels
    # print(val_labels[0].dtype)
    # test_labels = test_X[['last_visit', 'last_DX']].to_numpy()
    # # print dtype of test_labels
    # print(test_labels[0].dtype)
    # labels = [train_labels, val_labels, test_labels]
    #
    # num_features = train_features.shape[1]
    #
    # total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
    #
    # # convert the train data to a dataframe
    # train_X = pd.DataFrame(train_X)
    #
    # # add the last_visit and last_DX columns to the dataframe
    # train_X['last_visit'] = train_y[0]
    # train_X['last_DX'] = train_y[1]
    #
    # val_X = pd.DataFrame(val_X)
    # val_X['last_visit'] = val_y[0]
    # val_X['last_DX'] = val_y[1]
    #
    # test_X = pd.DataFrame(test_X)
    # test_X['last_visit'] = test_y[0]
    # test_X['last_DX'] = test_y[1]


    def objective(trial):
        N = trial.suggest_int('N', 1, 10)
        num_heads = trial.suggest_int('num_heads', 1, 10)
        d_model = trial.suggest_int('d_model', 1, 100)
        d_ff = trial.suggest_int('d_ff', 1, 100)
        drop_prob = trial.suggest_float('drop_prob', 0.1, 0.9)
        lr = trial.suggest_float('lr', 1e-10, 1e-5)
        coeff = trial.suggest_float('coeff', 0.1, 10)
        coeff2 = trial.suggest_float('coeff2', 0.1, 10)

        opt.N = N
        # opt.num_heads = num_heads
        # opt.d_model = d_model
        opt.d_ff = d_ff
        opt.drop_prob = drop_prob
        opt.lr = lr
        opt.coeff = coeff
        opt.coeff2 = coeff2

        train_features = train_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
        # check train_features is a numpy array
        print(type(train_features))

        val_features = val_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
        # check val_features is a numpy array
        print(type(val_features))
        test_features = test_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
        # check test_features is a numpy array
        print(type(test_features))
        features = [train_features, val_features, test_features]
        # check features is a list
        print(type(features))

        train_labels = train_X[['last_visit', 'last_DX']].to_numpy()
        # check train_labels is a numpy array
        print(type(train_labels))

        val_labels = val_X[['last_visit', 'last_DX']].to_numpy()
        # check val_labels is a numpy array
        print(type(val_labels))
        test_labels = test_X[['last_visit', 'last_DX']].to_numpy()
        # check test_labels is a numpy array
        print(type(test_labels))
        labels = [train_labels, val_labels, test_labels]
        # check labels is a list
        print(type(labels))

        num_features = train_features.shape[1]
        # check num_features is an integer
        print(type(num_features))

        total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]

        # train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
        # val_features = pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False).to_numpy()
        # test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
        # features = [train_features, val_features, test_features]
        #
        # train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
        # val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
        # test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
        # labels = [train_labels, val_labels, test_labels]
        #
        # num_features = train_features.shape[1]
        # print('train features shape', train_features.shape)
        # print('train labels shape', train_labels.shape)
        # print('val features shape', val_features.shape)
        # print('val labels shape', val_labels.shape)
        # print('test features shape', test_features.shape)
        # print('test labels shape', test_labels.shape)
        # print()
        #
        # print('num features', num_features)
        # total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
        # print('total data', total_data)
        # print()
        #
        # print('train max label', train_labels[:, 0].max())
        # print('train min label', train_labels[:, 0].min())
        # print('val max label', val_labels[:, 0].max())
        # print('val min label', val_labels[:, 0].min())
        # print('test max label', test_labels[:, 0].max())
        # print('test min label', test_labels[:, 0].min())

        # show dtype of train_features
        # print(train_features.dtype)

        # make sure the features are in numpy format
        # train_features = train_features.to_numpy()
        # val_features = val_features.to_numpy()
        # test_features = test_features.to_numpy()
        # features = [train_features, val_features, test_features]
        # # show dtype of train_labels
        # print(train_labels[0].dtype)
        # # show dtype of val_features
        # print(val_features.dtype)
        # # show dtype of val_labels
        # print(val_labels[0].dtype)
        # # show dtype of test_features
        # print(test_features.dtype)
        # # show dtype of test_labels
        # print(test_labels[0].dtype)

        score, test = main(features, labels, num_features)

        return score

        # train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
        # val_features = pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False).to_numpy()
        # test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
        # features = [train_features, val_features, test_features]
        #
        # train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
        # val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
        # test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
        # labels = [train_labels, val_labels, test_labels]
        #
        # num_features = train_features.shape[1]
        # print('train features shape', train_features.shape)
        # print('train labels shape', train_labels.shape)
        # print('val features shape', val_features.shape)
        # print('val labels shape', val_labels.shape)
        # print('test features shape', test_features.shape)
        # print('test labels shape', test_labels.shape)
        # print()
        #
        # print('num features', num_features)
        # total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
        # print('total data', total_data)
        # print()
        #
        # print('train max label', train_labels[:, 0].max())
        # print('train min label', train_labels[:, 0].min())
        # print('val max label', val_labels[:, 0].max())
        # print('val min label', val_labels[:, 0].min())
        # print('test max label', test_labels[:, 0].max())
        # print('test min label', test_labels[:, 0].min())

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print(study.best_trial.value)

    # modify opt to include the best hyperparameters
    opt.N = study.best_params['N']
    # opt.num_heads = study.best_params['num_heads']
    # opt.d_model = study.best_params['d_model']
    opt.d_ff = study.best_params['d_ff']
    opt.drop_prob = study.best_params['drop_prob']
    opt.lr = study.best_params['lr']
    opt.coeff = study.best_params['coeff']
    opt.coeff2 = study.best_params['coeff2']

    train_features = train_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    # check train_features is a numpy array
    print(type(train_features))

    val_features = val_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    # check val_features is a numpy array
    print(type(val_features))
    test_features = test_X.drop(['last_visit', 'last_DX'], axis=1).to_numpy()
    # check test_features is a numpy array
    print(type(test_features))
    features = [train_features, val_features, test_features]
    # check features is a list
    print(type(features))

    train_labels = train_X[['last_visit', 'last_DX']].to_numpy()
    # check train_labels is a numpy array
    print(type(train_labels))

    val_labels = val_X[['last_visit', 'last_DX']].to_numpy()
    # check val_labels is a numpy array
    print(type(val_labels))
    test_labels = test_X[['last_visit', 'last_DX']].to_numpy()
    # check test_labels is a numpy array
    print(type(test_labels))
    labels = [train_labels, val_labels, test_labels]
    # check labels is a list
    print(type(labels))

    num_features = train_features.shape[1]
    # check num_features is an integer
    print(type(num_features))

    # # run the best model
    # train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
    # val_features = pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False).to_numpy()
    # test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
    # features = [train_features, val_features, test_features]
    #
    # train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
    # val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
    # test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
    # labels = [train_labels, val_labels, test_labels]
    #
    # num_features = train_features.shape[1]
    # print('train features shape', train_features.shape)
    # print('train labels shape', train_labels.shape)
    # print('val features shape', val_features.shape)
    # print('val labels shape', val_labels.shape)
    # print('test features shape', test_features.shape)
    #
    # print('num features', num_features)
    # total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
    # print('total data', total_data)
    #
    # print('train max label', train_labels[:, 0].max())

    score, test = main(features, labels, num_features)

    preds.append(test)

print('Mean test concordance index', np.mean(preds))






# if __name__ == '__main__':
#     train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
#     val_features = pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False).to_numpy()
#     test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
#     features = [train_features, val_features, test_features]
#
#     train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
#     val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
#     test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
#     labels = [train_labels, val_labels, test_labels]
#
#     num_features = train_features.shape[1]
#     print('train features shape', train_features.shape)
#     print('train labels shape', train_labels.shape)
#     print('val features shape', val_features.shape)
#     print('val labels shape', val_labels.shape)
#     print('test features shape', test_features.shape)
#     print('test labels shape', test_labels.shape)
#     print()
#
#     print('num features', num_features)
#     total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
#     print('total data', total_data)
#     print()
#
#     print('train max label', train_labels[:, 0].max())
#     print('train min label', train_labels[:, 0].min())
#     print('val max label', val_labels[:, 0].max())
#     print('val min label', val_labels[:, 0].min())
#     print('test max label', test_labels[:, 0].max())
#     print('test min label', test_labels[:, 0].min())
#
#     main(features, labels, num_features)
