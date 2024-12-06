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
from lifelines import *
import pandas as pd
import argparse
from operator import itemgetter
# from concordance import concordance_index
import optuna
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.impute import KNNImputer
from skrebate import ReliefF

np.random.seed(1234)

codes = {"CN_MCI": 0, "Dementia": 1}

dat = pd.read_csv("data/mci_preprocessed_wo_csf.csv")
# drop first two columns
# dat.drop(dat.columns[0], axis=1, inplace=True)
# dat.drop(dat.columns[0], axis=1, inplace=True)
dat["last_DX"].replace(codes, inplace=True)

best_params = []
best_c_index = []

# define 10-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

# split off test set
dat_train, dat_test = train_test_split(dat, test_size=0.2, random_state=1234)

get_target = lambda df: (df["last_visit"].values, df["last_DX"].values)

best_params = []
best_c_index = []

parser = argparse.ArgumentParser(description="Survival analysis")
parser.add_argument("--max_time", type=int, default=17, help="Max number of months")
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--N", type=int, default=6, help="Number of modules")
parser.add_argument(
    "--num_heads", type=int, default=8
)  # number of heads in multi-head attention
parser.add_argument("--d_model", type=int, default=64)  # dimension of model
parser.add_argument("--d_ff", type=int, default=256)  # dimension of feedforward network
parser.add_argument(
    "--train_batch_size", type=int, default=16
)  # batch size for training
parser.add_argument("--drop_prob", type=float, default=0.1)  # dropout probability
parser.add_argument(
    "--lr", type=float, default=0.0001
)  # learning rate can be optimized by using a scheduler
parser.add_argument(
    "--coeff", type=float, default=1.0
)  # coefficient for the loss function
parser.add_argument(
    "--coeff2", type=float, default=1.0
)  # coefficient for the loss function
parser.add_argument("--data_dir", type=str, default="data")  # data directory
parser.add_argument(
    "--save_ckpt_dir", type=str, default="checkpoints"
)  # checkpoint directory
parser.add_argument("--report_interval", type=int, default=5)  # report interval
parser.add_argument(
    "--data_parallel", action="store_true", help="use data parallel?"
)  # use data parallel?
parser.add_argument(
    "--pred_method", type=str, choices=["mean", "median"], default="mean"
)  # prediction method
parser.add_argument(
    "--return_counts", type=bool, default=True
)  # return_counts is True if the number of events and censored are returned
parser.add_argument(
    "--mode", default="client"
)  # mode is client if the model is run on a client, server if the model is run on a server
parser.add_argument("--port", default=52162)  # port number
parser.add_argument("--host", default="127.0.0.1")  # host number
opt = parser.parse_args()
print(opt)


def dataframe_to_deepsurv_ds(x, t, e):
    # Extract the event and time columns as numpy arrays
    e = e.astype(np.int32)
    t = t.astype(np.float32)

    # Extract the patient's covariates as a numpy array
    x = x.astype(np.float32)

    # Return the deep surv dataframe
    return {"x": x, "e": e, "t": t}


# Create the checkpoint directory
if not os.path.exists(opt.save_ckpt_dir):
    os.makedirs(opt.save_ckpt_dir)


# class TranDataset(Dataset): # dataset class
#     def __init__(self, features, labels, is_train=True):
#         self.is_train = is_train
#         self.data = []
#
#         temp = []
#         for feature, label in zip(features, labels):
#             feature = torch.from_numpy(feature).float() # create tensor from numpy array
#             duration, is_observed = label[0], label[1] # duration is the time of event or censoring (in months) and is_observed is 1 if event is observed, 0 if censored
#             temp.append([duration, is_observed, feature]) # temp is a list of lists, each list is [duration, is_observed, feature]
#         sorted_temp = sorted(temp, key=itemgetter(0)) # sort temp by duration
#
#         if self.is_train: # if training, use sorted_temp
#             new_temp = sorted_temp # new_temp is a list of lists, each list is [duration, is_observed, feature]
#         else: # if testing, use the original order
#             new_temp = temp
#
#         for duration, is_observed, feature in new_temp: # for each list in new_temp
#             if is_observed: # if event is observed
#                 mask = opt.max_time * [1.] # mask is a list of 1s of length max_time
#                 label = duration * [1.] + (opt.max_time - duration) * [0.] # label is a list of 1s of length duration and 0s of length max_time - duration
#                 feature = torch.stack(opt.max_time * [feature]) # feature is a tensor of size (max_time, feature_size)
#                 self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(),
#                                   torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(),
#                                   torch.tensor(is_observed).byte().cuda()]) # self.data is a list of lists, each list is [feature, duration, mask, label, is_observed]
#             else: # if event is censored
#                 # NOTE plus 1 to include day 0
#                 mask = (duration + 1) * [1.] + (opt.max_time - (duration + 1)) * [0.] # mask is a list of 1s of length duration + 1 and 0s of length max_time - (duration + 1)
#                 label = opt.max_time * [1.] # label is a list of 1s of length max_time
#                 feature = torch.stack(opt.max_time * [feature]) # feature is a tensor of size (max_time, feature_size)
#                 self.data.append([feature.cuda(), torch.tensor(duration).float().cuda(),
#                                   torch.tensor(mask).float().cuda(), torch.tensor(label).cuda(),
#                                   torch.tensor(is_observed).byte().cuda()]) # self.data is a list of lists, each list is [feature, duration, mask, label, is_observed]
#
#     def __getitem__(self, index_a): # index_a is the index of the list in self.data
#         if self.is_train: # if training
#             if index_a == len(self.data) - 1: # if index_a is the last index
#                 index_b = np.random.randint(len(self.data)) # index_b is a random integer between 0 and len(self.data)
#             else:
#                 # NOTE self.data is sorted
#                 index_b = np.random.randint(index_a+1, len(self.data)) # index_b is a random integer between index_a + 1 and len(self.data)
#             return [ [self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a])) ] # return a list of lists, each list is [feature, duration, mask, label, is_observed]
#         else: # if testing
#             return self.data[index_a] # return a list, [feature, duration, mask, label, is_observed]
#
#     def __len__(self): # return the length of self.data
#         return len(self.data)

import torch
from torch.utils.data import Dataset
from operator import itemgetter
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


# class TranDataset(Dataset):
#     def __init__(self, features, labels, is_train=True, opt=None, max_length=None):
#         assert opt is not None, "The 'opt' configuration is missing."
#
#         self.is_train = is_train
#         self.data = []
#         self.max_length = max_length or opt.max_time
#         temp = []
#
#         for idx, (feature, label) in enumerate(zip(features, labels)):
#             print(
#                 f"Before conversion, feature at index {idx} is of type {type(feature)}"
#             )
#             feature = torch.from_numpy(feature).float()  # This is the problematic line.
#             print(
#                 f"After conversion, feature at index {idx} is of type {type(feature)}"
#             )
#             duration, is_observed = label[0], label[1]
#             temp.append([duration, is_observed, feature])
#
#             if isinstance(feature, np.ndarray):
#                 feature = torch.from_numpy(feature).float()
#             elif isinstance(feature, torch.Tensor):
#                 feature = feature.float()
#             duration, is_observed = label[0], label[1]
#
#             # Ensure the feature tensor has the max length by padding if necessary
#             print(
#                 f"Type of self.max_length: {type(self.max_length)}, Value: {self.max_length}"
#             )
#             print(f"Type of len(feature): {type(len(feature))}, Value: {len(feature)}")
#
#             if len(feature) < self.max_length:
#                 padding = torch.zeros(self.max_length - len(feature))
#                 feature = torch.cat([feature, padding])
#
#             self.data.append([duration, is_observed, feature])
#
#             try:
#                 # feature = torch.from_numpy(feature).float()
#                 duration, is_observed = label[0], label[1]
#                 temp.append([duration, is_observed, feature])
#             except Exception as e:
#                 logging.error(
#                     f"Error processing feature: {feature} and label: {label}. Error: {e}"
#                 )
#                 continue
#
#         sorted_temp = sorted(temp, key=itemgetter(0))
#         new_temp = sorted_temp if self.is_train else temp
#         for duration, is_observed, feature in new_temp:
#             try:
#                 if is_observed:
#                     mask = opt.max_time * [1.0]
#                     label = duration * [1.0] + (opt.max_time - duration) * [0.0]
#                     feature = torch.stack(opt.max_time * [feature])
#                     self.data.append(
#                         [
#                             self._to_cuda(feature),
#                             self._to_cuda(torch.tensor(duration).float()),
#                             self._to_cuda(torch.tensor(mask).float()),
#                             self._to_cuda(torch.tensor(label)),
#                             self._to_cuda(torch.tensor(is_observed).byte()),
#                         ]
#                     )
#                 else:
#                     mask = (duration + 1) * [1.0] + (opt.max_time - (duration + 1)) * [
#                         0.0
#                     ]
#                     label = opt.max_time * [1.0]
#                     feature = torch.stack(opt.max_time * [feature])
#                     self.data.append(
#                         [
#                             self._to_cuda(feature),
#                             self._to_cuda(torch.tensor(duration).float()),
#                             self._to_cuda(torch.tensor(mask).float()),
#                             self._to_cuda(torch.tensor(label)),
#                             self._to_cuda(torch.tensor(is_observed).byte()),
#                         ]
#                     )
#             except Exception as e:
#                 logging.error(
#                     f"Error processing duration: {duration}, is_observed: {is_observed}, feature: {feature}. Error: {e}"
#                 )
#
#         def _to_cuda(self, tensor):
#             if torch.cuda.is_available():
#                 return tensor.cuda()
#             return tensor
#
#         def __getitem__(self, index_a):
#             try:
#                 if self.is_train:
#                     if index_a == len(self.data) - 1:
#                         index_b = np.random.randint(len(self.data))
#                     else:
#                         index_b = np.random.randint(index_a + 1, len(self.data))
#                     return [
#                         [self.data[index_a][i], self.data[index_b][i]]
#                         for i in range(len(self.data[index_a]))
#                     ]
#                 else:
#                     return self.data[index_a]
#             except Exception as e:
#                 logging.error(f"Error getting item at index: {index_a}. Error: {e}")
#                 return None
#
#         def __len__(self):
#             return len(self.data)  #


class TranDataset(Dataset):
    def __init__(self, features, labels, is_train=True):
        self.is_train = is_train
        self.data = []

        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            duration, is_observed = label[1], label[0]
            temp.append([duration, is_observed, feature])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, is_observed, feature in new_temp:
            if is_observed:
                mask = opt.max_time * [1.0]
                label = duration * [1.0] + (opt.max_time - duration) * [0.0]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [
                        feature.cuda(),
                        torch.tensor(duration).float().cuda(),
                        torch.tensor(mask).float().cuda(),
                        torch.tensor(label).cuda(),
                        torch.tensor(is_observed).byte().cuda(),
                    ]
                )
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.0] + (opt.max_time - (duration + 1)) * [0.0]
                label = opt.max_time * [1.0]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [
                        feature.cuda(),
                        torch.tensor(duration).float().cuda(),
                        torch.tensor(mask).float().cuda(),
                        torch.tensor(label).cuda(),
                        torch.tensor(is_observed).byte().cuda(),
                    ]
                )

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a + 1, len(self.data))
            return [
                [self.data[index_a][i], self.data[index_b][i]]
                for i in range(len(self.data[index_a]))
            ]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)


# x = TranDataset(features[0], labels[0], is_train=True)

# for feature, duration, mask, label, is_observed in x:
#     # change to tensor
#     feature = torch.from_numpy(feature).float()
#     duration = torch.tensor(duration).float()
#     mask = torch.tensor(mask).float()
#     label = torch.tensor(label)
#     is_observed = torch.tensor(is_observed).byte()
#     break


def clones(module, N):  # return a ModuleList of N identical modules
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(N)]
    )  # NOTE copy.deepcopy() is used to avoid sharing parameters between multiple layers of the same type in the Transformer model (https://pytorch.org/docs/stable/nn.html#torch.nn.ModuleList)


def attention(query, key, value, mask=None, dropout=None):  # query, key, value are tensors of size (batch_size, max_time, d_model)
    d_k = query.size(-1)  # d_k is the last dimension of query
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # scores is a tensor of size (batch_size, max_time, max_time)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):  # multi-headed attention module
    def __init__(
        self, h, d_model, dropout=0.1
    ):  # h is the number of heads, d_model is the dimension of the model
        super(MultiHeadedAttention, self).__init__()  # initialize nn.Module
        assert d_model % h == 0  # assert d_model is divisible by h
        self.d_k = d_model // h  # self.d_k is the dimension of each head
        self.h = h  # self.h is the number of heads
        self.linears = clones(
            nn.Linear(d_model, d_model), 4
        )  # self.linears is a ModuleList of 4 identical Linear layers
        self.attn = None  # self.attn is the attention matrix
        self.dropout = nn.Dropout(p=dropout)  # self.dropout is a dropout layer

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(
                1
            )  # mask is a tensor of size (batch_size, 1, 1, max_time)
        nbatches = query.size(0)  # nbatches is the batch size

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -torch.tensor(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
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
            is_observed.append(is_observed_single)
            sigmoid_preds = encoder.forward(features)
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            total_surv_probs.append(surv_probs)

            if opt.pred_method == "mean":
                pred_duration = torch.sum(surv_probs).item()
            elif opt.pred_method == "median":
                pred_duration = 0
                while True:
                    if surv_probs[pred_duration] < 0.5:
                        break
                    else:
                        pred_duration += 1
                        if pred_duration == len(surv_probs):
                            break

            true_duration = durations.squeeze().item()
            pred_durations.append(pred_duration)
            true_durations.append(true_duration)

            if is_observed_single:
                pred_obs_durations.append(pred_duration)
                true_obs_durations.append(true_duration)

        total_surv_probs = torch.stack(total_surv_probs)

        pred_obs_durations = np.asarray(pred_obs_durations)
        true_obs_durations = np.asarray(true_obs_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))

        pred_durations = np.asarray(pred_durations)
        true_durations = np.asarray(true_durations)
        is_observed = np.asarray(is_observed, dtype=bool)

        print("pred durations OBS", pred_durations[is_observed].round())
        print("true durations OBS", true_durations[is_observed].round())

        print("pred durations CRS", pred_durations[~is_observed].round())
        print("true durations CRS", true_durations[~is_observed].round())

        test_cindex = concordance_index(true_durations, pred_durations, is_observed)
        print("c index", test_cindex, "mean abs error (OBS)", mae_obs)

    return test_cindex, mae_obs, total_surv_probs


# for features, durations, mask, label, is_observed_single in train_loader:  # test_loader:
#     print(is_observed_single.)
#     print(features)
#     print(durations)
#     print(mask)
#     print(label)
#     break


def checkpoint(model, total_surv_probs):
    datadir = opt.data_dir.replace("/", ".")
    model_out_path = "{}/best_model_trainset_{}.pth".format(opt.save_ckpt_dir, datadir)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # Save npy array
    surv_probs_out_path = "{}/best_surv_probs_test_{}.pth".format(
        opt.save_ckpt_dir, datadir
    )
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())


from torch.nn.utils.rnn import pad_sequence


# def custom_collate(batch):
#     # Split the batch into individual components
#
#     features, true_durations, mask, label, is_observed = zip(*batch)
#
#     # Handle the nested lists (from index_a and index_b)
#     if isinstance(features[0][0], torch.Tensor):
#         features_a, features_b = zip(*features)
#         features = (torch.stack(features_a), torch.stack(features_b))
#
#         true_durations_a, true_durations_b = zip(*true_durations)
#         mask_a, mask_b = zip(*mask)
#         label_a, label_b = zip(*label)
#         is_observed_a, is_observed_b = zip(*is_observed)
#
#         # Stack the tensors
#         true_durations = (torch.stack(true_durations_a), torch.stack(true_durations_b))
#         mask = (torch.stack(mask_a), torch.stack(mask_b))
#         label = (torch.stack(label_a), torch.stack(label_b))
#         is_observed = (torch.stack(is_observed_a), torch.stack(is_observed_b))
#     else:
#         # If they are not nested lists, then stack as usual
#         features = torch.stack(features)
#         true_durations = torch.stack(true_durations)
#         mask = torch.stack(mask)
#         label = torch.stack(label)
#         is_observed = torch.stack(is_observed)
#
#     return features, true_durations, mask, label, is_observed

# def tensor_to_numpy(t):
#     if t.is_cuda:
#         t = t.cpu()
#     return t.numpy()


def custom_collate(batch):
    try:
        # Unzip the batch
        list1, list2, list3 = zip(*batch)

        # Process the first sublist
        scalars_1, tensors_1 = zip(*list1)
        for scalar in scalars_1:
            if not isinstance(scalar, (int, float)):
                print(f"Non-scalar detected in scalars_1: {scalar}")
        scalars_1 = torch.tensor(scalars_1, device=tensors_1[0].device)
        tensors_1 = torch.stack(tensors_1)

        # Process the second sublist
        scalars_2, tensors_2 = zip(*list2)
        for scalar in scalars_2:
            if not isinstance(scalar, (int, float)):
                print(f"Non-scalar detected in scalars_2: {scalar}")
        scalars_2 = torch.tensor(scalars_2, device=tensors_2[0].device)
        tensors_2 = torch.stack(tensors_2)

        # Process the third sublist
        tensors_3a, tensors_3b = zip(*list3)
        tensors_3a = torch.stack(tensors_3a)
        tensors_3b = torch.stack(tensors_3b)

        return [
            list(zip(scalars_1, tensors_1)),
            list(zip(scalars_2, tensors_2)),
            list(zip(tensors_3a, tensors_3b)),
        ]

    except Exception as e:
        print(f"Error encountered: {e}")
        return batch  # Return the original batch in case of an error


def train(features, labels, encoder, opt):
    # DataLoader with custom_collate
    train_loader = DataLoader(
        TranDataset(features[0], labels[0], is_train=True),
        batch_size=opt.train_batch_size,
        shuffle=True,
    )

    # NOTE VAL batch size is 1
    val_loader = DataLoader(
        TranDataset(features[1], labels[1], is_train=False), batch_size=1, shuffle=False
    )

    best_val_cindex = -1
    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    for t in range(opt.num_epochs):
        print("epoch", t)
        encoder.train()
        tot_loss = 0.0

        for features, true_durations, mask, label, is_observed in train_loader:
            optimizer.zero_grad()

            is_observed_a = is_observed[0]
            print("is_observed_a", is_observed_a)
            mask_a = mask[0]
            print("mask_a", mask_a)
            mask_b = mask[1]
            print("mask_b", mask_b)
            label_a = label[0]
            print("label_a", label_a)
            label_b = label[1]
            print("label_b", label_b)
            true_durations_a = true_durations[0]
            print("true_durations_a", true_durations_a)
            true_durations_b = true_durations[1]
            print("true_durations_b", true_durations_b)

            sigmoid_a = encoder.forward(features[0])
            print("sigmoid_a", sigmoid_a)
            surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
            print("surv_probs_a", surv_probs_a)
            loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)
            print("loss", loss)

            sigmoid_b = encoder.forward(features[1])
            print("sigmoid_b", sigmoid_b)
            surv_probs_b = torch.cumprod(sigmoid_b, dim=1)
            print("surv_probs_b", surv_probs_b)

            cond_a = is_observed_a & (true_durations_a < true_durations_b)
            print("cond_a", cond_a)
            if torch.sum(cond_a) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)
                diff = mean_lifetimes_b[cond_a] - mean_lifetimes_a[cond_a]
                true_diff = true_durations_b[cond_a] - true_durations_a[cond_a]
                loss += opt.coeff * torch.mean(nn.ReLU()(true_diff - diff))

            cond_a2 = is_observed_a
            if torch.sum(cond_a2) > 0:
                mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)
                loss += opt.coeff2 * F.l1_loss(
                    mean_lifetimes_a[cond_a2], true_durations_a[cond_a2]
                )

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        print("train total loss", tot_loss)

        # Evaluate
        if t > 0 and t % opt.report_interval == 0:
            print("VAL")
            val_cindex, val_mae, val_total_surv_probs = evaluate(encoder, val_loader)

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_val_mae = val_mae

            print("current val cindex", val_cindex, "val mae", val_mae)
            print("BEST val cindex", best_val_cindex, "mae", best_val_mae)
    return best_val_cindex


def main(features, labels, num_features, opt):
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = Encoder(
        encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features
    ).cuda()
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
    results = train(features, labels, encoder, opt)
    return results


# create bayesian optimizer objective


# def objective(trial):
#
#     parser = argparse.ArgumentParser(description='Survival analysis')
#     parser.add_argument('--max_time', type=int, default=17, help='Max number of months')
#     parser.add_argument('--num_epochs', type=int, default=200)
#     parser.add_argument('--N', type=int, default=6, help='Number of modules')
#     parser.add_argument('--num_heads', type=int, default=8)  # number of heads in multi-head attention
#     parser.add_argument('--d_model', type=int, default=64)  # dimension of model
#     parser.add_argument('--d_ff', type=int, default=256)  # dimension of feedforward network
#     parser.add_argument('--train_batch_size', type=int, default=16)  # batch size for training
#     parser.add_argument('--drop_prob', type=float, default=0.1)  # dropout probability
#     parser.add_argument('--lr', type=float, default=0.0001)  # learning rate can be optimized by using a scheduler
#     parser.add_argument('--coeff', type=float, default=1.0)  # coefficient for the loss function
#     parser.add_argument('--coeff2', type=float, default=1.0)  # coefficient for the loss function
#     parser.add_argument('--data_dir', type=str, default='data')  # data directory
#     parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints')  # checkpoint directory
#     parser.add_argument('--report_interval', type=int, default=5)  # report interval
#     parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')  # use data parallel?
#     parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')  # prediction method
#     parser.add_argument("--return_counts", type=bool,
#                         default=True)  # return_counts is True if the number of events and censored are returned
#     parser.add_argument("--mode",
#                         default='client')  # mode is client if the model is run on a client, server if the model is run on a server
#     parser.add_argument("--port", default=52162)  # port number
#     parser.add_argument("--host", default='127.0.0.1')  # host number
#     opt = parser.parse_args()
#
#     # generate hyperparameters
#     #opt.num_heads = trial.suggest_int("num_heads", 1, 8)
#     #opt.d_model = trial.suggest_int("d_model", 1, 64)
#     opt.d_ff = trial.suggest_int("d_ff", 1, 64)
#     opt.drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5)
#     opt.lr = trial.suggest_float("lr", 1e-5, 1e-1)
#     opt.coeff1 = trial.suggest_float("coeff1", 0.0, 1.0)
#     opt.coeff2 = trial.suggest_float("coeff2", 0.0, 1.0)
#
#     c_index = main(features, labels, num_features, opt)
#     return -c_index

for i, (train_index, test_index) in enumerate(kfold.split(dat_train)):
    print(f"Fold {i}:")
    df_train, df_val = (
        dat_train.iloc[train_index],
        dat_train.iloc[test_index],
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
    # get_target = lambda df: (df[["last_DX", "last_visit"]].astype("float32"))
    y_train = np.array(
        (np.array(df_train["last_DX"]), np.array(df_train["last_visit"]))
    ).astype("int32")

    y_val = np.array(
        (np.array(df_val["last_DX"]), np.array(df_val["last_visit"]))
    ).astype("int32")
    y_val_test = np.array(
        (np.array(df_val_test["last_DX"]), np.array(df_val_test["last_visit"]))
    ).astype("float32")

    x_train = relief.fit_transform(x_train[:, 2:], y_train[0]).astype("float32")
    x_val = relief.transform(x_val[:, 2:]).astype("float32")
    # x_test = relief.transform(x_test[:, 2:]).astype("float32")
    x_val_test = relief.transform(x_val_test[:, 2:]).astype("float32")

    # train = (x_train, y_train)
    # val = (x_val, y_val)

    features = [x_train, x_val]
    labels = [y_train, y_val]
    num_features = x_train.shape[1]
    # save features

    features_train = features[0]
    labels_train = labels[0]

    features_train.shape

    # We don't need to transform the test labels
    # durations_test, events_test = get_target(df_test)

    def objective(trial):

        parser = argparse.ArgumentParser(description="Survival analysis")
        parser.add_argument(
            "--max_time", type=int, default=17, help="Max number of months"
        )
        parser.add_argument("--num_epochs", type=int, default=200)
        parser.add_argument("--N", type=int, default=6, help="Number of modules")
        parser.add_argument(
            "--num_heads", type=int, default=8
        )  # number of heads in multi-head attention
        parser.add_argument("--d_model", type=int, default=64)  # dimension of model
        parser.add_argument(
            "--d_ff", type=int, default=256
        )  # dimension of feedforward network
        parser.add_argument(
            "--train_batch_size", type=int, default=16
        )  # batch size for training
        parser.add_argument(
            "--drop_prob", type=float, default=0.1
        )  # dropout probability
        parser.add_argument(
            "--lr", type=float, default=0.0001
        )  # learning rate can be optimized by using a scheduler
        parser.add_argument(
            "--coeff", type=float, default=1.0
        )  # coefficient for the loss function
        parser.add_argument(
            "--coeff2", type=float, default=1.0
        )  # coefficient for the loss function
        parser.add_argument("--data_dir", type=str, default="data")  # data directory
        parser.add_argument(
            "--save_ckpt_dir", type=str, default="checkpoints"
        )  # checkpoint directory
        parser.add_argument("--report_interval", type=int, default=5)  # report interval
        parser.add_argument(
            "--data_parallel", action="store_true", help="use data parallel?"
        )  # use data parallel?
        parser.add_argument(
            "--pred_method", type=str, choices=["mean", "median"], default="mean"
        )  # prediction method
        parser.add_argument(
            "--return_counts", type=bool, default=True
        )  # return_counts is True if the number of events and censored are returned
        parser.add_argument(
            "--mode", default="client"
        )  # mode is client if the model is run on a client, server if the model is run on a server
        parser.add_argument("--port", default=52162)  # port number
        parser.add_argument("--host", default="127.0.0.1")  # host number
        opt = parser.parse_args()

        # generate hyperparameters
        # opt.num_heads = trial.suggest_int("num_heads", 1, 8)
        # opt.d_model = trial.suggest_int("d_model", 1, 64)
        opt.d_ff = trial.suggest_int("d_ff", 1, 64)
        opt.drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5)
        opt.lr = trial.suggest_float("lr", 1e-5, 1e-1)
        opt.coeff = trial.suggest_float("coeff", 0.0, 1.0)
        opt.coeff2 = trial.suggest_float("coeff2", 0.0, 1.0)

        c_index = main(features, labels, num_features, opt)

        # Evaluate the model on the validation set
        surv = model.predict_surv_df(x_val_test)
        eval_surv = EvalSurv(surv, y_val_test[0], y_val_test[1], censor_surv="km")
        c_index = eval_surv.concordance_td()

        # Return the negative c-index as the objective value
        return -c_index

    study = optuna.create_study(direction="maximize")

    # Run the optimization
    study.optimize(objective, n_trials=200, n_jobs=-1)  # 50 trials was 0.68 on test set

    # Print optimization results
    print("Number of finished trials:", len(study.trials))
    print("Best trial parameters:", study.best_trial.params)
    print("Best score:", study.best_value)

    best_params.append(study.best_params)
    best_c_index.append(1 - abs(study.best_value))
    deephit = pd.DataFrame(best_params)
    deephit["c_index"] = best_c_index
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

# num_durations = 60
# labtrans = DeepHitSingle.label_transform(num_durations)


get_target = lambda df: df.iloc[:, 0:2].to_numpy()
y_train = get_target(df_train)
# # change to numpy array
y_val = get_target(df_val)

x_train = relief.fit_transform(x_train[:, 2:], y_train[:, 0]).astype("float32")
x_val = relief.transform(x_val[:, 2:]).astype("float32")
x_test = relief.transform(x_test[:, 2:]).astype("float32")

val = (x_val, y_val)
durations_test, events_test = get_target(df_test)

params = pd.read_csv("data/deephit_optuna.csv")

# find the best params using a weighted average of the c_index and the number of parameters
params["c_index"] = 1 - params["c_index"]
params["n_params"] = params["n_layers"] * params["n_units_l1"]
params["weighted"] = params["c_index"] * params["n_params"]

best_params = params.sort_values("weighted").iloc[0].to_dict()
# drop c_index value
best_params.pop("c_index")
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
eval_surv = EvalSurv(surv, y_train[0], y_train[1], censor_surv="km")
c_index_train = eval_surv.concordance_td()
print("C-index train: {:.4f}".format(c_index_train))

# Evaluate the model on the test set
surv = model.predict_surv_df(x_test)
eval_surv = EvalSurv(surv, durations_test, events_test, censor_surv="km")
c_index_test = eval_surv.concordance_td()
print("C-index test: {:.4f}".format(c_index_test))


#
# for i, (train_index, test_index) in enumerate(kfold.split(dat_train)):
#     print('Fold', i)
#     df_train, df_val = dat_train.iloc[train_index], dat_train.iloc[test_index]
#     col_standardize = df_train.columns[2:]
#     col_leave = df_train.columns[:2]
#     standardize = [([col], StandardScaler()) for col in col_standardize]
#     col_knn = [([col], KNNImputer(n_neighbors=5)) for col in col_standardize]
#     leave = [(col, None) for col in col_leave]
#     relief = ReliefF(n_neighbors=50, n_features_to_select=100)
#
#     x_mapper = DataFrameMapper(leave + standardize)
#     knn_mapper = DataFrameMapper(leave + col_knn )
#     x_train = pd.DataFrame(x_mapper.fit_transform(df_train), columns=dat_train.columns)
#     x_train = pd.DataFrame(knn_mapper.fit_transform(x_train), columns=dat_train.columns)
#     x_val = pd.DataFrame(x_mapper.transform(df_val), columns=dat_train.columns)
#     x_val = pd.DataFrame(knn_mapper.transform(x_val), columns=dat_train.columns)
#
#     y_train = get_target(df_train)
#     train_labels = np.stack((y_train[0], y_train[1]), axis=-1)
#     # convert to dataframe for convenience
#     #y_train = pd.DataFrame(y_train)
#
#     y_val = get_target(df_val)
#     val_labels = np.stack((y_val[0], y_val[1]), axis=-1)
# #    y_val = pd.DataFrame(y_val)
#
#     x_train_relief = relief.fit_transform(x_train.iloc[:, 2:].values, y_train[1])
#
#     # convert to dataframe for convenience
#     #x_train_relief = pd.DataFrame(x_train_relief)
#
#
#     # apply relief to validation set
#     x_val_relief = relief.transform(x_val.iloc[:, 2:].values)
#
#     x_val_relief.shape
#     #x_val_relief = pd.DataFrame(x_val_relief)
#
#     # concate the features and labels to numpy arrays
#
#     features = [x_train_relief, x_val_relief]
#     labels = [train_labels, val_labels]
#     num_features = x_train_relief.shape[1]
#     # check class of num_features
# #    print(type(num_features))
#
#     # write the features and labels as numpy arrays
#
#     # np.save(opt.data_dir + '/features.npy', features)
#     # np.save(opt.data_dir + '/labels.npy', labels)
#     # valid = pd.concat([x_val_relief, y_val], axis=1)
#     #
#     # # Convert the dataframes to deepsurv compatible dataset
#     # train_data = dataframe_to_deepsurv_ds(x = x_train_relief, t = y_train[0], e = y_train[1])
#     # valid_data = dataframe_to_deepsurv_ds(x = x_val_relief, t = y_val[0], e = y_val[1])
#
#     study = optuna.create_study(direction='maximize')
#
#     # Run the optimization
#     study.optimize(objective, n_trials=100)
#
#     # Get the best hyperparameters and objective value
#     best_params.append(study.best_params)
#     best_c_index.append(study.best_value)
#
#
#
# # Assuming you have defined features and labels
# train_dataset = TranDataset(features[0], labels[0], is_train=True, opt=opt)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
#
# # train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, collate_fn=custom_collate)
#
# def inspect_dataset_item(dataset, index):
#     item = dataset[index]
#     print(f"Item at index {index}:")
#     print(f"Type: {type(item)}")
#     if isinstance(item, (list, tuple)):
#         for j, subitem in enumerate(item):
#             print(f"  Subitem {j}: {type(subitem)}")
#     print("------")
#
# # Sample usage:
# inspect_dataset_item(train_dataset, 0)
#
#
# def inspect_subitems(dataset, index):
#     item = dataset[index]
#     print(f"Item at index {index}:")
#     if isinstance(item, (list, tuple)):
#         for j, subitem in enumerate(item):
#             print(f"  Subitem {j}: Type {type(subitem)}")
#             if isinstance(subitem, (list, tuple)):
#                 for k, subsubitem in enumerate(subitem[:5]):  # Displaying first 5 elements
#                     print(f"    Subsubitem {k}: Type {type(subsubitem)}")
#     print("------")
#
# # Sample usage:
# inspect_subitems(train_dataset, 0)
#
# num_samples_to_inspect = opt.train_batch_size
# subitem_0_samples = [train_dataset[i][0] for i in range(num_samples_to_inspect)]
# print(subitem_0_samples)
#
# sample_item = train_dataset[0]
# print(type(sample_item))
# print([type(subitem) for subitem in sample_item])
#
# for i, sublist in enumerate(sample_item):
#     print(f"Sublist {i}:")
#     print("  Type:", type(sublist))
#     print("  First few elements:", sublist[:5])
#
#
# # Fetch a batch
# batch = next(iter(train_loader))
#
#
#
#
#
# # Check the structure of the batch
# for item in batch:
#     print(type(item), len(item))


#
# if __name__ == '__main__':
#     train_features = np.load(opt.data_dir + '/train_features.npy')
#     val_features = np.load(opt.data_dir + '/valid_features.npy')
#     test_features = np.load(opt.data_dir + '/test_features.npy')
#     features = [train_features, val_features, test_features]
#
#     train_labels = np.load(opt.data_dir + '/train_labels.npy').astype(np.int32)
#     val_labels = np.load(opt.data_dir + '/valid_labels.npy').astype(np.int32)
#     test_labels = np.load(opt.data_dir + '/test_labels.npy').astype(np.int32)
#     labels = [train_labels, val_labels, test_labels]
#
#     num_features = train_features.shape[1]
#
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
