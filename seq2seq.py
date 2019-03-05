import time
import os
import pyhocon
import torch
from torch import nn, optim
from models import *
from utils import *
from nn_blocks import *
import argparse
from train import initialize_env, create_Uttdata, make_batchidx

def parallelize(X, Y):
    return X, Y

def train(experiment):
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _ = create_Uttdata(config)

    vocab = utt_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    X_train, Y_train = vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = vocab.tokenize(X_valid, Y_valid)

