import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from utils import *
from train import initialize_env, create_data, device

model_num = 2

def evaluate():
    print('load vocab')
    config = initialize_env('DAonly')
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = create_data(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)

    X_test, Y_test = da_vocab.tokenize(X_test, Y_test)

    print('load models')
    encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
    context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
    model = DAonlyModel().to(device)
    encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(model_num))))
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(model_num))))
    context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_state{}.model'.format(model_num))))

    context_hidden = context.initHidden(device)

    correct = 0

    for seq_idx in range(0, len(X_test) - 1):
        X_seq = X_test[seq_idx]
        Y_seq = Y_test[seq_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'

        for i in range(0, len(X_seq)):
            print('\r{}/{} sequences evaluating'.format(i, len(X_seq)))
            X_tensor = torch.tensor([X_seq[i]]).to(device)
            Y_tensor = torch.tensor([Y_seq[i]]).to(device)
            decoder_output, context_hidden = model.predict(X_tensor, encoder, decoder, context, context_hidden)
            pred_idx = torch.argmax(decoder_output)
            if pred_idx.item() == Y_tensor.item(): correct += 1

    return correct / len(X_test)

if __name__ == '__main__':
    true_rate = evaluate()
    print(true_rate)