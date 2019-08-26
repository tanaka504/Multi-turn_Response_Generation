import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from nn_blocks import *
from utils import *
from train import initialize_env, parse
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint
import numpy as np
import pickle

def evaluate(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_test, Y_test, XU_test, YU_test, turn = create_traindata(config=config, prefix='test')
    da_vocab = da_Vocab(config=config, create_vocab=False)
    utt_vocab = utt_Vocab(config=config, create_vocab=False)

    X_test, Y_test = da_vocab.tokenize(X_test, Y_test)
    XU_test, YU_test = utt_vocab.tokenize(XU_test, YU_test)

    print('load models')
    encoder, context, decoder = None, None, None
    if config['use_da']:
        encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
        context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
        encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(args.epoch))))
        context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_state{}.model'.format(args.epoch))))

        decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                            da_hidden=config['DEC_HIDDEN']).to(device)

        decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(args.epoch))))

    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>'], fine_tuning=True).to(device)
    utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(args.epoch))))

    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(args.epoch))))

    utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=len(utt_vocab.word2id)).to(device)
    utt_decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(args.epoch))))
    model = EncoderDecoderModel(da_vocab=da_vocab, utt_vocab=utt_vocab, device=device,
                                da_encoder=encoder, utt_encoder=utt_encoder,
                                da_context=context, utt_context=utt_context,
                                da_decoder=decoder, utt_decoder=utt_decoder, config=config).to(device)

    da_context_hidden = context.initHidden(1, device) if config['use_da'] else None
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    result = []

    for seq_idx in range(0, len(X_test)):
        print('\r{}/{} conversation evaluating'.format(seq_idx+1, len(X_test)), end='')
        X_seq = X_test[seq_idx]
        Y_seq = Y_test[seq_idx]
        XU_seq = XU_test[seq_idx]
        YU_seq = YU_test[seq_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([[X_seq[i]]]).to(device)
            XU_tensor = torch.tensor([XU_seq[i]]).to(device)

            pred_seq, da_context_hidden, utt_context_hidden, decoder_output = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                                           da_context_hidden=da_context_hidden, utt_context_hidden=utt_context_hidden)
        Y_tensor = Y_seq[-1]
        YU_tensor = YU_seq[-1]
        if config['use_da']:
            pred_idx = torch.argmax(decoder_output).item()
            DA_pred = da_vocab.id2word[pred_idx]
        else:
            DA_pred = 'None'
        if not pred_seq[-1] == utt_vocab.word2id['<EOS>']:
            pred_seq.append(utt_vocab.word2id['<EOS>'])
        result.append({'DA_preds': DA_pred,
                       'DA_trues': da_vocab.id2word[Y_tensor],
                       'hyp': ' '.join([utt_vocab.id2word[wid] for wid in pred_seq]),
                       'ref': ' '.join([utt_vocab.id2word[wid] for wid in YU_tensor]),
                       'context': [' '.join(utt_vocab.id2word[wid] for wid in seq) for seq in XU_seq]})
    print()

    return result

def calc_average(y_true, y_pred):
    p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('p: {} | r: {} | f: {} | acc: {}'.format(p, r, f, acc))


def save_cmx(y_true, y_pred, expr):
    fontsize = 40
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(40, 30))
    plt.rcParams['font.size'] = fontsize
    heatmap = sns.heatmap(df_cmx, annot=True, fmt='d', cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('./data/images/cmx_{}.png'.format(expr))


if __name__ == '__main__':
    global args, device
    args, device = parse()
    result = evaluate(args.expr)
    with open('./data/images/results_{}.pkl'.format(args.expr), 'wb') as f:
        pickle.dump(result, f)
    # calc_average(y_true=result['DA_trues'], y_pred=result['DA_preds'])
