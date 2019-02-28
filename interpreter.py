import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from nn_blocks import *
from utils import *
from train import initialize_env, create_DAdata, create_Uttdata, device
import argparse
from pprint import pprint
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()

def interpreter(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _, _, turn = create_DAdata(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
    utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)

    print('load models')
    encoder, context = None, None
    if config['use_da']:
        encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
        context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
        encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(args.epoch))))
        context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_state{}.model'.format(args.epoch))))

    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DEC_HIDDEN']).to(device)

    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(args.epoch))))

    utt_encoder = None
    utt_context = None
    utt_decoder = None
    if config['use_utt'] or config['use_uttcontext']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=config['UTT_MAX_VOCAB']).to(device)
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(args.epoch))))
        utt_decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_dec_state{}.model'.format(args.epoch))))
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
        utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(args.epoch))))

    model = EncoderDecoderModel(device).to(device)

    da_context_hidden = context.initHidden(1, device) if config['use_da'] else None
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    print('ok, i\'m ready.')

    while 1:

        utterance = input('>> ').lower()

        if utterance == 'exit' or utterance == 'bye':
            print('see you again.')
            break

        XU_seq = en_preprocess(utterance)
        XU_seq = [utt_vocab.word2id[word] if word in utt_vocab.word2id.keys() else utt_vocab.word2id['<UNK>'] for word in XU_seq]

        # TODO: How to deal utterance's DA
        DA = da_vocab.word2id['<Statement>']
        X_tensor = torch.tensor([[DA]]).to(device)
        if config['turn']:
            turn_tensor = torch.tensor([[1]]).to(device)
        else:
            turn_tensor = None
        if config['use_utt'] or config['use_uttcontext']:
            XU_tensor = torch.tensor([XU_seq]).to(device)
        else:
            XU_tensor = None

        pred_seq, da_context_hidden, utt_context_hidden = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                                                      turn=turn_tensor,
                                                                      da_encoder=encoder, da_decoder=decoder, da_context=context,
                                                                      da_context_hidden=da_context_hidden,
                                                                      utt_encoder=utt_encoder, utt_decoder=utt_decoder, utt_context=utt_context,
                                                                      utt_context_hidden=utt_context_hidden,
                                                                      config=config, EOS_token=utt_vocab.word2id['<EOS>'], BOS_token=utt_vocab.word2id['<BOS>'])
        print()
        print(' '.join([utt_vocab.id2word[wid] for wid in pred_seq]))

        print()

    return 0

if __name__ == '__main__':
    interpreter(args.expr)

