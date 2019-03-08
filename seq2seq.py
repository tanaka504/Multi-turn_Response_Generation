import time
import os
import pyhocon
import torch
from torch import nn, optim
from models import *
from utils import *
from nn_blocks import *
from train import initialize_env, create_Uttdata, make_batchidx, minimize, parse
import random
import argparse
import pickle


def parallelize(X, Y):
    X = [utt for conv in X for utt in conv]
    Y = [utt for conv in Y for utt in conv]
    return X, Y

def train(experiment):
    print('experiment: ', experiment)
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _ = create_Uttdata(config)

    vocab = utt_Vocab(config, X_train + X_valid, Y_train + Y_valid)

    X_train, Y_train = minimize(X_train), minimize(Y_train)
    X_valid, Y_valid = minimize(X_valid), minimize(Y_valid)

    with open('./data/minidata.pkl', 'wb') as f:
        a, b = parallelize(X_train, Y_train)
        pickle.dump([(c, d) for c, d in zip(a, b)], f)


    X_train, Y_train = vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = vocab.tokenize(X_valid, Y_valid)

    X_train, Y_train = parallelize(X_train, Y_train)
    X_valid, Y_valid = parallelize(X_valid, Y_valid)
    print('Finish create dataset')
    
    lr = config['lr']
    batch_size = config['BATCH_SIZE']

    encoder = UtteranceEncoder(utt_input_size=len(vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=vocab.word2id['<UttPAD>']).to(device)
    decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=config['UTT_MAX_VOCAB']).to(device)

    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)

    model = seq2seq(device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2id['<UttPAD>'])

    start = time.time()
    
    print_total_loss = 0
    _valid_loss = None

    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e + 1))

        indexes = [i for i in range(len(X_train))]
        random.shuffle(indexes)
        k = 0
        while k < len(indexes):
            step_size = min(batch_size, len(indexes) - k)

            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            batch_idx = indexes[k: k + step_size]

            print('\r{}/{} pairs training ...'.format(k + step_size, len(X_train)), end='')

            X_seq = [X_train[seq_idx] for seq_idx in batch_idx]
            Y_seq = [Y_train[seq_idx] for seq_idx in batch_idx]
            
            max_xseq_len = max(len(x) + 1 for x in X_seq)
            max_yseq_len = max(len(y) + 1 for y in Y_seq)

            for si in range(len(X_seq)):
                X_seq[si] = X_seq[si] + [vocab.word2id['<UttPAD>']] * (max_xseq_len - len(X_seq[si]))
                Y_seq[si] = Y_seq[si] + [vocab.word2id['<UttPAD>']] * (max_yseq_len - len(Y_seq[si]))
            X_tensor = torch.tensor([x for x in X_seq]).to(device)
            Y_tensor = torch.tensor([y for y in Y_seq]).to(device)
            
            loss = model.forward(X=X_tensor, Y=Y_tensor, encoder=encoder, decoder=decoder, step_size=step_size, criterion=criterion, config=config)
            print_total_loss += loss

            encoder_opt.step()
            decoder_opt.step()

            k += step_size

        print()
            
        valid_loss = validation(X=X_valid, Y=Y_valid, model=model, encoder=encoder, decoder=decoder, vocab=vocab, config=config)

        if _valid_loss is None:
            torch.save(encoder.state_dict(), os.path.join(config['log_dir'], 'enc_beststate.model'))
            torch.save(decoder.state_dict(), os.path.join(config['log_dir'], 'dec_beststate.model'))
        else:
            if _valid_loss > valid_loss:
                torch.save(encoder.state_dict(), os.path.join(config['log_dir'], 'enc_beststate'))
                torch.save(decoder.state_dict(), os.path.join(config['log_dir'], 'dec_beststate.model'))

        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))
        
        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            torch.save(encoder.state_dict(), os.path.join(config['log_dir'], 'enc_state{}.model'.format(e + 1)))
            torch.save(decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))

        print()
        print('Finish training | exec time: %.4f [sec]' % (time.time() - start))

def validation(X, Y, model, encoder, decoder, vocab, config):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2id['<UttPAD>'])
    total_loss = 0

    for seq_idx in range(len(X)):
        X_seq = X[seq_idx]
        Y_seq = Y[seq_idx]

        X_tensor = torch.tensor([X_seq]).to(device)
        Y_tensor = torch.tensor([Y_seq]).to(device)

        loss = model.evaluate(X=X_tensor, Y=Y_tensor, encoder=encoder, decoder=decoder, criterion=criterion, config=config)
        total_loss += loss

    return total_loss

def interpreter(experiment):
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _ = create_Uttdata(config)
    vocab = utt_Vocab(config, X_train + X_valid, Y_train + Y_valid)

    encoder = UtteranceEncoder(utt_input_size=len(vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=vocab.word2id['<UttPAD>']).to(device)
    decoder = UtteranceDecoder(utterance_hidden_size=config['DEC_HIDDEN'], utt_embed_size=config['UTT_EMBED'], utt_vocab_size=config['UTT_MAX_VOCAB']).to(device)

    encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(args.epoch))))
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(args.epoch))))

    model = seq2seq(device).to(device)

    while 1:
        utterance = input('>> ').lower()

        if utterance == 'exit' or utterance == 'bye':
            print('see you again.')
            break

        X_seq = en_preprocess(utterance)
        X_seq = [vocab.word2id[word] if word in vocab.word2id.keys() else vocab.word2id['<UNK>'] for word in X_seq]

        X_tensor = torch.tensor([X_seq]).to(device)

        pred_seq = model.predict(X=X_tensor, encoder=encoder, decoder=decoder, config=config, EOS_token=vocab.word2id['<EOS>'], BOS_token=vocab.word2id['<BOS>'])

        print()
        print(' '.join([vocab.id2word[wid] for wid in pred_seq]))
        print()

    return 0

if __name__ == '__main__':
    global args, device
    args, device = parse()
    # train('seq2seq')
    interpreter('seq2seq')
