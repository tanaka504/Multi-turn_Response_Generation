import pickle, re
from pprint import  pprint
from train import initialize_env
from evaluation import calc_average
from utils import *
import pandas as pd
from collections import Counter
from nltk import tokenize
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import matplotlib.pyplot as plt

hyp_pattern = re.compile(r'^\<BOS\> (.*?)\<EOS\>$')


def get_dataframe(result):
    df = pd.DataFrame({'DA_pred': [], 'DA_true': [],
                       'hyp_len': [], 'hyp_txt': [],
                       'ref_len': [], 'ref_txt': []})

    for idx, case in enumerate(result):
        df.loc[idx] = [case['DA_preds'],
                       case['DA_trues'],
                       len(case['hyp'].split(' ')),
                       re.sub(r'\<unk\>', '<UNK>', case['hyp']),
                       len(case['ref'].split(' ')),
                       re.sub(r'\<unk\>', '<UNK>', case['ref'])]
    return df

def plotfig(X1, Y1, Y2, xlabel, ylabel, imgname):
    plt.figure(figsize=(40, 30))
    plt.rcParams['font.size'] = 48
    plt.scatter(X1, Y1, c='blue', s=100, label='Merge')
    plt.scatter(X1, Y2, c='red', s=100, label='Separate')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join('./data/images/', imgname))

    
def quantitative_evaluation(expr):
    with open('./data/images/results_{}.pkl'.format(expr), 'rb') as f:
        result = pickle.load(f)
    df_result = get_dataframe(result)
    with open('./data/model/utterance_vocab.dict', 'rb') as f:
        utt_vocab = pickle.load(f)
    
    # make documents for each dialogue-act
    documents = {tag : ' '.join([sentence for sentence in df_result[df_result['DA_true'] == tag]['ref_txt']]) for tag in set(df_result['DA_true'])}
    hyp_words = {tag : Counter([word for sentence in df_result[df_result['DA_pred'] == tag]['hyp_txt'] for word in sentence.split(' ')]) for tag in set(df_result['DA_pred'])}
    ref_words = {tag : Counter([word for sentence in df_result[df_result['DA_true'] == tag]['ref_txt'] for word in tokenize.word_tokenize(sentence)]) for tag in set(df_result['DA_true'])}

    tf_idf = tfidf(document=[sentence for sentence in documents.values()])
    # tf_idf = tfidf(document=[document for document in documents.values()])
    keywords = {tag: [kwd for kwd in kwds] for kwds, tag in zip(tf_idf.get_topk(50), documents.keys())}
    df_corr = pd.DataFrame({'{}/pearson'.format(expr):[], '{}/spearman'.format(expr):[],})
    for t in set(df_result['DA_true']):
        if t in hyp_words.keys():
            vocab = keywords[t]
            s_ref = [0 for _ in range(len(vocab))]
            s_proposal = [0 for _ in range(len(vocab))]
            for w, c in ref_words[t].items():
                if w in vocab:
                    s_ref[vocab.index(w)] = c
            for w, c in hyp_words[t].items():
                if w in vocab:
                    s_proposal[vocab.index(w)] = c
            # plotfig(X1=s_ref, Y1=s_proposal, Y2=s_proposal1, xlabel='reference', ylabel='Models', imgname='scatter_{}.png'.format(t))
            df_corr.loc[t] = [pearsonr(s_proposal, s_ref)[0], spearmanr(s_proposal, s_ref)[0]]

        else:
            print('No {} in hypothesis'.format(t))

    print(df_corr)
    df_corr.to_csv('./data/images/keyword_correlation.csv')

    df_tag = pd.DataFrame({'pearson':[], 'spearman':[]})
    for a, b in combinations(set(df_result['DA_true']), 2):
        s_a = [0 for _ in range(len(utt_vocab))]
        s_b = [0 for _ in range(len(utt_vocab))]
        for w, c in ref_words[a].items():
            s_a[utt_vocab[w]] = c
        for w, c in ref_words[b].items():
            s_b[utt_vocab[w]] = c
        df_tag.loc['{} & {}'.format(a, b)] = [pearsonr(s_a, s_b)[0], spearmanr(s_a, s_b)[0]]

    print(df_tag)
    df_tag.to_csv('./data/images/tagcorrelations.csv')

    mpmi = MPMI(documents=documents)

    df_mpmi = pd.DataFrame({'{}/MPMI'.format(expr):[]})

    for t in set(df_result['DA_true']):
        if t in hyp_words.keys():
            sentences = [[word for word in sentence.split(' ')] for sentence in df_result[df_result['DA_pred'] == t]['hyp_txt']]
            df_mpmi.loc[t] = [mpmi.get_score(sentences, t)]
        else:
            print('No {} in hypothesis'.format(t))
    
    print(df_mpmi)
    df_mpmi.to_csv('./data/images/mpmi.csv')

def main():
    experiments = ['kgcvae']
    for expr in experiments:
        quantitative_evaluation(expr)


if __name__ == '__main__':
    main()

