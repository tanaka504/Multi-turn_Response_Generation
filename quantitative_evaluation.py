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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.spatial.distance import cosine
from pprint import pprint
from NRG_evaluation import BoW_score


hyp_pattern = re.compile(r'^\<BOS\> (.*?)\<EOS\>$')


def get_dataframe(result):
    df = pd.DataFrame({'DA_pred': [], 'DA_true': [],
                       'hyp_len': [], 'hyp_txt': [],
                       'ref_len': [], 'ref_txt': []})

    for idx, case in enumerate(result):
        df.loc[idx] = [case['DA_preds'][0],
                       case['DA_trues'][0],
                       len(case['hyps'][0].split(' ')),
                       re.sub(r'\<unk\>', '<UNK>', case['hyps'][0]),
                       len(case['refs'][0].split(' ')),
                       re.sub(r'\<unk\>', '<UNK>', case['refs'][0])]
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
    with open('./data/results/result_{}.pkl'.format(expr), 'rb') as f:
        result = pickle.load(f)
    df_result = get_dataframe(result)
    with open('./data/model/utterance_vocab.dict', 'rb') as f:
        utt_vocab = pickle.load(f)
    
    y_true = []
    y_pred = []
    for ele in result:
        y_true.append(ele['DA_preds'][0])
        y_pred.append(ele['DA_trues'][0])
    calc_average(y_true=y_true, y_pred=y_pred)
    
    # make documents for each dialogue-act
    documents = {tag : ' '.join([sentence for sentence in df_result[df_result['DA_true'] == tag]['ref_txt']]) for tag in set(df_result['DA_true'])}
    hyp_words = {tag : Counter([word for sentence in df_result[df_result['DA_pred'] == tag]['hyp_txt'] for word in sentence.split(' ')]) for tag in set(df_result['DA_pred'])}
    ref_words = {tag : Counter([word for sentence in df_result[df_result['DA_true'] == tag]['ref_txt'] for word in tokenize.word_tokenize(sentence)]) for tag in set(df_result['DA_true'])}

    tf_idf = tfidf(document=[sentence for sentence in documents.values()])
    keywords = {tag: [kwd for kwd in kwds] for kwds, tag in zip(tf_idf.get_topk(50), documents.keys())}
    df_corr = pd.DataFrame({'1-Index':[], '{}/pearson'.format(expr):[], '{}/spearman'.format(expr):[],})
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
            df_corr.loc[t] = [t, pearsonr(s_proposal, s_ref)[0], spearmanr(s_proposal, s_ref)[0]]

        else:
            print('No {} in hypothesis'.format(t))

    print(df_corr)
    df_corr.to_csv('./data/images/keyword_correlation_{}.csv'.format(expr))

    # df_tag = pd.DataFrame({'pearson':[], 'spearman':[]})
    # for a, b in combinations(set(df_result['DA_true']), 2):
    #     s_a = [0 for _ in range(len(utt_vocab))]
    #     s_b = [0 for _ in range(len(utt_vocab))]
    #     for w, c in ref_words[a].items():
    #         s_a[utt_vocab[w]] = c
    #     for w, c in ref_words[b].items():
    #         s_b[utt_vocab[w]] = c
    #     df_tag.loc['{} & {}'.format(a, b)] = [pearsonr(s_a, s_b)[0], spearmanr(s_a, s_b)[0]]
    #
    # print(df_tag)
    # df_tag.to_csv('./data/images/tagcorrelations.csv')

    mpmi = MPMI(documents=documents)

    df_mpmi = pd.DataFrame({'1-Index':[], '{}/MPMI'.format(expr):[]})

    for t in set(df_result['DA_true']):
        if t in hyp_words.keys():
            sentences = [[word for word in sentence.split(' ')] for sentence in df_result[df_result['DA_pred'] == t]['hyp_txt']]
            df_mpmi.loc[t] = [t, mpmi.get_score(sentences, t)]
        else:
            print('No {} in hypothesis'.format(t))
    
    print(df_mpmi)
    df_mpmi.to_csv('./data/images/mpmi_{}.csv'.format(expr))
    return df_result, df_corr, df_mpmi

def kgCVAE_evaluation(expr):
    with open('./data/results/result_{}.pkl'.format(expr), 'rb') as f:
        results = pickle.load(f)
    with open('./data/model/utterance_vocab.dict', 'rb') as f:
        utt_vocab = pickle.load(f)
    bow_score = BoW_score(w2v_path='./data/model/glove.6B.200d.txt', vocab=utt_vocab)
    DA_prec, DA_recall = [], []
    BLEU_1_prec, BLEU_1_recall = [], []
    BLEU_2_prec, BLEU_2_recall = [], []
    BLEU_3_prec, BLEU_3_recall = [], []
    BLEU_4_prec, BLEU_4_recall = [], []
    A_bow_prec, A_bow_recall = [], []
    E_bow_prec, E_bow_recall = [], []
    for idx, result in enumerate(results, 1):
        print('\r*** {}/{} ***'.format(idx, len(results)), end='')
        # DA prec/recall
        DA_prec.append(np.mean([1 if da in result['DA_trues'] else 0 for da in result['DA_preds']]))
        DA_recall.append(np.mean([1 if da in result['DA_preds'] else 0 for da in result['DA_trues']]))

        # BLEU-n prec/recall
        references = [ref.split(' ') for ref in result['refs']]
        hypothesis = [hyp.split(' ') for hyp in result['hyps']]
        BLEU_1_prec.append(np.max([np.max([calc_bleu(ref, hyp, 1) for ref in references]) for hyp in hypothesis]))
        BLEU_1_recall.append(np.max([np.max([calc_bleu(ref, hyp, 1) for hyp in hypothesis]) for ref in references]))

        BLEU_2_prec.append(np.max([np.max([calc_bleu(ref, hyp, 2) for ref in references]) for hyp in hypothesis]))
        BLEU_2_recall.append(np.max([np.max([calc_bleu(ref, hyp, 2) for hyp in hypothesis]) for ref in references]))

        BLEU_3_prec.append(np.max([np.max([calc_bleu(ref, hyp, 3) for ref in references]) for hyp in hypothesis]))
        BLEU_3_recall.append(np.max([np.max([calc_bleu(ref, hyp, 3) for hyp in hypothesis]) for ref in references]))

        BLEU_4_prec.append(np.max([np.max([calc_bleu(ref, hyp, 4) for ref in references]) for hyp in hypothesis]))
        BLEU_4_recall.append(np.max([np.max([calc_bleu(ref, hyp, 4) for hyp in hypothesis]) for ref in references]))

        # A/E-bow prec/recall
        A_bow, E_bow = bow_score.get_score(refs=references, hyps=hypothesis)
        A_bow_prec.append(A_bow[0])
        A_bow_recall.append(A_bow[1])
        E_bow_prec.append(E_bow[0])
        E_bow_recall.append(E_bow[1])
    print()
    result = {'DA prec': sum(DA_prec) / len(DA_prec),
    'DA recall': sum(DA_recall) / len(DA_recall),
    'BLEU-1 prec': float(np.mean(BLEU_1_prec)),
    'BLEU-1 recall': float(np.mean(BLEU_1_recall)),
    'BLEU-2 prec': float(np.mean(BLEU_2_prec)),
    'BLEU-2 recall': float(np.mean(BLEU_2_recall)),
    'BLEU-3 prec': float(np.mean(BLEU_3_prec)),
    'BLEU-3 recall': float(np.mean(BLEU_3_recall)),
    'BLEU-4 prec': float(np.mean(BLEU_4_prec)),
    'BLEU-4 recall': float(np.mean(BLEU_4_recall)),
    'A-bow prec': float(np.mean(A_bow_prec)),
    'A-bow recall': float(np.mean(A_bow_recall)),
    'E-bow prec': float(np.mean(E_bow_prec)),
    'E-bow recall': float(np.mean(E_bow_recall)),}
    # with open('./data/results/QE_{}.csv'.format(expr), 'w') as out_f:
    #     out_f.write('\n'.join(['{},{}'.format(k, v) for k, v in result.items()]))
    return pd.DataFrame(result, index=[expr]).T


def calc_bleu(ref, hyp, n):
    try:
        return sentence_bleu(references=[ref], hypothesis=hyp, smoothing_function=SmoothingFunction().method7, weights=[1/n for i in range(1, n+1)])
    except:
        return 0.0

def calc_bow_sim(ref, hyp):
    return 0, 0

def merge_df(_df_corr, _df_mpmi, _df_m2m):
    df_corr = pd.read_csv('./data/results/keyword_correlation.csv')
    df_mpmi = pd.read_csv('./data/results/mpmi.csv')
    df_m2m = pd.read_csv('./data/results/many2many_eval.csv')

    df_corr = pd.merge(df_corr, _df_corr, on='1-Index', how='outer')
    df_mpmi = pd.merge(df_mpmi, _df_mpmi, on='1-Index', how='outer')
    df_m2m = pd.concat([df_m2m, _df_m2m], axis=1)
    df_corr.to_csv('./data/results/20191003_keyword_correlation.csv')
    df_mpmi.to_csv('./data/results/20191003_mpmi.csv')
    df_m2m.to_csv('./data/results/20191003_many2many_eval.csv')


def main():
    # df_result, df_corr, df_mpmi = quantitative_evaluation('kgCVAE_DA_catLater')
    df_m2m = kgCVAE_evaluation('kgCVAE_w2v')
    print(df_m2m)
    input()
    experiments = ['kgCVAE_DA_woFeat_catLater']
    for expr in experiments:
        print(expr)
        _df_m2m = kgCVAE_evaluation(expr)
        # _df_result, _df_corr, _df_mpmi = quantitative_evaluation(expr)
        # df_corr = pd.merge(df_corr, _df_corr, on='1-Index', how='outer')
        # df_mpmi = pd.merge(df_mpmi, _df_mpmi, on='1-Index', how='outer')
        df_m2m = pd.concat([df_m2m, _df_m2m], axis=1)
    merge_df(df_corr, df_mpmi, df_m2m)
    # df_corr.to_csv('./data/results/20190918_keyword_correlation.csv')
    # df_mpmi.to_csv('./data/results/20190918_mpmi_merge_last.csv')
    # df_m2m.to_csv('./data/results/20190923_many2many_eval.csv')
    
def initialize():
    # df_result, df_corr, df_mpmi = quantitative_evaluation('kgCVAE')
    df_m2m = kgCVAE_evaluation('kgCVAE')
    experiments = ['kgCVAE_woFeat', 'kgCVAE_DA_woFeat', 'kgCVAE_DA',
                   'kgCVAE_w2v', 'kgCVAE_woFeat_w2v', 'kgCVAE_DA_w2v', 'kgCVAE_DA_woFeat_w2v']
    for expr in experiments:
        print(expr)
        _df_m2m = kgCVAE_evaluation(expr)
        # _df_result, _df_corr, _df_mpmi = quantitative_evaluation(expr)
        # df_corr = pd.merge(df_corr, _df_corr, on='1-Index', how='outer')
        # df_mpmi = pd.merge(df_mpmi, _df_mpmi, on='1-Index', how='outer')
        df_m2m = pd.concat([df_m2m, _df_m2m], axis=1)
    # merge_df(df_corr, df_mpmi, df_m2m)
    # df_corr.to_csv('./data/results/keyword_correlation.csv')
    # df_mpmi.to_csv('./data/results/mpmi.csv')
    df_m2m.to_csv('./data/results/many2many_eval.csv')


if __name__ == '__main__':
    # initialize()
    main()
