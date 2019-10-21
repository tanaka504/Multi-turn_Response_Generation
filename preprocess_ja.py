import re, os, json
import random
import MeCab


"""
JAISTタグ付きコーパス用の前処理スクリプト
"""


corpus_dir = './data/JAIST_corpus/jaist/'
out_dir = './data/corpus/jaist/'
line_pattern = re.compile(r'^(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)$')
file_pattern = re.compile(r'^data([0-9]*?)\.txt$')
tagger = MeCab.Tagger('-Owakati')

def data_split(files):
    indexes = [i for i in range(len(files))]
    random.shuffle(indexes)
    dev_test = len(files) // 10
    test_files = [files[x] for x in indexes[:dev_test]]
    dev_files = [files[x] for x in indexes[dev_test: dev_test*2]]
    train_files = [files[x] for x in indexes[dev_test*2:]]
    return train_files, dev_files, test_files


def sentence_cleaner(sentence):
    sentence = sentence.translate(str.maketrans({chr(0xFF01 + i): chr(0x0021 + i) for i in range(94)}))
    sentence = re.sub(r'[\(|\（](.*?)[\)|\）]', '', sentence)
    sentence = re.sub(r'[\<|\＜](.*?)[\>|\＞]', '', sentence)
    sentence = re.sub(r'\【(.*?)\】', '', sentence)
    sentence = tagger.parse(sentence)
    return sentence

def conv_split(m):
    if m is None:
        return True
    sentence = sentence_cleaner(m.group(3))
    if '＊＊＊' in sentence:
        return True
    if sentence == '':
        return True

    return False

def preprocess(filename, prefix):
    serial = 0
    conv_turn = 0
    f = open(os.path.join(corpus_dir, filename), 'r')
    out_f = open(os.path.join(out_dir, '{}_{}_0.jsonlines'.format(filename.split('.')[0], prefix)), 'w')
    das = []
    sentences = []
    for line in f.readlines():
        m = line_pattern.search(line)
        if conv_split(m) and conv_turn > 0:
            out_f.close()
            serial += 1
            out_f = open(os.path.join(out_dir, '{}_{}_{}.jsonlines'.format(filename.split('.')[0], prefix, serial)), 'w')
            das = []
            sentences = []
            conv_turn = 0
            continue
        sentence = sentence_cleaner(m.group(3)).strip()
        da = m.group(5)
        if m.group(4) == '1':
            if len(das) > 0:
                out_f.write(json.dumps({'caller': current_caller,
                                        'DA': das,
                                        'sentence': sentences}, ensure_ascii=False))
                out_f.write('\n')
            das = [da]
            sentences = [sentence]
        else:
            das.append(da)
            sentences.append(sentence)
        conv_turn += 1
        current_caller = m.group(2)

def main():
    files = [f for f in os.listdir(corpus_dir) if file_pattern.match(f)]

    def process(files, prefix):
        for idx, filename in enumerate(files, 1):
            print('\r {} *** {}/{} ***'.format(prefix, idx, len(files)), end='')
            preprocess(filename, prefix)
        print()

    train, dev, test = data_split(files)
    process(train, 'train')
    process(dev, 'dev')
    process(test, 'test')

if __name__ == '__main__':
    main()
