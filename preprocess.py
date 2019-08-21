import os, re, json, sys

file_pattern = re.compile(r'^sw\_([0-9]+?)\_([0-9]+?)\.utt\.txt$')
line_pattern = re.compile(r'^(.*?)\t(.*?)\t(.*?)$')

tmp_da = {'A': None, 'B': None}

def preprocess(dir_path, filename):
    with open(os.path.join(dir_path, filename), 'r') as f, \
        open(os.path.join('./data/corpus/', filename[:-7] + 'jsonlines'), 'w') as out_f:
        data = f.read().split('\n')
        prev_caller = None
        das = []
        sentences = []

        for line in data:
            m = line_pattern.search(line)
            if not m is None:
                current_caller = m.group(1)
                if m.group(2) == '+':
                    da = tmp_da[current_caller]
                else:
                    da = m.group(2)
                    tmp_da[current_caller] = da
                assert da is not None, filename
                if current_caller == prev_caller:
                    das.append(da)
                    sentences.append(m.group(3))
                else:
                    if len(das) > 0 and len(sentences) > 0:
                        out_f.write(json.dumps({'caller': prev_caller,
                                                'DA': das,
                                                'sentence': sentences}))
                        out_f.write('\n')
                    das = [da]
                    sentences = [m.group(3)]
                    prev_caller = current_caller


def FileIter():
    for i in range(14):
        dir_path = os.path.join('./data/swda', 'sw{:02}utt'.format(i))
        print('preprocessing in {}'.format(dir_path))
        files = [f for f in os.listdir(dir_path) if file_pattern.match(f)]
        for i, filename in enumerate(files, 1):
            preprocess(dir_path, filename)
            print('\rFinish preprocess {}/{} files'.format(i, len(files)), end='')
        print()

def modify(dirname, filename):
    with open(os.path.join(dirname, filename), 'r') as f:
        for idx, line in enumerate(f.readlines(), 1):
            print('\r{} conversations'.format(idx), end='')
            prev_caller = None
            das = []
            sentences = []
            jsondata = json.loads(line)
            with open(os.path.join('./data/corpus/', 'sw_{}_{}.jsonlines'.format(re.search(r'(.*?)\.jsonl', filename).group(1), '{:04}'.format(idx))), 'w') as out_f:
                for utterance in jsondata['utts']:
                    current_caller = utterance[0]
                    da = utterance[2][0]
                    tmp_da[current_caller] = da
                    if current_caller == prev_caller:
                        das.append(da)
                        sentences.append(utterance[1])
                    else:
                        if len(das) > 0 and len(sentences) > 0:
                            out_f.write(json.dumps({'caller': prev_caller,
                                                    'DA': das,
                                                    'sentence': sentences}))
                            out_f.write('\n')
                        das = [da]
                        sentences = [utterance[1]]
                        prev_caller = current_caller
        print()


def main():
    dirname = './data/json_data'
    files = [f for f in os.listdir(dirname) if re.match(r'(.*?)\.jsonl', f)]
    for filename in files:
        print(filename)
        modify(dirname, filename)

if __name__ == '__main__':
    main()