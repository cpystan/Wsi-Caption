import json
import re
from collections import Counter
import os

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = '../ocr/datasets/TCGA_BRCA'
        self.threshold = args.threshold
        #self.dataset_name = args.dataset_name
        self.dataset_name = 'BRCA'
        if self.dataset_name == 'BRCA':
            self.clean_report = self.clean_report_brca

       
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []
        root = self.ann_path
        for dir in os.listdir(root):
            file_name = os.path.join(root, dir, 'annotation')

            anno = json.loads(open(file_name, 'r').read())
            tokens = self.clean_report(anno).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def clean_report_brca(self, report):
        report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ')\
            .replace(' 10. ', ' ').replace(' 11. ', ' ').replace(' 12. ', ' ').replace(' 13. ', ' ').replace(' 14.', ' ')    \
            .replace(' 1. ', ' ').replace(' 2. ', ' ') \
            .replace(' 3. ', ' ').replace(' 4. ', ' ').replace(' 5. ', ' ').replace(' 6. ', ' ').replace(' 7. ', ' ').replace(' 8. ', ' ') .replace(' 9. ', ' ')   \
            .strip().lower() + ' ').split('. ')
        sent_cleaner = lambda t: re.sub('[#,?;*!^&_+():-\[\]{}]', '', t.replace('"', '').
                                    replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) 
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
