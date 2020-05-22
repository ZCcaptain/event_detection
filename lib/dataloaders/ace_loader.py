from torch.utils.data import Dataset
from lib.utils.tools import load_dict
from lib.utils.tools import load_embedding
from numpy import array
from functools import partial
from torch.utils.data.dataloader import DataLoader
import os
import torch


class ACE_Dataset(Dataset):
    def __init__(self, hyper, data_path):
        self.hyper = hyper
        self.data_root = hyper.data_root
        wdict = load_dict(os.path.join(self.data_root, 'ace/dicts/word_dict.txt'))
        edict = load_dict(os.path.join(self.data_root, 'ace/dicts/ent_dict.txt'))
        ydict = load_dict(os.path.join(self.data_root, 'ace/dicts/label_dict.txt'))
        ydict = {k.lower(): v for k, v in ydict.items()}
        # self.word_emb = load_embedding(os.path.join(self.data_root, 'ace/embeddings/200.txt'))
        self.data_loaded = self._load_data_ent(data_path, wdict, edict, ydict, hyper.max_l)
        #data_loaded存放格式  sen, ent, y = [], [], []
        print('done')
        # label_num = [len(d) if len(d) > 1 else 0 for d in self.data_loaded[2]]
        # print(sum(label_num) / len(label_num)) #计算一个句子中超过一种事件的比例
        # pass

    def _load_data_ent(self, data_path, wdict, edict, ydict, max_len):
        sen, ent, y = [], [], []
        for line in open(data_path):
            line = line.strip()
            if not line:
                continue
            if len(line.split('\t')) < 3:
                continue
            wds = line.split('\t')[:-1]
            wds, ents = zip(*[x.split(' ') for x in wds])
            ls = line.split('\t')[-1].strip().lower().split(' ')
            words, ents, labels = self._sent2array_ent((wds, ents, ls), wdict, edict, ydict, max_len)
            sen.append(words)
            ent.append(ents)
            y.append(labels)
        # return [array(sen, dtype='int32'), array(ent, dtype='int32'), y]
        return [sen, ent, y]

    def _sent2array_ent(self, sent, wdict, edict, ydict, max_len):
        words = list(map(lambda x: wdict.get(x, wdict['OTHER-WORDS-ID']), sent[0]))
        ents = list(map(lambda x: edict.get(x, edict['NEGATIVE']), sent[1]))
        MAX_SEN_LEN = max_len
        # if len(words) < MAX_SEN_LEN:
        #     words += ([-1] * (MAX_SEN_LEN - len(words)))
        #     ents += ([edict['NEGATIVE']] * (MAX_SEN_LEN - len(ents)))
        if len(words) > MAX_SEN_LEN:
            words = words[:MAX_SEN_LEN]
            ents = ents[:MAX_SEN_LEN]
        labels = [ydict.get(x.lower(), 'negative') for x in sent[2]]
        return words, ents, labels

    def __len__(self):
        return len(self.data_loaded[0])

    def label_hot(self, y):
        gold_hot = torch.zeros(self.hyper.n_class)
        gold_hot[y] = 1
        return gold_hot

    def __getitem__(self, item):
        tokens = torch.tensor(self.data_loaded[0][item])
        entity_tag = torch.tensor(self.data_loaded[1][item])
        event = self.data_loaded[2][item]
        token_len = tokens.shape[0]
        event = self.label_hot(event)
        return tokens, entity_tag, event, token_len

class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip(*data))
        self.tokens_id = transposed_data[0]
        self.entity_tag = transposed_data[1]
        self.event_id = transposed_data[2]
        self.token_len = transposed_data[3]

def collate_fn(batch):
    return Batch_reader(batch)


Custom_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)