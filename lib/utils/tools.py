import numpy as np
from numpy import array


def load_dict(path):
    '''
    word_dict
    label_dict
    entype_dict
    ret: {word1:id1, word2:id2,...}
    '''
    fin = open(path)
    ret = {}
    for idx, line in enumerate(fin):
        if not line.strip(): continue
        ret[line.strip()] = (idx + 1)  # id from 1, 0 is for empty word/label/entype
    return ret


def _load_random_embeddings(dim, word_dict_p):
    word_dict = load_dict(word_dict_p)
    larggest_id = word_dict['OTHER-WORDS-ID']
    rng = np.random.RandomState(3135)
    return rng.uniform(low=-0.5, high=0.5, size=(larggest_id + 1, dim))

def load_embedding(path):
    '''
    load word embedding or
    position embedsing or
    label embedding
    '''
    if path is None:
        return _load_random_embeddings(100, 'data/dicts/word_dict.txt')
    fin = open(path)
    data = []
    for line in fin:
        if not line.strip(): continue
        data.append(list(map(float, line.strip().split(' '))))
    return array(data, dtype='float32')
    #return array(data)