from lib.utils.tools import load_dict
import os
from lib.config.hyper import Hyper
from collections import OrderedDict
from operator import itemgetter
from itertools import groupby

if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    ydict = load_dict(os.path.join(hyper.data_root, 'ace/dicts/label_dict.txt'))

    train_path = os.path.join(hyper.data_root, hyper.dataset, 'corpus_trigger_test.txt')
    all_num = 0
    multi_word_num = 0
    sents = []
    with open('./multi_triggers_sent.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            sents.append(line)
    for line in open(train_path):
        line = line.strip()
        if not line:
            continue
        if len(line.split('\t')) < 3:
            continue
        all_num += 1
        line_wds = line.split('\t')
        wds, ents, event = zip(*[x.split(' ') for x in line_wds])
        lst = []
        for x in line_wds:
            line_token_dict = {'word': x.split(' ')[0], 'ent_tag': x.split(' ')[1], 'event_y': x.split(' ')[2]}
            lst.append(line_token_dict)
        for key, group in groupby(lst, itemgetter('event_y')):
            # print(key)
            # print(list(group))
            if key != 'NEGATIVE' and len(list(group)) > 1:
                if ' '.join(wds) in sents:
                    # print(key)
                    # print(wds)
                    w_dict = OrderedDict()
                    for i in range(len(wds)):
                        w_dict[wds[i]] = event[i]
                    print(w_dict)
                    print(' ')
                multi_word_num += 1
                # print(ents)
                # print(event)
                # print(line)

                pass

    print(all_num)
    print(multi_word_num)




