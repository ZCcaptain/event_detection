from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
from lib.utils.tools import load_dict
import os
from lib.dataloaders.ace_loader import ACE_Dataset, Custom_loader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import cm 
import seaborn as sns
import pickle

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def evaluation(model, data_loader, wdict, ydict, sents):
    # pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
    model.eval()
    count = 0
    with torch.no_grad():
        t_result = []
        for batch_ndx, sample in enumerate(data_loader):
            output = model(sample, is_train=False)
            word_id = sample.tokens_id[0].tolist()
            words = [wdict[i] for i in word_id]
            for i in range(output['predict'].shape[0]):
                pred = torch.nonzero(output['predict'][i, :]).squeeze().tolist()
                if not isinstance(pred, list):
                    pred = [pred]
                gold_ans = torch.nonzero(sample.event_id[i]).squeeze().tolist()
                if not isinstance(gold_ans, list):
                    gold_ans = [gold_ans]
                t_result.append((pred, gold_ans))
                if ' '.join(words) in sents and len(gold_ans) >= 1:
                    labels = [ydict[i] for i in gold_ans]
                    attention_prob_all = output['att_prob'].squeeze(0)
                    attention_prob = attention_prob_all[:,gold_ans].cpu().numpy()
                    
                    result = {'data':attention_prob, 'labels':labels, 'words':words}

                    print(pred, gold_ans, ' '.join(words)[:20])
                    f, ax = plt.subplots(figsize = (3, 2))
                    cmap  = cm.Blues
                    sns.heatmap(attention_prob, cmap = cmap, linewidths = 0.05, ax = ax, yticklabels = words, xticklabels=  labels   )
                    ax.set_xlabel('words')
                    ax.set_ylabel('event')
                    plt.yticks(fontsize=3)
                    # if ' '.join(words) in sents:
                    f.savefig('4/'+ ' '.join(words)[:20] + '.jpg',  dpi=500,bbox_inches='tight')
                    with open("4/data" + str(count) + ".pkl", 'wb') as f:
                        pickle.dump(result, f)
                    count += 1
    print(count)







if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    wdict = load_dict(os.path.join(hyper.data_root, 'ace/dicts/word_dict.txt'))
    wdict = {v:k for k, v in wdict.items()}
    ydict = load_dict(os.path.join(hyper.data_root, 'ace/dicts/label_dict.txt'))
    ydict = {v:k for k, v in ydict.items()}
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    model_path = 'data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/24_0.724.pt'
    load_model(model, model_path)
    sents = []
    with open('./multi_triggers_sent.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            sents.append(line)
    


    test_path = os.path.join(hyper.data_root, hyper.dataset, hyper.test)
    data_set = ACE_Dataset(hyper, test_path)
    data_loader = Custom_loader(data_set, batch_size=1, pin_memory=True)
    evaluation(model, data_loader, wdict, ydict, sents)
