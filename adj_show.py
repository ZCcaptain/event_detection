from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
import numpy as np

import matplotlib.pyplot as plt

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def normalization(datingDatamat):
   max_arr = datingDatamat.max(axis=0)
   min_arr = datingDatamat.min(axis=0)
   ranges = max_arr - min_arr
   norDataSet = np.zeros(datingDatamat.shape)
   m = datingDatamat.shape[0]
   norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
   norDataSet = norDataSet/np.tile(ranges,(m,1))
   return norDataSet

if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    model_path = '/home/zengdj/workspace/event_detection/data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/45_0.751.pt'
    load_model(model, model_path)
    model.eval()

    adj = model.adj.cpu().detach().numpy()[1:,:]
    adj = normalization(adj)
    # plt.matshow(adj, cmap='gray')
    plt.matshow(adj)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("test.jpg")
    plt.show()
