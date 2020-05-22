from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
import os
from lib.dataloaders.ace_loader import ACE_Dataset, Custom_loader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def get_att_vec(model, data_loader):
    pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
    model.eval()
    with torch.no_grad():
        all_vectors = []
        all_labels = []
        for batch_ndx, sample in pbar:
            output = model(sample, is_train=False)
            for i in range(output['att_vec'].shape[0]):
                att_vec = output['att_vec'][i]
                gold_ans = torch.nonzero(sample.event_id[i]).squeeze().tolist()
                if not isinstance(gold_ans, list):
                    gold_ans = [gold_ans]
                if gold_ans[0] != 26 and len(gold_ans) > 1 :
                    all_labels += gold_ans
                    vectors = att_vec[gold_ans, :].cpu().numpy()
                    all_vectors.append(vectors)

        tmp = all_vectors[0]
        for av in all_vectors[1:]:
            tmp = np.concatenate((tmp, av))
            # tmp.concatenate(av)
    return tmp, all_labels

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
   """
    token = ['go', 'mo', 'bo', 'co','m.']
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min) # 对数据进行归一化处理
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, 35))

    fig = plt.figure() # 创建图形实例
    ax = plt.subplot(111)# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=colors[label[i]])
        # plt.scatter(data[i, 0], data[i, 1], 'g.')
        # plt.text(data[i, 0], data[i, 1], str(label[i]), fontdict={'weight': 'bold', 'size': 6})
    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=plt.cm.Set1(label[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

    plt.xticks() # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值

    return fig

def draw_weight(weight, label):
    ts = TSNE(n_components=2,init = 'pca',random_state = 100)
    # ts = TSNE(n_components=2)
    result = ts.fit_transform(weight)
    fig = plot_embedding(result, label, 't_SNE Embedding of digits')
    # plt.show()

    plt.savefig('./sne.png', dpi=500, bbox_inches='tight')

if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    model_path = 'data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/24_0.724.pt'
    load_model(model, model_path)

    test_path = os.path.join(hyper.data_root, hyper.dataset, hyper.test)
    data_set = ACE_Dataset(hyper, test_path)
    data_loader = Custom_loader(data_set, batch_size=100, pin_memory=True)

    all_vectors, all_labels = get_att_vec(model, data_loader)
    draw_weight(all_vectors, all_labels)
    pass
