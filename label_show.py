from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib


# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    token = ['go', 'mo', 'bo', 'co','m.']
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) # 对数据进行归一化处理
    fig = plt.figure() # 创建图形实例
    ax = plt.subplot(111)# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], 'go')
        plt.text(data[i, 0], data[i, 1], str(label[i]), fontdict={'weight': 'bold', 'size': 6})
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
    # ts = TSNE(n_components=2,init = 'pca',random_state = 100)
    ts = TSNE(n_components=2)
    result = ts.fit_transform(weight)
    fig = plot_embedding(result, label, 't_SNE Embedding of digits')
    plt.show()
    plt.savefig('./sne.png', dpi=500, bbox_inches='tight')

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    model_path = '/home/zengdj/workspace/event_detection/data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/45_0.751.pt'
    load_model(model, model_path)
    model.eval()

    # weight = model.evt_embedding.weight.cpu().detach().numpy()[1:,:]

    x = model.gc1(model.evt_embedding.weight, model.adj)
    x = model.relu(x)
    x = model.gc2(x, model.adj)

    weight = x.cpu().detach().numpy()[1:,:]

    label = [l for l in model.ydict]

    np.save('weigth.npy', weight)
    np.save('label.npy', label)


    draw_weight(weight, label)
