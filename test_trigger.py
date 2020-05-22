from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
import os
from lib.dataloaders.ace_loader_trigger import ACE_Dataset_Trigger, Custom_trigger_loader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def evaluate_results_binary(event_list, att_prob, predict, neg_id=26, topK=1):
    TP, FP, FN = 0, 0, 0
    TP_FN = 0
    for i in range(len(event_list)):
        event = event_list[i]
        prob = att_prob[i]
        pred = predict[i]
        pred = torch.nonzero(pred).squeeze().tolist()
        if not isinstance(pred, list):
            pred = [pred]

        positive_label = [e for e in event if e != neg_id]
        positive_label = list(set(positive_label))
        TP_FN += len(positive_label)
        #处理预测结果，计算TP（True Positive）和FP（False Positive）,计算预测的正样本中的真阳性和假阳性
        for p in pred:
            if p != neg_id:
                # 判断预测事件是否再event_list中, 如果不再就不处理
                if p in event:
                    # 判断attention的tigger是否和给定位置的相同
                    # 再prob中找topK位置
                    prob_values, indices = prob[:, p].topk(topK,largest=True, sorted=True)
                    # gold_indices = event.index(p)
                    gold_indices = [i for i, v in enumerate(event) if v == p]
                    for gi in gold_indices:
                        if gi in indices:
                            TP += 1
                    # TP += 1
                else:
                    FP += 1
            else:
                # 预测negative，实际上是positive. 计算FN（False Negative）
                if len(positive_label) > 0:
                    FN += 1
    prec = 1.0*TP/(TP+FP)
    rec = 1.0*TP/TP_FN
    f1 = 2 * prec * rec / (prec + rec + 0.000001)
    return prec, rec, f1

# TODO 发现一个句子有多个相同的关系，除去这种情况比较一下效果，另外，对这种情况需要比较一下多个attention的值是否差不多
def evaluate_results_right(event_list, att_prob, predict, neg_id=26, topK=1):
    TP, FP, FN = 0, 0, 0
    right = 0
    for i in range(len(event_list)):
        event = event_list[i]
        prob = att_prob[i]
        pred = predict[i]
        pred = torch.nonzero(pred).squeeze().tolist()
        if not isinstance(pred, list):
            pred = [pred]

        #处理预测结果, 计算TP（True Positive）和FP（False Positive）,计算预测的正样本中的真阳性和假阳性
        for p in pred:
            if p != neg_id:
                # 判断预测事件是否在event_list中, 如果不再就不处理
                if p in event:
                    right += 1
                    # 判断attention的tigger是否和给定位置的相同
                    # 再prob中找topK位置
                    prob_values, indices = prob[:, p].topk(topK, largest=True, sorted=True)
                    gold_indices = [i for i, v in enumerate(event) if v == p]
                    for gi in gold_indices:
                        if gi in indices:
                            TP += 1
    prec = 1.0 * TP/(right+0.0000001)
    return prec


if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")

    test_path = os.path.join(hyper.data_root, hyper.dataset, hyper.test_triger)
    data_set = ACE_Dataset_Trigger(hyper, test_path)
    data_loader = Custom_trigger_loader(data_set, batch_size=len(data_set.data_loaded[0]), pin_memory=True)
    # data_loader = Custom_trigger_loader(data_set, batch_size=5, pin_memory=True)
    pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))

    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    # model_path = 'data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/45_0.751.pt'
    model_path = 'data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/70_0.7033.pt'

    load_model(model, model_path)
    model.eval()
    predict, att_prob, event_list = [], [], []
    with torch.no_grad():
        for batch_ndx, sample in pbar:
            output = model(sample, is_train=False)
            # event = sample.event_id
            event_list += sample.event_id
            # att_prob = output['att_prob']
            att_prob += output['att_prob']
            # predict = output['predict']
            predict += output['predict']
    prec, rec, f1 = evaluate_results_binary(event_list, att_prob, predict, topK=1)
    print("precision=%f, recall=%f, f1=%f" % (prec, rec, f1))

    prec = evaluate_results_right(event_list, att_prob, predict, topK=1)
    print("precision=%f" % prec)
