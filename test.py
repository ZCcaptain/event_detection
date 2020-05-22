from lib.config.hyper import Hyper
import torch
from lib.models.rethink import RethinkNaacl
import os
from lib.dataloaders.ace_loader import ACE_Dataset, Custom_loader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def evaluation(model, data_loader):
    pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
    model.eval()
    with torch.no_grad():
        t_result = []
        for batch_ndx, sample in pbar:
            output = model(sample, is_train=False)
            for i in range(output['predict'].shape[0]):
                pred = torch.nonzero(output['predict'][i, :]).squeeze().tolist()
                if not isinstance(pred, list):
                    pred = [pred]
                gold_ans = torch.nonzero(sample.event_id[i]).squeeze().tolist()
                if not isinstance(gold_ans, list):
                    gold_ans = [gold_ans]
                t_result.append((pred, gold_ans))
        out, f1 = evaluate_results_binary(t_result, 26)
        return out, f1

def evaluate_results_binary(result, neg_id):
    total_p, total_g, right, total, total_right = 0, 0, 0, 0, 0
    for _p, g in result:
        if len(_p) < 1:
            _p = [neg_id]
        total += len(_p)
        if g[0] != neg_id:
            total_g += len(g)
        for p in _p:
            if p != neg_id:
                total_p += 1
            if p in g:
                total_right += 1
            if p != neg_id and p in g:
                right += 1
    if total_p == 0:
        total_p = 1
    acc = 1.0 * total_right / total
    pre = 1.0 * right / total_p
    rec = 1.0 * right / total_g
    f1 = 2 * pre * rec / (pre + rec + 0.000001)
    out = 'Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (total, total_p, total_right, right, total_g)
    out += 'Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
    return out, f1




if __name__ == '__main__':
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    device = torch.device("cuda:%d" % hyper.gpu if torch.cuda.is_available() else "cpu")
    model = RethinkNaacl(hyper).to(device)
    # model_path = 'data/ace/saved_models/max_model_tmp/22_0.7183.pt'
    model_path = 'data/ace/saved_models/e9e28cbe7498c1808bbcf6c43d6f041e68d3f3be/24_0.724.pt'
    load_model(model, model_path)

    test_path = os.path.join(hyper.data_root, hyper.dataset, hyper.test)
    data_set = ACE_Dataset(hyper, test_path)
    data_loader = Custom_loader(data_set, batch_size=len(data_set.data_loaded[0]), pin_memory=True)

    eval_out = evaluation(model, data_loader)
    print("Test###" + eval_out[0])
