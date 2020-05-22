from lib.config.hyper import Hyper
from lib.dataloaders.ace_loader import ACE_Dataset, Custom_loader
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from lib.models.rethink import RethinkNaacl
import torch
from torch.optim import Adam


def save_model(model, model_dir, epoch):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, str(epoch) + '.pt'))

def evaluation(model, eval_type, epoch, hyper):
    if eval_type.lower() == 'train':
        train_path = os.path.join(hyper.data_root, hyper.dataset, hyper.train)
        data_set = ACE_Dataset(hyper, train_path)
        label_out = 'Train # '
        loader = Custom_loader(data_set, batch_size=hyper.eval_batch, pin_memory=True)
    elif eval_type.lower() == 'test':
        test_path = os.path.join(hyper.data_root, hyper.dataset, hyper.test)
        data_set = ACE_Dataset(hyper, test_path)
        label_out = 'Test # '
        loader = Custom_loader(data_set, batch_size=len(data_set.data_loaded[0]), pin_memory=True)

    pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))
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
    optimizer = Adam(model.parameters(), weight_decay=hyper.L2_weight)
    # optimizer = Adam(model.parameters())
    train_path = os.path.join(hyper.data_root, hyper.dataset, hyper.train)
    train_set = ACE_Dataset(hyper, train_path)
    loader = Custom_loader(train_set, batch_size=hyper.batch_size, pin_memory=True, shuffle=True)

    max_f1 = 0
    for epoch in range(hyper.epoch_num):
        model.train()
        pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                    total=len(loader))
        all_loss = 0
        for batch_idx, sample in pbar:
            optimizer.zero_grad()
            # print(sample)
            output = model(sample)
            loss = output['loss']

            # for param in model.parameters():
            #     loss += torch.sum(torch.abs(param))*hyper.L1_weight

            loss.backward()
            optimizer.step()
            pbar.set_description(output['description'](
                epoch, hyper.epoch_num))
            all_loss += loss
        print('Epoch %d --- All loss = %f' % (epoch, all_loss))
        # eval
        eval_out = evaluation(model, 'train', epoch, hyper)
        print("Train###" + eval_out[0])
        eval_out = evaluation(model, 'test', epoch, hyper)
        print("Test###" + eval_out[0])
        print("==================================================================")

        if hyper.save_model != 'no':
            if hyper.save_model == 'max':
                if eval_out[1] > max_f1 and eval_out[1] > hyper.save_model_threshold:
                    save_model(model, os.path.join(hyper.data_root, hyper.dataset, hyper.model_dir),
                               str(epoch)+'_'+str(round(eval_out[1], 4)))
            else:
                save_model(model, os.path.join(hyper.data_root, hyper.dataset, hyper.model_dir),
                           str(epoch)+'_f1_'+str(round(eval_out[1], 4)))


