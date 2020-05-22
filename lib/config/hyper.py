from dataclasses import dataclass
import json
import os


@dataclass
class Hyper(object):
    def __init__(self, path):
        # self.emb_dim: int
        # self.max_l: int
        # self.n_class: int
        # self.n_ent: int
        # self.dim_ent: int
        # self.l2_weight: float
        # self.epoch_num: int
        # self.alpha: float
        # self.batch_size: int
        # self.gpu: int

        self.__dict__ = json.load(open(path, 'r'))


if __name__ == "__main__":
    hyper = Hyper('/home/zengdj/workspace/event_detection/experiments/naacl.json')
    print(hyper)