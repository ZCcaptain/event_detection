from overrides import overrides
import torch

class F1_abc(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset=False):
        if reset:
            self.reset()
        f1, p, r = 2 * self.A / (self.B +
                                 self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}
        return result

    def __call__(self, predictions,
                 gold_labels):
        raise NotImplementedError

class EventF1(F1_abc):

    @overrides
    def __call__(self, predictions, gold_labels):
        idx = torch.nonzero(predictions)
        pass