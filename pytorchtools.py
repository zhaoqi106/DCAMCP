import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.meter_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, meter, model):

        score = meter

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(meter, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(meter, model)
            self.counter = 0

    def save_checkpoint(self, meter, model):
        if self.verbose:
            self.trace_func(f'meter decreased ({self.meter_max:.6f} --> {meter:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.meter_max = meter






