import numpy as np
import torch
import scipy
from scipy.stats import sem


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval