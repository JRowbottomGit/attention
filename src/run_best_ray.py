import argparse
import json
import numpy as np
from functools import partial
import os, time
from ray import tune
from ray.tune import Analysis
from ray.tune import CLIReporter

from ray_tune import train_ray_int
from utils import get_sem, mean_confidence_interval


def loop_best(opt):
  models = ['AGNN','GAT']
  layers = [1,2]#,4,8,16]
  att_type_AGNN = ['AGNN','cosine','scaled_dot','pearson','spearman']
  att_type_GAT = ['GAT','cosine','scaled_dot','pearson','spearman']
  for model in models:
    for layer in layers:
      for att_type in att_type_AGNN if model == 'AGNN' else att_type_GAT:
        best_params_dir = get_best_specific_params_dir(opt, layer, model, att_type)
        with open(best_params_dir + '/params.json') as f:
          best_params = json.loads(f.read())
        # allow params specified at the cmd line to override
        best_params_ret = {**best_params, **opt}
        # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
        best_params_ret['epoch'] = opt['epoch']

        print("Running with parameters {}".format(best_params_ret))

        data_dir = os.path.abspath("../data")
        reporter = CLIReporter(
          metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch", "training_iteration"])

        if opt['name'] is None:
          name = opt['folder'] + '_test'
        else:
          name = opt['name']

        result = tune.run(
          partial(train_ray_int, data_dir=data_dir),
          name=name,
          resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
          search_alg=None,
          keep_checkpoints_num=3,
          checkpoint_score_attr='accuracy',
          config=best_params_ret,
          num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
          scheduler=None,
          max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
          local_dir='../ray_tune',
          progress_reporter=reporter,
          raise_on_failed_trial=False)

        df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)
        df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")), mode='a')

        # try:
        #   df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")), mode='a')
        # except:
        #   pass

        print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

        test_accs = df['test_acc'].values
        print("test accuracy {}".format(test_accs))
        log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
        print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


def get_best_specific_params_dir(opt, layer, model, att_type):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric=opt['metric'], mode='max')
  print(df)
  df.columns = [c.replace('config/', '') for c in df.columns]
  print(df)
  # newdf = df.loc[(df.num_layers == layers) & (df.model == model) & (df.att_type == att_type)]
  print(layers)
  print(model)
  print(att_type)
  newdf = df.loc[(df['num_layers'] == layer) & (df['model'] == model) & (df['att_type'] == att_type)]

  best_params_dir = newdf.sort_values('accuracy', ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir

def get_best_params_dir(opt):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric=opt['metric'], mode='max')
  best_params_dir = df.sort_values('accuracy', ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir

def run_best_params(opt):
  best_params_dir = get_best_params_dir(opt)
  with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())
  # allow params specified at the cmd line to override
  best_params_ret = {**best_params, **opt}
  # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
  best_params_ret['epoch'] = opt['epoch']

  print("Running with parameters {}".format(best_params_ret))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch", "training_iteration"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray_int, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr='accuracy',
    config=best_params_ret,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)
  try:
    df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
  except:
    pass

  print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

  test_accs = df['test_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")

  args = parser.parse_args()

  opt = vars(args)
  # run_best_params(opt)
  loop_best(opt)