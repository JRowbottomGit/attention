import argparse
import os
import time
from functools import partial
import dgl
from dgl.data import register_data_args

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models import GAT, AGNN

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from train import train, evaluate, test, get_dataset, get_optimizer


def average_test(models, datas):
    results = [test(model, data) for model, data in zip(models, datas)]
    train_accs, val_accs, tmp_test_accs = [], [], []

    for train_acc, val_acc, test_acc in results:
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        tmp_test_accs.append(test_acc)

    return train_accs, val_accs, tmp_test_accs


def train_ray_rand(opt, checkpoint_dir=None, data_dir="../data"):
    pass
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   dataset = get_dataset(opt, data_dir, opt['not_lcc'])
#
#   models = []
#   datas = []
#   optimizers = []
#
#   for split in range(opt["num_splits"]):
#     dataset.data = set_train_val_test_split(
#       np.random.randint(0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)
#     datas.append(dataset.data)
#
#     model = GNN(opt, dataset, device)
#     train_this = train
#
#     models.append(model)
#
#     if torch.cuda.device_count() > 1:
#       model = nn.DataParallel(model)
#
#     model, data = model.to(device), dataset.data.to(device)
#     parameters = [p for p in model.parameters() if p.requires_grad]
#
#     optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
#     optimizers.append(optimizer)
#
#     # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
#     # should be restored.
#     if checkpoint_dir:
#       checkpoint = os.path.join(checkpoint_dir, "checkpoint")
#       model_state, optimizer_state = torch.load(checkpoint)
#       model.load_state_dict(model_state)
#       optimizer.load_state_dict(optimizer_state)
#
#   for epoch in range(1, opt["epoch"]):
#     loss = np.mean([train_this(model, optimizer, data) for model, optimizer, data in zip(models, optimizers, datas)])
#     train_accs, val_accs, tmp_test_accs = average_test(models, datas)
#     with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
#       best = np.argmax(val_accs)
#       path = os.path.join(checkpoint_dir, "checkpoint")
#       torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
#     tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs), train_acc=np.mean(train_accs),
#                 forward_nfe=model.fm.sum,
#                 backward_nfe=model.bm.sum)


def train_ray(args, checkpoint_dir=None, data_dir="../data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_dataset(args)
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]

    models = []
    optimizers = []
    datas = [g for i in range(args.num_init)]

    for split in range(args.num_init):
        if args.model == 'GAT':
            model = GAT(g,
                        args.num_layers,
                        num_feats,
                        args.num_hidden,
                        n_classes,
                        heads,
                        F.elu,
                        args.in_drop,
                        args.attn_drop,
                        args.negative_slope,
                        args.residual)
        elif args.model == 'AGNN':
            model = AGNN(g,
                         args.num_layers,
                         num_feats,
                         args.num_hidden,
                         n_classes,
                         args.in_drop,
                         args)
        train_this = train

        model = model.to(device)
        models.append(model)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # model = model.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]

        optimizer = get_optimizer(args.optimizer, parameters, lr=args.lr, weight_decay=args.decay)
        optimizers.append(optimizer)

        # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
        # should be restored.
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(1, args.epoch):
        loss = np.mean([train_this(model, optimizer, data)[0] for model, optimizer in zip(models, optimizers)])
        train_accs, val_accs, tmp_test_accs = average_test(models, datas)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            best = np.argmax(val_accs)
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
        tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs),
                    train_acc=np.mean(train_accs))


def set_cora_search_space(opt):
    args.decay = tune.loguniform(0.001, 0.1)  # weight decay l2 reg
    args.hidden_dim = tune.sample_from(lambda _: 2 ** np.random.randint(6, 8))  # hidden dim of X in dX/dt
    opt["lr"] = tune.uniform(0.01, 0.2)
    opt["input_dropout"] = 0.5
    opt["optimizer"] = tune.choice(["adam", "adamax"])
    opt["dropout"] = tune.uniform(0, 0.15)  # output dropout
    # opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))  #
    # opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  # hidden dim for attention
    return opt


def set_pubmed_search_space(opt):
    pass
    return opt


def set_citeseer_search_space(opt):
    pass
    return opt


def set_search_space(opt):
    if opt["dataset"] == "Cora":
        return set_cora_search_space(opt)
    elif opt["dataset"] == "Pubmed":
        return set_pubmed_search_space(opt)
    elif opt["dataset"] == "Citeseer":
        return set_citeseer_search_space(opt)
        return set_arxiv_search_space(opt)


def main(opt):
    data_dir = os.path.abspath("../data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = set_search_space(opt)
    # scheduler = ASHAScheduler(
    #   metric=opt['metric'],
    #   mode="max",
    #   max_t=opt["epoch"],
    #   grace_period=opt["grace_period"],
    #   reduction_factor=opt["reduction_factor"],
    # )

    scheduler = FIFOScheduler()
    reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe",
                        "backward_nfe"]
    )
    # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    # search_alg = AxSearch(metric=opt['metric'])
    search_alg = None

    train_fn = train_ray if opt["num_splits"] == 0 else train_ray_rand

    result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=opt["name"],
        resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=opt['metric'],
        config=opt,
        num_samples=opt["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="'sgd','rmsprop','adagrad','adam','adamax'")
    parser.add_argument("--model", type=str, default="AGNN",
                        help="AGNN,GAT")
    parser.add_argument("--att-type", type=str, default="pearson",
                        help="AGNN,cosine,scaled_dot,pearson,spearman")
    parser.add_argument("--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed")

    # ray args
    parser.add_argument("--num_samples", type=int, default=20, help="number of ray trials")
    parser.add_argument("--gpus", type=float, default=0, help="number of gpus per trial. Can be fractional")
    parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
    parser.add_argument(
        "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials")
    parser.add_argument(
        "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs")
    parser.add_argument("--name", type=str, default="ray_exp")
    parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
    parser.add_argument("--num_init", type=int, default=4, help="Number of random initializations >= 0")

    args = parser.parse_args()
    print(args)
    opt = vars(args)
    main(opt)
