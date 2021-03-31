#code adapted from
# https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/gat
"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.utils import split_dataset
from models import GAT, AGNN
from utils import EarlyStopping


def get_dataset(opt):
    # load and preprocess dataset
    if opt['dataset'] == 'cora':
        data = CoraGraphDataset()
    elif opt['dataset'] == 'citeseer':
        data = CiteseerGraphDataset()
    elif opt['dataset'] == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(opt['dataset']))
    return data


def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

@torch.no_grad()
def test(model, g):
  model.eval()
  features = g.ndata['feat']
  logits = model(features)
  accs = []
  for mask_type in ['train_mask', 'val_mask', 'test_mask']:
    mask = g.ndata[mask_type]
    labels = g.ndata['label'][mask]
    acc = accuracy(logits[mask],labels)
    accs.append(acc)
  return accs

def train(model, optimizer, features, train_mask, labels):
    loss_fcn = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    logits = model(features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, logits


def main(opt):
    data = get_dataset(opt)
    g = data[0]
    if opt['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(opt['gpu'])

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
    heads = ([opt['num_heads']] * opt['num_layers']) + [opt['num_out_heads']]
    if opt['model'] == 'GAT':
        model = GAT(g,
                    opt['num_layers'],
                    num_feats,
                    opt['num_hidden'],
                    n_classes,
                    heads,
                    F.elu,
                    opt['in_drop'],
                    opt['attn_drop'],
                    opt['negative_slope'],
                    opt['residual'],
                    opt)
    elif opt['model'] == 'AGNN':
        model = AGNN(g,
                     opt['num_layers'],
                     num_feats,
                     opt['num_hidden'],
                     n_classes,
                     opt['in_drop'],
                     opt)
    print(model)
    if opt['early_stop']:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = get_optimizer(opt['optimizer'], parameters=model.parameters(),
                                lr=opt['lr'], weight_decay=opt['weight_decay'])

    # initialize graph
    dur = []
    for epoch in range(opt['epochs']):
        # model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        # logits = model(features)
        # loss = loss_fcn(logits[train_mask], labels[train_mask])
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        loss, logits = train(model, optimizer, features, train_mask, labels)

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if opt['fastmode']:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if opt['early_stop']:
                if stopper.step(val_acc, model):
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if opt['early_stop']:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

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
    parser.add_argument("--model", type=str, default="GAT",
                        help="AGNN,GAT")
    parser.add_argument("--att-type", type=str, default="spearman",
                        help="AGNN,GAT,cosine,scaled_dot,pearson,spearman")

    args = parser.parse_args()
    print(args)
    opt = vars(args)
    main(opt)

