"""Torch Module for Attention-based Graph Neural Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import functional as F
from torchsort import soft_rank, soft_sort

# from .... import function as fn
# from ..softmax import edge_softmax
# from ....utils import expand_as_pair
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

class AGNNConv(nn.Module):
    r"""Attention-based Graph Neural Network layer from paper `Attention-based
    Graph Neural Network for Semi-Supervised Learning
    <https://arxiv.org/abs/1803.03735>`__.

    .. math::
        H^{l+1} = P H^{l}

    where :math:`P` is computed as:

    .. math::
        P_{ij} = \mathrm{softmax}_i ( \beta \cdot \cos(h_i^l, h_j^l))

    Parameters
    ----------
    init_beta : float, optional
        The :math:`\beta` in the formula.
    learn_beta : bool, optional
        If True, :math:`\beta` will be learnable parameter.
    """
    def __init__(self,args,
                 init_beta=1.,
                 learn_beta=True):
        super(AGNNConv, self).__init__()
        self.args = args
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor([init_beta]))
        else:
            self.register_buffer('beta', th.Tensor([init_beta]))

    def forward(self, graph, feat):
        r"""Compute AGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *)` and :math:`(N_{out}, *})`, the the :math:`*` in the later
            tensor must equal the previous one.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        graph = graph.local_var()

        feat_src, feat_dst = expand_as_pair(feat)
        graph.srcdata['h'] = feat_src
        if self.args.att_type == "AGNN":
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple):
                graph.dstdata['norm_h'] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
            cos = graph.edata.pop('cos')
            e = self.beta * cos
        #SAME AS AGNN
        elif self.args.att_type == "cosine":
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple):
                graph.dstdata['norm_h'] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
            cos = graph.edata.pop('cos')
            e = self.beta * cos
        elif self.args.att_type == "scaled_dot":
            if isinstance(feat, tuple):
                graph.dstdata['h'] = feat_dst
            # compute dot
            graph.apply_edges(fn.u_dot_v('h', 'h', 'dot'))
            dot = graph.edata.pop('dot')
            e = self.beta * dot
        elif self.args.att_type == "pearson":
            src_mu = th.mean(feat_src, dim=1, keepdim=True)
            graph.srcdata['norm_h'] = F.normalize(feat_src - src_mu, p=2, dim=-1)
            if isinstance(feat, tuple):
                dst_mu = th.mean(feat_dst, dim=1, keepdim=True)
                graph.dstdata['norm_h'] = F.normalize(feat_dst - dst_mu, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
            cos = graph.edata.pop('cos')
            e = self.beta * cos
        elif self.args.att_type == "spearman":
            # F.normalize(feat_src, p=2, dim=1).detach().numpy()
            ranked_src = soft_rank(1000*F.normalize(feat_src, p=2, dim=-1))#, regularization_strength=0.1)
            src_mu = th.mean(ranked_src, dim=1, keepdim=True)
            graph.srcdata['norm_h'] = F.normalize(ranked_src - src_mu, p=2, dim=-1)
            if isinstance(feat, tuple):
                ranked_dst = soft_rank(1000*F.normalize(feat_dst, p=2, dim=-1), regularization_strength=0.1)
                dst_mu = th.mean(ranked_dst, dim=1, keepdim=True)
                graph.dstdata['norm_h'] = F.normalize(ranked_dst - dst_mu, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
            cos = graph.edata.pop('cos')
            e = self.beta * cos

        graph.edata['p'] = edge_softmax(graph, e)
        graph.update_all(fn.u_mul_e('h', 'p', 'm'), fn.sum('m', 'h'))
        return graph.dstdata.pop('h')