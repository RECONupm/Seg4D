# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""
#%% IMPORT LIBRARIES
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU
from torch_geometric.nn.unpool import knn_interpolate
from torch.nn import Identity
from torch_geometric.nn.pool import knn
from torch_geometric.nn import global_mean_pool
from torch_cluster import knn_graph
from torch_cluster import fps
from torch_scatter import scatter_max
from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch.nn import Linear as Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import reset


#%% DEFINE CLASSES
class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper

    Inputs (from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/point_transformer_conv.html):
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        pos_nn : (torch.nn.Module, optional): A neural network
            :math:`h_\mathbf{\Theta}` which maps relative spatial coordinates
            :obj:`pos_j - pos_i` of shape :obj:`[-1, 3]` to shape
            :obj:`[-1, out_channels]`.
            Will default to a :class:`torch.nn.Linear` transformation if not
            further specified. (default: :obj:`None`)
        attn_nn : (torch.nn.Module, optional): A neural network
            :math:`\gamma_\mathbf{\Theta}` which maps transformed
            node features of shape :obj:`[-1, out_channels]`
            to shape :obj:`[-1, out_channels]`. (default: :obj:`None`)
        add_self_loops (bool, optional) : If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, share_planes: int = 8, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.share_planes = share_planes

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(3, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels)
        self.lin_src = Linear(in_channels[0], out_channels)
        self.lin_dst = Linear(in_channels[1], out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        delta = self.pos_nn(pos_j - pos_i)
        alpha = alpha_j - alpha_i + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return (alpha.unsqueeze(1) * (x_j + delta).view(
            -1, self.share_planes, x_j.shape[1] // self.share_planes)).view(
                -1, x_j.shape[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels, bias=False)
        self.lin_out = Lin(out_channels, out_channels, bias=False)

        self.pos_nn = Seq(MLP([3, 3]), Lin(3, out_channels))

        self.attn_nn = Seq(BN(out_channels), ReLU(),
                           MLP([out_channels, out_channels // 8]),
                           Lin(out_channels // 8, out_channels // 8))

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn,
                                                add_self_loops=False)

        self.bn1 = BN(in_channels)
        self.bn2 = BN(in_channels)
        self.bn3 = BN(in_channels)

    def forward(self, x, pos, edge_index):
        x_skip = x.clone()
        x = self.bn1(self.lin_in(x)).relu()
        x = self.bn2(self.transformer(x, pos, edge_index)).relu()
        x = self.bn3(self.lin_out(x))
        x = (x + x_skip).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([3 + in_channels, out_channels], bias=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        relative_pos = pos[id_k_neighbor[1]] - pos[id_clusters][
            id_k_neighbor[0]]

        # get neighbors features and add relative positions as features
        x = torch.cat([relative_pos, x[id_k_neighbor[1]]], axis=1)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x, id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True, bias=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class TransitionSummit(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp_sub = Seq(Lin(in_channels, in_channels), ReLU())
        self.mlp = MLP([2 * in_channels, in_channels])

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # compute the mean of features batch_wise
        x_mean = global_mean_pool(x, batch=batch)
        x_mean = self.mlp_sub(x_mean)  # (batchs, features)

        # reshape back to (N_points, features)
        counts = batch.unique(return_counts=True)[1]
        x_mean = torch.cat(
            [x_mean[i].repeat(counts[i], 1) for i in range(x_mean.shape[0])],
            dim=0)

        # transform features
        x = self.mlp(torch.cat((x, x_mean), 1))
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], bias=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        blocks = [1, 2, 3, 5, 2]

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(
                Seq(
                    TransitionDown(in_channels=dim_model[i],
                                   out_channels=dim_model[i + 1], k=self.k),
                    *[
                        TransformerBlock(in_channels=dim_model[i + 1],
                                         out_channels=dim_model[i + 1])
                        for k in range(blocks[1:][i])
                    ]))

        # summit layers
        self.mlp_summit = TransitionSummit(dim_model[-1])

        self.transformer_summit = Seq(*[
            TransformerBlock(
                in_channels=dim_model[-1],
                out_channels=dim_model[-1],
            ) for i in range(1)
        ])

        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(
                Seq(
                    TransitionUp(in_channels=dim_model[n - i],
                                 out_channels=dim_model[n - i - 1]),
                    *[
                        TransformerBlock(in_channels=dim_model[n - i - 1],
                                         out_channels=dim_model[n - i - 1])
                        for k in range(1)
                    ]))

        # class score computation
        self.mlp_output = Seq(MLP([dim_model[0], dim_model[0]]),
                              Lin(dim_model[0], out_channels))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        # add dummy features in case there is none
        x = torch.ones((pos.shape[0],1)).to(pos.device) if x is None else x #torch.cat((pos, x), 1)

        out_x = []
        out_pos = []
        out_batch = []
        edges_index = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)
        edges_index.append(edge_index)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):

            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
            for layer in self.encoders[i][1:]:
                x = layer(x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
            edges_index.append(edge_index)

        # summit
        x = self.mlp_summit(x, batch=batch)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        for layer in self.transformer_summit:
            x = layer(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                    pos=out_pos[-i - 2],
                                    pos_sub=out_pos[-i - 1],
                                    batch_sub=out_batch[-i - 1],
                                    batch=out_batch[-i - 2])

            edge_index = edges_index[-i - 2]

            for layer in self.decoders[i][1:]:
                x = layer(x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)
