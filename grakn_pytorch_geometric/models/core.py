from typing import Callable, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from grakn_pytorch_geometric.models.embedding import Embedder


class KGCN(torch.nn.Module):
    """
    Not the real GKCN yet. Will fix create model with
    supersteps in a following commit.
    This quickly overfits due too large number of weights.
    """

    def __init__(
        self,
        node_types=None,
        edge_types=None,
        categorical_attributes=None,
        continuous_attributes=None,
        edge_output_size=3,
        node_output_size=3,
        latent_size=16,
    ):
        super(KGCN, self).__init__()

        node_types = node_types or []
        edge_types = edge_types or []
        categorical_attributes = categorical_attributes or []
        continuous_attributes = continuous_attributes or []

        self.node_embedder = Embedder(
            types=node_types,
            type_embedding_dim=5,
            attr_embedding_dim=6,
            categorical_attributes=categorical_attributes,
            continuous_attributes=continuous_attributes,
        )

        self.edge_embedder = Embedder(types=edge_types, type_embedding_dim=5)

        self.conv1 = GraknConv(
            nn_node=nn.Sequential(
                nn.Linear(12 + latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
            ),
            nn_edge=nn.Sequential(
                nn.Linear(12 + 6 + 12, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
            ),
        )
        self.conv2 = GraknConv(
            nn_node=nn.Sequential(
                nn.Linear(2 * latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
            ),
            nn_edge=nn.Sequential(
                nn.Linear(3 * latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
            ),
        )
        self.conv3 = GraknConv(
            nn_node=nn.Sequential(
                nn.Linear(latent_size + edge_output_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, node_output_size),
            ),
            nn_edge=nn.Sequential(
                nn.Linear(2 * latent_size + latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, edge_output_size),
            ),
        )

    def forward(self, data):
        x_node, x_edge, edge_index, = (
            data.x,
            data.edge_attr,
            data.edge_index,
        )

        x_node = self.node_embedder(x_node)
        x_edge = self.edge_embedder(x_edge)
        x_node, x_edge = self.conv1(x_node, edge_index, x_edge)
        x_node, x_edge = self.conv2(x_node, edge_index, x_edge)
        x_node, x_edge = self.conv3(x_node, edge_index, x_edge)
        return x_node, x_edge


class GraknConv(MessagePassing):
    r"""

    Args:
        nn_node (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_edge (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.

        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        nn_node: Callable,
        nn_edge: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(GraknConv, self).__init__(**kwargs)
        self.nn_node = nn_node
        self.nn_edge = nn_edge
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_node)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        src, dst = edge_index
        edge_repr = torch.cat([x[src], edge_attr, x[dst]], dim=-1)

        edge_repr = self.nn_edge(edge_repr)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_repr, size=size)

        return self.nn_node(out), edge_repr

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        concatenated = torch.cat([x_j, edge_attr], dim=-1)
        return torch.cat([x_j, edge_attr], dim=-1)
