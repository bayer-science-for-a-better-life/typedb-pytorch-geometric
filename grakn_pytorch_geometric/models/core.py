from typing import Callable, Union, Sequence, Optional, Mapping, Hashable, Tuple, Type
from numbers import Number
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

from grakn_pytorch_geometric.models.embedding import Embedder


class KGCN(torch.nn.Module):
    """
    Model like KGCN in tensorflow with graphnets. Maybe missing
    a few layer norms.

    Args:
        node_types (list): list of node type names. Order is important! Order should be the
            oder that is used in the datalaoder to map nodes type names to
            integers in pytorch tensors.
        edge_types (list): list of edge type names. Order is important! Order should be the
            oder that is used in the datalaoder to map edge type names to
            integers in pytorch tensors.
        categorical_attributes (dict): dict of {"attribute_name": ["catergory_1", "category_2", ...]}
        continuous_attributes (dict): dict of {"attribute_name": (min_value, max_value)}
        node_output_size (int): size of the output layer for the neural net predicting
            node properties.
        edge_output_size (int): size of the output layer for the neural net predicting
            edge properties.
        latent_size (int): size of
        num_layers: number of layers in each of encoder, convolution blocks, and decoder.
    """

    def __init__(
        self,
        node_types: Sequence = None,
        edge_types: Sequence = None,
        categorical_attributes: Optional[Mapping[Hashable, Sequence]] = None,
        continuous_attributes: Optional[
            Mapping[Hashable, Tuple[Number, Number]]
        ] = None,
        node_output_size: int = 3,
        edge_output_size: int = 3,
        latent_size: int = 16,
        num_layers: int = 2,
    ):
        super(KGCN, self).__init__()

        node_types = node_types or []
        edge_types = edge_types or []

        node_embedder = Embedder(
            types=node_types,
            type_embedding_dim=5,
            attr_embedding_dim=6,
            categorical_attributes=categorical_attributes,
            continuous_attributes=continuous_attributes,
        )

        edge_embedder = Embedder(types=edge_types, type_embedding_dim=5)

        self.node_encoder = nn.Sequential(
            node_embedder,
            mlp(
                [node_embedder.n_out_features] + [latent_size] * num_layers,
                activate_final=True,
            ),
        )

        self.edge_encoder = nn.Sequential(
            edge_embedder,
            mlp(
                [edge_embedder.n_out_features] + [latent_size] * num_layers,
                activate_final=True,
            ),
        )

        self.conv = GraknConv(
            nn_node=mlp(
                [3 * latent_size] + [latent_size] * num_layers, activate_final=True
            ),
            nn_edge=mlp(
                [6 * latent_size] + [latent_size] * num_layers, activate_final=True
            ),
        )

        self.node_decoder = mlp([latent_size] * (num_layers + 1) + [node_output_size])
        self.edge_decoder = mlp([latent_size] * (num_layers + 1) + [edge_output_size])

    def forward(
        self,
        data: torch_geometric.data.batch.Batch,
        steps: int = 5,
        return_all_steps: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            data (torch_geometric.data.batch.Batch):
            steps (int): number of times graph convolution block is applied.
            return_all_steps (bool): if True, return not only the output
                after all steps, but add a dimension holding the output
                of every step.
        Returns: torch.Tensor logits

        """

        x_node, x_edge, edge_index, = (
            data.x,
            data.edge_attr,
            data.edge_index,
        )

        x_node = self.node_encoder(x_node)
        x_edge = self.edge_encoder(x_edge)
        x_node_0 = x_node
        x_edge_0 = x_edge
        node_predictions = []
        edge_predictions = []
        for _ in range(steps):
            x_node, x_edge = self.conv(
                torch.cat([x_node_0, x_node], dim=1),
                edge_index,
                torch.cat([x_edge_0, x_edge], dim=1),
            )
            pred_node = self.node_decoder(x_node)
            pred_edge = self.edge_decoder(x_edge)
            node_predictions.append(pred_node)
            edge_predictions.append(pred_edge)
        if return_all_steps:
            return torch.stack(node_predictions, dim=-1), torch.stack(
                edge_predictions, dim=-1
            )
        return pred_node, pred_edge


class KGCNNoLoopBack(torch.nn.Module):
    """
    A model with 3 graph convolution layers.
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

    def __init__(self, nn_node: Callable, nn_edge: Callable, **kwargs):
        kwargs.setdefault("aggr", "add")
        super(GraknConv, self).__init__(**kwargs)
        self.nn_node = nn_node
        self.nn_edge = nn_edge
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_node)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tuple[Tensor, Tensor]:

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


def mlp(
    layer_sizes: Sequence[int],
    activation: Type[nn.Module] = nn.ReLU,
    activate_final: bool = False,
):
    """
    Simple utility function to create a multilayer perceptron.

    Args:
        layer_sizes list[int]: list with size of each layer.
        activation torch.nn.Module: activation function class
        activate_final (bool): whether to apply the activation function
            to the last layer.

    Returns: torch.nn.Module of the neural mlp.
    """

    transformations = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        transformations.append(nn.Linear(in_size, out_size))
        transformations.append(activation())
    if not activate_final:
        transformations = transformations[:-1]
    return nn.Sequential(*transformations)
