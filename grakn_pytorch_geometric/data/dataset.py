import torch
import networkx as nx
import torch_geometric
from grakn_dataloading.networkx import GraknNetworkxDataSet


class GraknPytorchGeometricDataSet(torch_geometric.data.dataset.Dataset):
    """
    Pytorch Geometric DataSet:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset
    Using the more generic GraknNetworkxDataSet but returns Pytorch Geometric Data objects
    instead of networkx graphs.

    """

    def __init__(
        self,
        example_indices,
        get_query_handles_for_id,
        keyspace,
        uri="localhost:48555",
        infer=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        networkx_transform=None,
        caching=False,
    ):
        super(GraknPytorchGeometricDataSet, self).__init__(
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self._networkx_dataset = GraknNetworkxDataSet(
            example_indices=example_indices,
            get_query_handles_for_id=get_query_handles_for_id,
            keyspace=keyspace,
            uri=uri,
            infer=infer,
            transform=networkx_transform,
        )

        self.caching = caching
        self._cache = {}

    def len(self):
        return len(self._networkx_dataset)

    def get(self, idx):
        if self.caching and idx in self._cache:
            return self._cache[idx]
        graph = networkx_to_pytorch_geometric(self._networkx_dataset[idx])
        if self.caching:
            self._cache[idx] = graph
        return graph


def networkx_to_pytorch_geometric(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    # this line in the original pytorch geometric from_networkx
    # code does the wrong thing when edge_index has 3 dimensions
    # instead of 2 (which it has somehow in our case, does not fit
    # with networkx docs. Maybe because MultiDiGraph)
    # data['edge_index'] = edge_index.view(2, -1)
    data["edge_index"] = edge_index[:2, :]
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
