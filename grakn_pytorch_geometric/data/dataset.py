from typing import Sequence, Callable, Optional
import torch
import networkx as nx
import torch_geometric

from kglib.kgcn_data_loader.dataset.grakn_networkx_dataset import GraknNetworkxDataSet


class GraknPytorchGeometricDataSet(torch_geometric.data.dataset.Dataset):
    """
    Subclass of Pytorch Geometric DataSet:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset

    Acts as a connector between the Grakn Database and Pytorch Geometric. Intenally it uses the more
    generic GraknNetworkxDataSet in kglib but returns Pytorch Geometric Data objects instead of
    networkx graphs.

    Args:
        example_indices: sequence of indices that are used for training, validation or test.
            These indices are handed to get_query_handles_for_id method during datalaoding.
        get_query_handles_for_id (callable):  function taking an example_index as its input and
            returns an iterable, each element containing a Graql query, a function to sample the
            answers, and a QueryGraph object which must be the Grakn graph representation of the
            query. This tuple is termed a "query_handle".
        database (str): name of the database.
        uri (str): uri of the database, exmaple: "localhost:1729".
        infer (bool): whether to use Grakn reasoning.
        networkx_transform (callable, optional): transform applied to networkx graph
            before it becomes a pytorch geometric graph.
        caching (bool, optional): keep sample graphs fetched from the database in memory.
        *args: args for torch_geometric.data.dataset.Dataset
        **kwargs: kwargs for torch_geometric.data.dataset.Dataset

    """

    def __init__(
        self,
        example_indices: Sequence,
        get_query_handles_for_id: Callable,
        database: str,
        uri: str = "localhost:1729",
        infer: bool = True,
        networkx_transform: Optional[Callable[[nx.Graph], nx.Graph]] = None,
        caching: bool = False,
        *args,
        **kwargs
    ):
        super(GraknPytorchGeometricDataSet, self).__init__(*args, **kwargs)
        self._networkx_dataset = GraknNetworkxDataSet(
            example_indices=example_indices,
            get_query_handles_for_id=get_query_handles_for_id,
            database=database,
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

    Modified from Pytorch Geometric.

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
