from typing import Optional, Tuple, Union
import torch

def ignore_class(pred: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out samples from batch where the target is ignore_index.

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): targets
        ignore_index (int, optional): target index that should be filtered out

    Returns: 2-tuple of prediction and target torch.Tensors
    """
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    return pred, target


def correct(pred: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> torch.BoolTensor:
    """
    Get correctness of each prediction vs. target.

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): targets
        ignore_index (int, optional): target index that should be filtered out

    Returns: torch.Tensor with 1 and 0 indicating correctness of predictions

    """
    pred, target = ignore_class(pred, target, ignore_index)
    predicted_index = torch.argmax(pred, dim=-1)
    return predicted_index == target


def existence_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
    """
    Accuracy.

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): targets
        ignore_index (int, optional): target index that should be
            filtered out before calculating metric

    Returns: torch.Tensor Accuracy

    """
    return correct(pred, target, ignore_index).float().mean()


def fraction_solved(pred: torch.Tensor, target: torch.Tensor, batch: torch.LongTensor, ignore_index: Optional[int] = None) -> float:
    """
    Fraction of graphs for which all nodes are
    classified correctly.

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): targets
        batch (torch.IntTensor):
        ignore_index (int, optional): target index that should be
            filtered out before calculating metric

    Returns: torch.FloatTensor fraction of graphs for which all
        nodes are classified correctly.
    """

    n_solved, n_total = n_graphs_solved(
        pred, target, batch, ignore_index, return_n_total=True
    )
    return n_solved / n_total


def n_graphs_solved(pred: torch.Tensor, target: torch.Tensor, batch: torch.LongTensor, ignore_index: Optional[int] = None, return_n_total: bool = False) -> Union[int, Tuple[int, int]]:
    """
    Number of graphs for which all nodes are
    classified correctly.

    Args:
        pred (torch.Tensor): predictions
        target (torch.Tensor): targets
        batch (torch.IntTensor):
        ignore_index (int, optional): target index that should be
            filtered out before calculating metric
        return_n_total (bool): whether to return the total number
            of graphs as well.

    Returns (int, tuple): Number of graphs for which all nodes are
        classified correctly. If return_n_total is set to True then
        (number correct graphs, total number of graphs)

    """
    splitter = BatchSplitter(batch)
    n_solved = 0
    for pred_batch, target_batch in zip(splitter(pred), splitter(target)):
        n_solved += correct(pred_batch, target_batch, ignore_index).all()
    if return_n_total:
        return n_solved, splitter.n_graphs
    return n_solved


class BatchSplitter:
    """
    Splitting up tensors by pytorch geometric batch indices.
    Only works for nodes.

    Args:
        batch (torch.Tensor): tensor with indices specifying
            to which subgraph in the batch each node belongs.
    """

    def __init__(self, batch: torch.Tensor):
        self.masks = [batch == i for i in batch.unique()]
        self.n_graphs = len(self.masks)

    def __call__(self, tensor) -> torch.Tensor:
        """
        Split batch in tensors with all nodes belonging
        to same subgraph.
        Args:
            tensor (torch.Tensor): batch containing subgraphs

        Yields:
            tensors with all nodes belonging to same subgraph.
        """
        for mask in self.masks:
            yield tensor[mask]
