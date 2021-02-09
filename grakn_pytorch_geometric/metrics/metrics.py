import torch


def ignore_class(pred, target, ignore_index=None):
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    return pred, target


def correct(pred, target, ignore_index=None):
    pred, target = ignore_class(pred, target, ignore_index)
    predicted_index = torch.argmax(pred, dim=-1)
    return predicted_index == target


def existence_accuracy(pred, target, ignore_index=None):
    return correct(pred, target, ignore_index).float().mean()


def fraction_solved(pred, target, batch, ignore_index=None):
    """Fraction of graphs for which all nodes are
    classified correctly.
    """
    n_solved, n_total = n_graphs_solved(pred, target, batch, ignore_index, return_n_total=True)
    return n_solved / n_total


def n_graphs_solved(pred, target, batch, ignore_index=None, return_n_total=False):
    """Number of graphs for which all nodes are
    classified correctly.
    """
    splitter = BatchSplitter(batch)
    n_solved = 0
    for pred_batch, target_batch in zip(splitter(pred), splitter(target)):
        n_solved += correct(pred_batch, target_batch, ignore_index).all()
    if return_n_total:
        return n_solved, splitter.n_graphs
    return n_solved


class BatchSplitter:
    """Splitting up tensors by pytorch geometric batch indeces.
    Only works for nodes.
    """

    def __init__(self, batch):
        self.masks = [batch == i for i in batch.unique()]
        self.n_graphs = len(self.masks)

    def __call__(self, tensor):
        for mask in self.masks:
            yield tensor[mask]
