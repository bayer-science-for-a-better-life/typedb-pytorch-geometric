import torch


def correct(pred, target, ignore_index=None):
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]
    predicted_index = torch.argmax(pred, dim=-1)
    return predicted_index == target


def existence_accuracy(pred, target, ignore_index=None):
    return correct(pred, target, ignore_index).float().mean()


def fraction_solved(pred, target, batch, ignore_index=None):
    """Fraction of entire
    """
    splitter = BatchSplitter(batch)
    n_solved = 0
    for pred_batch, target_batch in zip(splitter(pred), splitter(target)):
        n_solved += correct(pred_batch, target_batch, ignore_index).all()
    return n_solved / splitter.n_graphs


class BatchSplitter():
    """Splitting up tensors by pytorch geometric batch indeces.
       Only works for nodes.
    """
    def __init__(self, batch):
        self.masks = [batch == i for i in batch.unique()]
        self.n_graphs = len(self.masks)

    def __call__(self, tensor):
        for mask in self.masks:
            yield tensor[mask]
