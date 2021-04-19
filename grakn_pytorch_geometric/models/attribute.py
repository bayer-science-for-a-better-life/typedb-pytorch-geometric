import abc
from typing import Optional
import torch
import torch.nn as nn


class Attribute(nn.Module, abc.ABC):
    """
    Abstract base class for Attribute value embedding models
    """

    def __init__(self, attr_embedding_dim: int, name: Optional[str] = None):
        super(Attribute, self).__init__()
        self._attr_embedding_dim = attr_embedding_dim
        self._type_name = name

    def extra_repr(self) -> str:
        if self._type_name:
            return "name={}, out_features={}".format(
                self._type_name, self._attr_embedding_dim
            )
        return "out_features={}".format(self._attr_embedding_dim)


class ContinuousAttribute(Attribute):
    """
    Continuous attribute. Turn a continuous attribute into a vector.
    This is done so it can have the same size as categorical
    embeddings of other nodes.

    Args:
        attr_embedding_dim (int): size of the embedding.
    """
    def __init__(self, attr_embedding_dim: int, name: Optional[str] = None):
        super(ContinuousAttribute, self).__init__(attr_embedding_dim, name)
        # for now input is size 1 (when pytorch 1.8 comes out can be made dependent on input)
        # a three layer MLP is used like in KGCN. Three layers do maybe not make
        # too much sense but keeping it like the tensorflow version for now.
        # todo: when pytorch 1.8 comes out, change first Linear to LazyLinear
        self.embedder = nn.Sequential(
            nn.Linear(1, attr_embedding_dim),
            nn.ReLU(),
            nn.Linear(attr_embedding_dim, attr_embedding_dim),
            nn.ReLU(),
            nn.Linear(attr_embedding_dim, attr_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value)
        # todo: tensorboard histogram of continuous attribute embedding
        return embedding


class CategoricalAttribute(Attribute):
    """
    Categorical attribute.

    Args:
        num_categories (int): number of values the attribute can take.
        attr_embedding_dim (int): size of the embedding.
    """
    def __init__(self, num_categories: int, attr_embedding_dim: int, name: Optional[str] = None):
        super(CategoricalAttribute, self).__init__(attr_embedding_dim, name)
        self.embedder = nn.Embedding(
            num_embeddings=num_categories, embedding_dim=attr_embedding_dim
        )

    def forward(self, attribute_value):
        # todo: tensorboard histogram here of continuous attributes
        embedding = self.embedder(attribute_value.squeeze().long())
        # todo: tensorboard histogram of continuous attibute embedding
        return embedding


class BlankAttribute(Attribute):
    """
    Creates an attribute embedding with just zeros. This is
    useful for nodes without attributes to keep the size of
    their representation vectors the same size as vectors

    Args:
        attr_embedding_dim (int): size of the embedding.
    """
    def __init__(self, attr_embedding_dim: int, name: Optional[str] = None):
        super(BlankAttribute, self).__init__(attr_embedding_dim, name)

    def forward(self, attribute_value):
        shape = (attribute_value.size(0), self._attr_embedding_dim)
        return torch.zeros(shape)
