"""Pytorch version of kglib/kgcn_tensorflow/models/embedding.py

In the tensorflow version there is a seperate ThingEmbedder and RoleEmbedder, but looking into it,
the RoleEmbedder is just the same as the ThingEmbedder, but without an value encoding.
"""

from typing import Optional, Tuple, Mapping, Sequence, Hashable
from numbers import Number
import torch
import torch.nn as nn
from typedb_pytorch_geometric.models.attribute import (
    ContinuousAttribute,
    CategoricalAttribute,
    BlankAttribute,
)


class Embedder(nn.Module):
    """
    Module to embed the 3-vector (preexistence, type, value) from Grakn. Each type
    has its own embedder to embed the value.

    Args:
        types (list): (node or edge) types. Order is important! Order should be the
            oder that is used in the datalaoder to map nodes or edge type names to
            integers in pytorch tensors.
        encode_preexistence (bool): whether to start each embedding with the pre-existence bit
        type_embedding_dim (int): size of the type embedding
        attr_embedding_dim (int): size of the attribute embedding
        categorical_attributes (dict): dict of {"attribute_name": ["catergory_1", "category_2", ...]}
        continuous_attributes (dict): dict of {"attribute_name": (min_value, max_value)}
    """

    def __init__(
        self,
        types: Sequence,
        encode_preexistence: bool = True,
        type_embedding_dim: int = 0,
        attr_embedding_dim: int = 0,
        categorical_attributes: Optional[Mapping[Hashable, Sequence]] = None,
        continuous_attributes: Optional[
            Mapping[Hashable, Tuple[Number, Number]]
        ] = None,
    ):
        super(Embedder, self).__init__()

        self.n_out_features = (
            bool(encode_preexistence) + type_embedding_dim + attr_embedding_dim
        )

        self.type_embedder = None
        self.attr_embeddder = None

        self._encode_preexistence = encode_preexistence
        if type_embedding_dim:
            self.type_embedder = nn.Embedding(
                num_embeddings=len(types), embedding_dim=type_embedding_dim
            )
        if attr_embedding_dim:
            self.attr_embeddder = TypewiseEncoder(
                types=types,
                embedding_dim=attr_embedding_dim,
                categorical_attributes=categorical_attributes,
                continuous_attributes=continuous_attributes,
            )

    def forward(self, X):
        embedding = [torch.zeros((X.size(0), 0))]
        if self._encode_preexistence:
            embedding.append(X[:, 0:1])
        if self.type_embedder:
            embedding.append(self.type_embedder(X[:, 1].long()))
        if self.attr_embeddder:
            embedding.append(self.attr_embeddder(X[:, 1], X[:, 2:]))
        return torch.cat(embedding, dim=-1)

    def extra_repr(self) -> str:
        return "out_features={}".format(self.n_out_features)


class TypewiseEncoder(nn.Module):
    """
    Encodes all values (categorical or continuous) into embedding_dim dimensions.
    Each node or edge type gets its own
    type_pytorch_geometric.embedding.attribute.Atribute which does
    the actual encoding of individual values.
    Types that are not mentioned categorical_attributes or continuous_attributes
    are encoded with zero's of length embedding_dim to force all attributes to
    be encoded by the same lenhth vector.

    Args:
        types (list): (node or edge) types. Order is important! Order should be the
            oder that is used in the datalaoder to map nodes or edge type names to
            integers in pytorch tensors.
        embedding_dim (int): size of attribute embedding dim
        categorical_attributes (dict): dict of {"attribute_name": ["catergory_1", "category_2", ...]}
        continuous_attributes (dict): dict of {"attribute_name": (min_value, max_value)}
    """

    def __init__(
        self,
        types: Sequence,
        embedding_dim: int,
        categorical_attributes: Optional[Mapping[Hashable, Sequence]] = None,
        continuous_attributes: Optional[
            Mapping[Hashable, Tuple[Number, Number]]
        ] = None,
    ):
        super(TypewiseEncoder, self).__init__()
        self._types = types
        self._embedding_dim = embedding_dim
        self._encoders_for_types = [
            BlankAttribute(attr_embedding_dim=self._embedding_dim)
        ] * len(types)
        self._construct_categorical_embedders(categorical_attributes)
        self._construct_continuous_embedders(continuous_attributes)
        self._encoders_for_types = nn.ModuleList(self._encoders_for_types)

    def forward(self, types, features):
        shape = (features.size(0), self._embedding_dim)
        types = types.long()
        encoded_features = torch.zeros(shape)

        for i, encoder in enumerate(self._encoders_for_types):
            mask = types == i
            encoded_features[mask] = encoder(features[mask])
        return encoded_features

    def _construct_categorical_embedders(self, categorical_attributes):
        if not categorical_attributes:
            return
        for attribute_type, categories in categorical_attributes.items():
            attr_typ_index = self._types.index(attribute_type)
            embedder = CategoricalAttribute(
                num_categories=len(categories),
                attr_embedding_dim=self._embedding_dim,
                name=attribute_type,
            )
            self._encoders_for_types[attr_typ_index] = embedder

    def _construct_continuous_embedders(self, continuous_attributes):
        if not continuous_attributes:
            return
        for attribute_type in continuous_attributes.keys():
            attr_typ_index = self._types.index(attribute_type)
            embedder = ContinuousAttribute(
                attr_embedding_dim=self._embedding_dim, name=attribute_type
            )
            self._encoders_for_types[attr_typ_index] = embedder
