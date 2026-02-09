"""Link prediction decoders."""

from __future__ import annotations

import torch
from torch import nn


class DotProductDecoder(nn.Module):
    """Dot product decoder for link prediction.

    Computes scores as the dot product of source and destination embeddings.
    """

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Compute edge scores.

        :param z_src: Source node embeddings [num_edges, dim].
        :param z_dst: Destination node embeddings [num_edges, dim].
        :return: Edge scores [num_edges].
        """
        return (z_src * z_dst).sum(dim=-1)


class MLPDecoder(nn.Module):
    """MLP decoder for link prediction.

    Concatenates source and destination embeddings and passes through MLP.

    :param in_channels: Embedding dimension.
    :param hidden_channels: Hidden layer dimension.
    :param dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Compute edge scores.

        :param z_src: Source node embeddings [num_edges, dim].
        :param z_dst: Destination node embeddings [num_edges, dim].
        :return: Edge scores [num_edges].
        """
        z = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(z).squeeze(-1)


class BilinearDecoder(nn.Module):
    """Bilinear decoder for link prediction.

    Computes scores as z_src^T W z_dst.

    :param in_channels: Embedding dimension.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(in_channels, in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Compute edge scores.

        :param z_src: Source node embeddings [num_edges, dim].
        :param z_dst: Destination node embeddings [num_edges, dim].
        :return: Edge scores [num_edges].
        """
        return self.bilinear(z_src, z_dst).squeeze(-1)


DECODER_REGISTRY: dict[str, type[nn.Module]] = {
    "dot": DotProductDecoder,
    "mlp": MLPDecoder,
    "bilinear": BilinearDecoder,
}


def get_decoder(name: str, **kwargs) -> nn.Module:
    """Get a decoder by name.

    :param name: Decoder name ('dot', 'mlp', 'bilinear').
    :raises KeyError: If decoder is not registered.
    :return: Decoder instance.
    """
    if name not in DECODER_REGISTRY:
        available = ", ".join(sorted(DECODER_REGISTRY.keys()))
        raise KeyError(f"Unknown decoder '{name}'. Available: {available}")
    return DECODER_REGISTRY[name](**kwargs)
