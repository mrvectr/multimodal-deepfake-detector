import torch


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """
    Create 1D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension. Must be even.
        pos: Position tensor of shape (M,).

    Returns:
        Positional embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    # Half of the dimensions are used for sine,
    # half for cosine.
    omega = torch.arange(
        embed_dim // 2,
        dtype=torch.float32,
        device=pos.device,
    )

    # Generate different frequencies.
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))

    # Ensure positions are flattened.
    pos = pos.reshape(-1).float()

    # Position × frequency.
    out = torch.outer(pos, omega)

    # Sinusoidal encoding.
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # Final shape: (M, embed_dim)
    return torch.cat([emb_sin, emb_cos], dim=1)