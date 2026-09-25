# losses/contrastive.py

import torch
import torch.nn.functional as F


def contrastive_loss(
    video_emb: torch.Tensor,   # (B, D) — mean pooled video encoder output
    audio_emb: torch.Tensor,   # (B, D) — mean pooled audio encoder output
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-modal contrastive loss (CLIP-style).

    Pulls together embeddings from the same video+audio clip,
    pushes apart embeddings from different clips in the batch.

    This is optional in Phase 1 — used alongside reconstruction
    loss to further enforce audio-visual alignment.

    Args:
        video_emb   : (B, D) L2-normalized video embeddings
        audio_emb   : (B, D) L2-normalized audio embeddings
        temperature : softmax temperature scaling factor

    Returns:
        scalar loss
    """
    B = video_emb.shape[0]

    # L2 normalize both
    video_emb = F.normalize(video_emb, dim=-1)
    audio_emb = F.normalize(audio_emb, dim=-1)

    # similarity matrix (B, B)
    logits = torch.matmul(video_emb, audio_emb.T) / temperature

    # diagonal = positive pairs (same clip)
    labels = torch.arange(B, device=video_emb.device)

    # symmetric loss: video→audio and audio→video
    loss_v2a = F.cross_entropy(logits,   labels)
    loss_a2v = F.cross_entropy(logits.T, labels)

    return (loss_v2a + loss_a2v) / 2