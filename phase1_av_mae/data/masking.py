import torch

def random_masking(x: torch.Tensor, masking_ratio: float):
    
    B,N,D = x.shape #batch, sequence length(for each image), embedding dimension
    N_keep = int(N * (1 - masking_ratio)) #number of tokens to keep
    
    noise = torch.rand(B, N, device=x.device) #step1: generate random noise
    ids_shuffle = torch.argsort(noise, dim=1) #step2: sort noise for each sample
    
    ids_keep = ids_shuffle[:, :N_keep] #step3: keep the first N_keep ids
    ids_mask = ids_shuffle[:, N_keep:] #step4: keep the rest of the ids as mask
    
    ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D) #step5: expand the ids_keep to match the embedding dimension
    x_visible = torch.gather(
        x,
        dim=1,
        index=ids_keep_expanded
    )
    
    #step6: generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(B, N, device=x.device)
    mask[:, :N_keep] = 0

    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore
    )
    
    return x_visible, mask, ids_restore