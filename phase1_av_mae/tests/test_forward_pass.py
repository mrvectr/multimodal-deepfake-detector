# tests/test_forward_pass.py

import torch
import pytest
from omegaconf import OmegaConf
from models.crossmodal_mae import CrossModalMAE


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {
            "video_patch_size"  : [2, 16, 16],
            "audio_patch_size"  : [16, 16],
            "encoder_embed_dim" : 192,
            "encoder_depth"     : 2,
            "encoder_num_heads" : 3,
            "decoder_embed_dim" : 96,
            "decoder_depth"     : 2,
            "decoder_num_heads" : 3,
            "mask_ratio_video"  : 0.4,
            "mask_ratio_audio"  : 0.4,
            "norm_pix_loss"     : True,
        },
        "data": {
            "num_frames"       : 4,
            "img_size"         : 32,
            "n_mels"           : 32,
            "target_length"    : 32,
            "sample_rate"      : 16000,
            "clip_len_seconds" : 1,
            "n_fft"            : 400,
            "hop_length"       : 160,
        },
        "train": {
            "device": "cpu",
        }
    })


def test_forward_pass_runs(cfg):
    model    = CrossModalMAE(cfg)
    video    = torch.randn(2, 3, 4, 32, 32)
    waveform = torch.randn(2, 16000)

    total_loss, loss_v, loss_a = model(video, waveform)

    assert total_loss.item() > 0,  "Total loss should be positive"
    assert loss_v.item()     > 0,  "Video loss should be positive"
    assert loss_a.item()     > 0,  "Audio loss should be positive"


def test_loss_is_scalar(cfg):
    model    = CrossModalMAE(cfg)
    video    = torch.randn(2, 3, 4, 32, 32)
    waveform = torch.randn(2, 16000)

    total_loss, loss_v, loss_a = model(video, waveform)

    assert total_loss.shape == torch.Size([]), "Total loss must be scalar"
    assert loss_v.shape     == torch.Size([]), "Video loss must be scalar"
    assert loss_a.shape     == torch.Size([]), "Audio loss must be scalar"


def test_backward_pass(cfg):
    model    = CrossModalMAE(cfg)
    video    = torch.randn(2, 3, 4, 32, 32)
    waveform = torch.randn(2, 16000)

    total_loss, _, _ = model(video, waveform)
    total_loss.backward()

    # check at least one parameter has gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward pass"


def test_different_inputs_give_different_losses(cfg):
    model    = CrossModalMAE(cfg)
    video_a  = torch.randn(2, 3, 4, 32, 32)
    video_b  = torch.randn(2, 3, 4, 32, 32)
    waveform = torch.randn(2, 16000)

    loss_a, _, _ = model(video_a, waveform)
    loss_b, _, _ = model(video_b, waveform)

    assert loss_a.item() != loss_b.item(), \
        "Different inputs should give different losses"