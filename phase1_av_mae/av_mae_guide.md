# Understanding Your Cross-Modal Masked Audio-Video Autoencoder

A ground-up walkthrough of `phase1_av_mae`, written for someone new to AI/ML.

---

## Part 0 — What the whole thing is trying to do

You have videos of people talking. Each video has **pictures** (frames) and **sound** (a waveform). Nobody has labelled them. You want a neural network that learns something useful about faces, lips, and speech *without* labels.

The trick, from the **Masked Autoencoder (MAE)** paper, is to invent a task the data can grade by itself:

1. Chop the video into small 3D blocks and the audio into small 2D blocks. Each block becomes a **token**.
2. Throw away 40% of those tokens at random. The model never sees them.
3. Feed only the surviving 60% into an encoder.
4. Ask a decoder to guess the raw pixel/spectrogram values of the thrown-away blocks.
5. Score the guess against the original, which you still have on disk.

There is no human labelling anywhere. The label *is* the data you hid. This family of methods is called **self-supervised learning**, and the specific flavour is a **pretext task**.

Why this teaches anything: to fill in a hidden patch of a mouth, the network has to have internally figured out what a mouth is, how it moves, and — because your decoder sees audio and video together — what sound a given mouth shape makes. Those internal features are the real product. The reconstructed video is a throwaway by-product.

The `40%` is `mask_ratio_video: 0.40` / `mask_ratio_audio: 0.40` in `configs/baseline.yaml`. (Side note for later: the original MAE used 75% for images and VideoMAE used 90% for video, because video is extremely redundant across time. 40% is on the easy side — the model can often solve it by copying neighbouring pixels. Worth revisiting once it trains.)

---

## Part 1 — What a video actually *is*, numerically

Forget "video file" for a moment. To a neural network a video clip is a single **tensor**: a multi-dimensional grid of floating-point numbers.

### The dimensions

A single frame is an image. An image of size 224×224 in colour is a grid of numbers with shape:

```
(3, 224, 224)
 │    │    └── W: width in pixels
 │    └─────── H: height in pixels
 └──────────── C: colour channels — Red, Green, Blue
```

That's 3 × 224 × 224 = **150,528 numbers** for one frame. Each number is the brightness of one colour at one pixel. In your code they're scaled to the range `[0, 1]` by `arr = np.array(img, dtype=np.float32) / 255.0` in `data/dataset.py` (raw image files store 0–255 integers; networks prefer small floats).

Stack `T` frames along a new time axis and you get a clip:

```
(3, T, 224, 224)
 │   │
 │   └── T: number of frames
 └────── C
```

Then you process many clips at once for efficiency — that's the **batch** dimension `B`, always first:

```
video : (B, 3, T, H, W)
```

This exact shape appears as the docstring of `CrossModalMAE.forward` and is what `make_synthetic_batch` in `scripts/verify_baseline.py` fabricates with `torch.randn(B, 3, T, H, W)`.

### Your specific numbers

From `configs/baseline.yaml`:

```yaml
clip_len_seconds: 5     # each training clip is 5 seconds long
fps: 25                 # source videos run at 25 frames per second
num_frames: 16          # but we only KEEP 16 frames
img_size: 224           # every frame resized to 224 × 224
```

A 5-second clip at 25 fps physically contains 5 × 25 = **125 frames**. You keep only 16 of them. `data/dataset.py` does this with:

```python
indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
```

`np.linspace(0, 124, 16)` gives evenly spaced frame numbers: 0, 8, 16, 24, ... 124. So you're sampling roughly every 8th frame — effectively watching the clip at ~3.2 fps. This is standard practice (VideoMAE, ViViT all do it): consecutive frames are nearly identical, so most of them are wasted compute. The cost is that very fast motion — exactly what lip movements are — gets aliased. Keep this in mind as a knob to tune later.

Also note `img_size: 224` and the resize `img.resize((self.img_size, self.img_size))`: this **squashes** the aspect ratio rather than cropping. A 16:9 video becomes a squarish distorted frame. For talking-face data where the face is centred, a centre-crop-then-resize is usually better.

So one clip in your pipeline is a tensor of shape `(3, 16, 224, 224)` = **2,408,448 numbers**.

**Topics to study here:** tensors and tensor shapes; NumPy/PyTorch indexing; image representation (RGB, bit depth); video containers vs codecs (mp4 is a container, H.264 is a codec); frame rate and temporal sampling.

---

## Part 2 — What audio actually *is*, and why it becomes a picture

### Raw waveform

Sound is air pressure changing over time. A microphone samples that pressure `sample_rate` times per second. Your config says `sample_rate: 16000`, so 16,000 numbers per second of audio.

```
5 seconds × 16,000 Hz = 80,000 numbers
waveform : (B, 80000)
```

`data/dataset.py` computes exactly this: `self.n_samples = self.sample_rate * self.clip_seconds`, mixes stereo down to mono with `.mean(axis=0)`, resamples to 16 kHz if needed, and pads or crops to exactly 80,000 samples so every item in a batch has the same length (a `DataLoader` cannot stack tensors of different sizes).

### Why not feed the waveform directly

80,000 raw numbers where the meaningful structure (pitch, formants, phonemes) lives in *frequency*, not amplitude-over-time. Transformers would struggle. So we convert to a **mel-spectrogram**, which turns audio into an image-like 2D grid.

### The Short-Time Fourier Transform (STFT)

The idea: slide a small window along the waveform, and for each window ask "how much of each frequency is present here?" That question is answered by the **Fourier transform**.

Your parameters:

```yaml
n_fft: 400        # window length in samples = 400/16000 = 25 milliseconds
hop_length: 160   # slide forward by 160 samples = 10 milliseconds each step
n_mels: 128       # summarise into 128 frequency bands
```

25 ms windows with 10 ms hops is the near-universal default in speech processing — it's roughly the timescale at which speech is stationary.

Number of time steps produced:

```
80,000 samples ÷ 160 hop  + 1  ≈  501 time frames
```

Each time frame gets 128 mel values. So the mel-spectrogram is:

```
mel : (B, 1, 128, 501)
       │  │   │    └── time frames
       │  │   └─────── mel frequency bins
       │  └─────────── "channels" = 1, like a greyscale image
       └────────────── batch
```

That `(1, 128, 501)` shape is deliberately image-shaped. A `Conv2d` can't tell the difference between a greyscale photo and a spectrogram, which is why the same Vision Transformer machinery works on both.

**"Mel"** means the 128 frequency bands aren't evenly spaced in Hz — they're spaced to match human hearing, which resolves low frequencies much more finely than high ones. `torch.log(mel + 1e-6)` in `_extract_mel` then compresses the range, because loudness is perceived logarithmically (this is also why we use decibels). The `+ 1e-6` prevents `log(0) = -inf`.

### ⚠️ A real problem in your code

`models/audio_encoder.py` has `target_length: int = 128` and `_extract_mel` does:

```python
mel = mel[:, :, :self.target_length]     # crop to 128 time frames
```

128 time frames × 10 ms = **1.28 seconds**. Your video covers 5 seconds. You are throwing away 75% of the audio and keeping only the beginning, so the audio and video in the same training sample describe **different moments in time**. For a cross-modal model whose entire premise is that audio and video correspond, this quietly destroys the learning signal.

Fixes, pick one:
- Set `target_length: 512` (≈5.1 s) and keep patch size 16 → 32 time patches instead of 8.
- Or increase `hop_length` to ~625 to compress 5 s into 128 frames (loses temporal detail).
- Either way, `target_length` must be divisible by the audio patch time size.

I'd take the first option.

**Topics to study here:** sampling rate and the Nyquist theorem; Fourier transform and STFT; spectrograms; the mel scale and MFCCs; log-compression / decibels. Meta-topic: **digital signal processing (DSP) for speech**.

---

## Part 3 — From tensors to tokens (this is the heart of it)

A Transformer doesn't consume pixels. It consumes a **sequence of vectors**, exactly like a language model consumes a sequence of word-vectors. So you must convert your grid of pixels into a list of vectors. Each vector is a **token**.

### Video: cutting the clip into "tubes"

`video_patch_size: [2, 16, 16]` means each patch is:

- 2 frames deep in **time**
- 16 pixels tall
- 16 pixels wide
- × 3 colour channels (implicit)

Picture a little 3D brick — a "tube" — carved out of the video volume. One brick contains 2 × 16 × 16 × 3 = **1,536 raw numbers**.

How many bricks fit in the clip?

```
time  : 16 frames ÷ 2   =  8
height: 224 px    ÷ 16  = 14
width : 224 px    ÷ 16  = 14

total = 8 × 14 × 14 = 1,568 video tokens
```

Those exact three lines are in `models/video_encoder.py`:

```python
self.t_patches = num_frames // patch_size[0]   # 8
self.h_patches = img_size   // patch_size[1]   # 14
self.w_patches = img_size   // patch_size[2]   # 14
self.num_patches = self.t_patches * self.h_patches * self.w_patches   # 1568
```

Notice these are integer divisions — if `img_size` weren't divisible by 16, you'd silently drop pixels. Always check divisibility when you change these numbers.

### The tokenizer is just a convolution

```python
self.patch_embed = nn.Conv3d(
    in_channels=3, out_channels=embed_dim,
    kernel_size=patch_size, stride=patch_size,
)
```

This is the single most elegant trick in the file, and it confuses everyone at first. A convolution normally slides a window with overlap. Here **`kernel_size == stride`**, so the windows tile the volume perfectly with **zero overlap**. Each window position sees exactly one 2×16×16×3 brick and outputs `embed_dim` numbers.

So this `Conv3d` is mathematically identical to: "flatten each brick into a 1,536-vector, then multiply by a learned 1536 × 384 matrix." It's a patch-cutter and a linear projection fused into one op, and it's fast because cuDNN is highly optimised for convolutions.

`embed_dim: 384` is the width of the model — every token, from here until the decoder, is a 384-dimensional vector. (384 with 6 heads = 64 dims per attention head, the standard ViT-Small configuration. `full_train.yaml` bumps this to 768/12 = ViT-Base.)

Output shape: `(B, 384, 8, 14, 14)`. Then:

```python
x = rearrange(x, "b d t h w -> b (t h w) d")     # (B, 1568, 384)
```

`einops.rearrange` flattens the three spatial-temporal axes into one sequence axis. The order `(t h w)` matters enormously — it defines token #0 = (t=0,h=0,w=0), token #1 = (t=0,h=0,w=1), … token #195 = (t=1,h=0,w=0), and so on. **Everything downstream — positional encoding, masking, the reconstruction target — must use this same ordering**, or the model will be scored against the wrong patches. (It is consistent in your code; `patchify_video` uses `-> b (t h w) ...` too. Good.)

### Audio: cutting the spectrogram into squares

`audio_patch_size: [16, 16]` — 16 mel bins × 16 time frames. Same `kernel=stride` trick, but `Conv2d` since the spectrogram is 2D.

```
frequency: 128 mels ÷ 16 = 8
time     : 128 frames ÷ 16 = 8
total = 8 × 8 = 64 audio tokens
```

Each audio patch holds 16 × 16 × 1 = **256** raw log-mel values.

Now look at the imbalance: **1,568 video tokens vs 64 audio tokens**. Video dominates the concatenated decoder sequence 96%/4%. Combined with the audio-cropping bug, audio is close to a rounding error in this model right now. Fixing `target_length` to 512 gets you to 256 audio tokens, which is healthier.

**Topics to study here:** convolution arithmetic (kernel, stride, padding, output size); the Vision Transformer (ViT) patch-embedding idea; ViViT / VideoMAE tubelet embedding; `einops` notation. Meta-topic: **tokenization of continuous signals**.

---

## Part 4 — Positional encoding: telling the model where each token came from

A Transformer's self-attention is **permutation invariant**: shuffle the input tokens and you get the same outputs, shuffled. It has no built-in notion of "token 5 is above token 4". But spatial position obviously matters for video.

The fix: add a position-dependent vector to each token before the Transformer sees it.

`models/positional_encoding.py` implements the classic sinusoidal scheme from *Attention Is All You Need*. For position `p` and dimension index `i`:

```
frequency ω_i = 1 / 10000^(2i/d)
embedding = [ sin(p·ω_0), sin(p·ω_1), …, cos(p·ω_0), cos(p·ω_1), … ]
```

Low dimensions oscillate fast, high dimensions oscillate slowly. Think of it as writing the position in a smooth, continuous binary-like code. The useful property is that the offset between two positions can be expressed as a linear transform of their encodings, so the network can learn *relative* position easily.

These are **fixed, not learned** — hence `register_buffer` rather than `nn.Parameter`. A buffer moves with `.to(device)` and gets saved in checkpoints, but receives no gradients.

### Factorised 3D encoding

Your video has three axes, so `_build_pos_embed` gives each axis its own 1D encoding of size `d = 384 // 3 = 128` and **adds** them:

```python
pe = pe_t[:, None, None, :] + pe_h[None, :, None, :] + pe_w[None, None, :, :]
```

Those `None`s are `np.newaxis` — they insert size-1 axes so PyTorch **broadcasting** expands each term to a full (8, 14, 14, 128) grid before adding. Broadcasting is worth understanding properly; it shows up constantly.

Then `pe.reshape(-1, d)` flattens with the same `(t, h, w)` ordering as the tokens. Consistent. Good.

Audio does the same over (frequency, time) with `d = 384 // 2 = 192`.

**Where it's applied:** in `crossmodal_mae.py`, `x = x + pos_embed` happens **before masking**. This is critical and your code gets it right. If you masked first and added positions after, the surviving tokens would be numbered 0..939 instead of carrying their true original positions, and the model would have no idea where anything was.

### ⚠️ A design weakness in the decoder

`models/decoder.py` builds its positional embeddings as a **flat 1D** encoding over indices 0…1567, losing the 3D structure. More importantly, it then does:

```python
x = torch.cat([video_full, audio_full], dim=1)
```

Video positions run 0…1567 and audio positions run 0…63 — **overlapping ranges, same encoding function**. Video token #7 and audio token #7 get the *identical* positional vector. The decoder has no reliable way to tell "this is an audio token" from "this is a video token."

Fix: add a learned **modality embedding** — one vector added to all video tokens, a different one added to all audio tokens:

```python
self.modality_video = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
self.modality_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
# then: video_full = video_full + self.modality_video, etc.
```

This is exactly what CAV-MAE and similar cross-modal models do. It's a small change with a large effect.

**Topics to study here:** permutation invariance of attention; sinusoidal vs learned positional embeddings; broadcasting; relative and rotary position encodings (RoPE) as modern alternatives.

---

## Part 5 — The masking, line by line

`data/masking.py` is short but dense. It's the standard MAE implementation and the `argsort` trick is genuinely clever. Let me trace it with a toy example.

Suppose `B=1`, `N=5` tokens, `masking_ratio=0.4`.

```python
N_keep = int(N * (1 - masking_ratio))          # int(5 * 0.6) = 3
```

**Step 1 — random priority for each token**

```python
noise = torch.rand(B, N)        # e.g. [0.7, 0.2, 0.9, 0.1, 0.5]
```

**Step 2 — sort**

```python
ids_shuffle = torch.argsort(noise, dim=1)      # [3, 1, 4, 0, 2]
```

`argsort` returns *indices*, not values. Read it as: "the smallest noise is at position 3, next smallest at position 1, then 4, then 0, then 2." So `ids_shuffle` is a **random permutation** of 0..4 — a shuffled order of tokens.

**Steps 3 & 4 — split**

```python
ids_keep = ids_shuffle[:, :3]    # [3, 1, 4]   → these tokens survive
ids_mask = ids_shuffle[:, 3:]    # [0, 2]      → these are hidden
```

**Step 5 — actually gather the surviving tokens**

```python
ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
x_visible = torch.gather(x, dim=1, index=ids_keep_expanded)
```

`torch.gather` picks rows by index. But it needs the index tensor to have the same rank as `x` (which is `(B, N, D)`), so `unsqueeze(-1)` adds a dimension → `(B, 3, 1)` and `.expand(-1, -1, D)` repeats it across all 384 feature dims → `(B, 3, 384)`. The `-1`s mean "leave this dimension alone."

`expand` doesn't copy memory, it just creates a stride-0 view — cheap. Worth knowing the difference between `expand` and `repeat`.

Result: `x_visible` has shape `(B, 3, D)` and contains tokens 3, 1, 4 **in that shuffled order**.

**Step 6 — the binary mask, and the unshuffle map**

```python
mask = torch.ones(B, N)
mask[:, :N_keep] = 0            # [0, 0, 0, 1, 1]  — in SHUFFLED order
ids_restore = torch.argsort(ids_shuffle, dim=1)
mask = torch.gather(mask, dim=1, index=ids_restore)
```

`ids_restore = argsort(ids_shuffle)` is the **inverse permutation**. If `ids_shuffle` says "shuffled slot 0 holds original token 3", then `ids_restore` says "original token 3 lives in shuffled slot 0." Here: `argsort([3,1,4,0,2]) = [3, 1, 4, 0, 2]`... let me actually compute it — the smallest value (0) is at index 3, next (1) at index 1, then (2) at index 4, (3) at index 0, (4) at index 2 → `ids_restore = [3, 1, 4, 0, 2]`. (Coincidentally the same here; in general it isn't.)

Gathering `mask` with `ids_restore` converts the mask from shuffled order back to **original token order**: `[1, 0, 1, 0, 0]` — meaning original tokens 0 and 2 were masked, tokens 1, 3, 4 survived. That matches `ids_keep = [3,1,4]`. ✓

**Convention: `mask == 1` means masked/hidden, `mask == 0` means visible.** This is used in the loss to score *only* the hidden patches.

`ids_restore` is returned because the decoder needs it to put everything back in the right order (Part 7).

### Your real numbers

```
Video: 1568 tokens → keep int(1568 × 0.6) = 940,  mask 628
Audio:   64 tokens → keep int(64 × 0.6)   = 38,   mask 26
```

Note video and audio are masked **independently** (two separate `random_masking` calls in `crossmodal_mae.py`). That's deliberate: sometimes a lip patch is hidden while the corresponding audio is visible, forcing audio→video inference, and vice versa. That cross-modal pressure is the whole point of the design.

**Topics to study here:** `torch.gather` / `scatter`; permutations and inverse permutations; `expand` vs `repeat` vs `reshape`; random vs structured masking (tube masking in VideoMAE, block masking in BEiT).

---

## Part 6 — The encoder: attention over only the visible tokens

`crossmodal_mae.py` Step 3:

```python
for block in self.video_encoder.blocks:
    video_visible = block(video_visible)
video_visible = self.video_encoder.norm(video_visible)
```

Only 940 tokens go in, not 1,568. That's the efficiency win of MAE: self-attention costs **O(N²)**, so cutting tokens by 40% cuts attention cost by ~64%. At 75% masking the saving is ~94% — which is why MAE can train huge models cheaply.

### What's inside a `Block`

`timm.models.vision_transformer.Block` is one standard Transformer encoder layer:

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Two sub-parts, each wrapped in a **residual connection** (the `x +`) and preceded by **LayerNorm** (this is "pre-norm", which trains more stably than the original post-norm design).

**Multi-head self-attention**, in one paragraph: each token produces three vectors — a Query ("what am I looking for"), a Key ("what do I offer"), and a Value ("what I'll pass along"). Every token's Query is dot-producted against every other token's Key to produce a compatibility score; those scores go through a softmax to become weights that sum to 1; each token's output is the weighted average of all Values. The `/√d` scaling before softmax keeps the dot products from growing too large and saturating the softmax. "Multi-head" means doing this 6 times in parallel on 64-dim slices (384/6) so different heads can specialise — one on nearby patches, one on the same patch across time, and so on.

This is the mechanism by which a visible patch of the left cheek can inform the reconstruction of a hidden patch of the mouth.

**MLP**: two linear layers with a GELU in between, expanding to `mlp_ratio=4.0` × 384 = 1536 and back. Attention mixes information *across* tokens; the MLP transforms each token *individually*. Roughly two thirds of the parameters live here.

`qkv_bias=True` just adds bias terms to the Q/K/V projections — a minor ViT convention.

### ⚠️ A structural note

`VideoEncoder.forward()` exists and does patch-embed → pos-embed → blocks → norm, but `CrossModalMAE` **never calls it**. It reaches inside and calls `self.video_encoder.patch_embed(...)` and `self.video_encoder.blocks` directly, because masking has to happen in the middle. That's functional but fragile — if you ever edit `VideoEncoder.forward`, the main model won't notice. Cleaner: give the encoder a `forward(x, mask_ratio)` that does the masking internally and returns `(tokens, mask, ids_restore)`, as the official MAE repo does.

**Topics to study here:** self-attention and multi-head attention; Query/Key/Value; softmax; residual connections and why they enable deep nets; LayerNorm vs BatchNorm; pre-norm vs post-norm; GELU; the `timm` library. Meta-topic: **the Transformer architecture** — this is the single highest-value thing on your list.

---

## Part 7 — The decoder: putting the puzzle back together

`models/decoder.py`. Four moves.

### 1. Narrow the width

```python
self.video_proj = nn.Linear(encoder_embed_dim, decoder_embed_dim)   # 384 → 192
```

The decoder is deliberately smaller and shallower (192 wide, 4 layers vs the encoder's 384/6). Reason: the decoder is **thrown away** after pretraining. You only keep the encoder for downstream tasks. So you spend as little compute on it as you can get away with.

### 2. Insert mask tokens

```python
self.mask_token_video = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
```

One single learned vector, shared across every hidden position. It's a placeholder meaning "something was here, figure out what." It carries no content — all the information about *what* goes here comes from (a) the positional embedding added right after, and (b) attention to the visible tokens.

`nn.Parameter` (not a buffer) means it's **learned** by gradient descent. Initialised with `std=0.02`, the standard ViT init.

### 3. Unshuffle with `ids_restore`

```python
mask_tokens = mask_token.expand(B, N_masked, -1)
full = torch.cat([visible_tokens, mask_tokens], dim=1)     # (B, 1568, 192)
ids = ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1])
full = torch.gather(full, dim=1, index=ids)
full = full + pos_embed
```

Trace it: `visible_tokens` are in *shuffled* order (slots 0…939), then 628 identical mask tokens are appended (slots 940…1567). So `full` is the complete sequence **in shuffled order**. Gathering by `ids_restore` — the inverse permutation from Part 5 — sorts it back into **original spatial order**. Now token #i genuinely corresponds to patch #i of the video.

Only *then* is `pos_embed` added, so every position (visible and masked alike) gets the correct location signal. The mask tokens become distinguishable from each other purely by their position.

### 4. Cross-modal attention, then heads

```python
x = torch.cat([video_full, audio_full], dim=1)      # (B, 1568 + 64, 192)
for block in self.blocks: x = block(x)
```

This concatenation is the cross-modal part of "CrossModalMAE." Because self-attention attends over the whole sequence, a masked video token can now attend to audio tokens and vice versa. A hidden mouth patch can be reconstructed partly from the sound of the phoneme being spoken. That's the representation you actually want.

(This is where the missing modality embedding from Part 4 hurts, and where the 1568:64 imbalance hurts.)

Finally:

```python
pred_video = x[:, :self.num_video_patches, :]
pred_audio = x[:, self.num_video_patches:, :]
pred_video = self.video_head(pred_video)   # Linear(192 → 1536)
pred_audio = self.audio_head(pred_audio)   # Linear(192 → 256)
```

Split back by modality, then project each token to the **raw pixel count of one patch**: 2·16·16·3 = 1536 for video, 16·16·1 = 256 for audio. The output is literally the predicted pixel values, flattened.

**Topics to study here:** encoder–decoder asymmetry in MAE; learned tokens (`[CLS]`, `[MASK]`) as a general idea; `nn.Parameter` vs `register_buffer`; weight initialisation.

---

## Part 8 — The loss: how the guess is scored

`losses/reconstruction.py`.

### Building the target

The target is the original data, chopped the *same way* the tokenizer chopped it:

```python
patches = rearrange(video,
    "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
    pt=2, ph=16, pw=16)
```

Read the left side as a factorisation: the T axis (16) is split into `t=8` groups of `pt=2`; the H axis (224) into `h=14` groups of `ph=16`; same for W. The right side says: sequence axis is `(t h w)` — **matching the encoder's flatten order** — and each token's features are the 1536 raw values of that brick. `(B, 1568, 1536)`, exactly the shape of `pred_video`.

### `norm_pix_loss`

```python
mean = patches.mean(dim=-1, keepdim=True)
var  = patches.var(dim=-1, keepdim=True)
return (patches - mean) / (var + 1e-6).sqrt()
```

Each patch is normalised to zero mean, unit variance **independently**. `dim=-1` means across the 1536 values inside one patch; `keepdim=True` preserves the shape for broadcasting.

Why: without it, MSE is dominated by getting the average brightness right, and the model learns to output blurry grey blobs with the correct mean. Normalising removes the mean and contrast from the target so the loss has to care about **texture and structure**. The MAE paper measured a solid accuracy gain from this. `norm_pix_loss: true` in your config — correct.

Consequence to be aware of: the model no longer predicts absolute pixel values, so if you want to *visualise* reconstructions you must un-normalise using the stored per-patch mean/var. `utils/visualization.py` will need that.

### Masked MSE

```python
loss = (pred - target) ** 2      # (B, N, patch_dim)
loss = loss.mean(dim=-1)         # (B, N)    average within each patch
loss = (loss * mask).sum() / mask.sum()
```

The last line is the important one. `mask` is 1 for hidden patches, 0 for visible. Multiplying zeroes out every visible patch's contribution, and dividing by `mask.sum()` (the number of hidden patches) gives a proper mean over only those.

**Why exclude visible patches:** the encoder saw them. Reconstructing them is copying, not understanding. Scoring them would let the model score well on a trivial task and dilute the gradient from the part that matters.

Then `total_loss = loss_video + loss_audio` in `crossmodal_mae.py`. Both weighted equally — reasonable as a start, but given the 1568:64 token imbalance you may later want a weighting coefficient or to normalise per-modality.

**Topics to study here:** MSE / L2 loss; masked and weighted losses; feature normalisation; `keepdim` and broadcasting rules; why perceptual/adversarial losses are sometimes preferred for reconstruction.

---

## Part 9 — The training loop

`engine/trainer.py`. The standard skeleton is:

```
for each epoch:
  for each batch:
    optimizer.zero_grad()        # clear old gradients
    loss = model(batch)          # forward pass
    loss.backward()              # backpropagation: compute ∂loss/∂parameter
    clip_grad_norm_(...)         # rescale if gradients are too big
    optimizer.step()             # update every parameter
  scheduler.step()               # adjust learning rate
```

**AdamW** (`build_optimizer`): an adaptive optimiser that keeps a running estimate of each parameter's gradient mean and variance, and uses them to give each parameter its own effective step size. The "W" means weight decay is applied *decoupled* from the gradient — mathematically the correct way to do L2 regularisation with Adam. `weight_decay: 0.05` is the ViT/MAE standard.

**Learning rate schedule** (`build_scheduler`): linear **warmup** for 10 epochs, then **cosine annealing** down to `min_lr`. Warmup exists because at initialisation the gradients are large and noisy; a full learning rate immediately can blow up the model. Cosine decay lets the model take big steps early and fine steps at the end. This warmup+cosine pair is the default for essentially every modern Transformer.

**`grad_clip: 1.0`**: if the total gradient norm exceeds 1.0, rescale the whole gradient down. Cheap insurance against one bad batch destroying the run.

**AMP** (`amp: true` in `full_train.yaml`): automatic mixed precision — most operations run in 16-bit floats instead of 32-bit, roughly halving memory and speeding things up on modern GPUs. `GradScaler` multiplies the loss by a large factor before `backward()` so tiny gradients don't underflow to zero in fp16, then divides it out before the update.

**`verify_baseline.py`** is a smart piece of engineering and you should run it before anything else. It feeds **one fixed random batch** 200 times and checks the loss collapses toward zero. This is the **overfit-a-single-batch** test — the fastest way to detect a broken forward pass, a shape mismatch, a detached gradient, or a misaligned target. If a model can't memorise one sample, no amount of data will help. Make this a habit for every model you ever build.

**Topics to study here:** gradient descent and backpropagation; the computational graph / autograd; Adam and AdamW; learning-rate warmup and cosine schedules; gradient clipping; mixed-precision training; epochs vs steps vs batches; `DataLoader`, `num_workers`, `pin_memory`.

---

## Part 10 — Bugs you should fix before you run anything

I found several things that will crash or silently degrade training. Listing them by severity.

**1. `torch.einsum` with parentheses — crashes immediately.** In `crossmodal_mae.py`:

```python
video_tokens = torch.einsum("bdthw->b(thw)d", video_tokens)
audio_tokens = torch.einsum("bdft->b(ft)d", audio_tokens)
```

`torch.einsum` subscripts must be single letters; it cannot merge axes and does not accept `()`. This raises a `RuntimeError` on the very first forward pass. Replace with einops (already imported elsewhere):

```python
from einops import rearrange
video_tokens = rearrange(video_tokens, "b d t h w -> b (t h w) d")
audio_tokens = rearrange(audio_tokens, "b d f t -> b (f t) d")
```

**2. `torchaudio_melspec` is undefined — crashes at model construction.** In `AudioEncoder.__init__`:

```python
self.mel = torch.nn.Sequential(torchaudio_melspec(...))
```

There's no such name imported or defined → `NameError`. And `self.mel` is never used; `_extract_mel` builds its own transform. Just delete the block.

**3. `_extract_mel` rebuilds the MelSpectrogram on every forward call**, and hardcodes `sample_rate=16000, n_fft=400, hop_length=160` instead of using the constructor arguments you carefully passed in from config. Build it once in `__init__` as a submodule:

```python
self.mel_transform = AT.MelSpectrogram(
    sample_rate=sample_rate, n_fft=n_fft,
    hop_length=hop_length, n_mels=n_mels)
```

Then `_extract_mel` just calls `self.mel_transform(waveform)`. This is both a correctness bug (config ignored) and a speed bug.

**4. Audio cropped to 1.28 s while video covers 5 s** (Part 2). Set `target_length: 512` in config and thread it through, or the cross-modal objective is learning on mismatched pairs.

**5. Missing modality embedding in the decoder** (Part 4). Video and audio tokens are indistinguishable to the decoder.

**6. `torch.cuda.amp.GradScaler` is deprecated** in recent PyTorch. Use `torch.amp.GradScaler("cuda", enabled=self.use_amp)`.

**7. `data/transforms.py` is completely empty.** No augmentation, no normalisation. For MAE pretraining the standard recipe is random resized crop + horizontal flip, applied *consistently across all frames of a clip* (you must not flip frame 3 and not frame 4). Worth writing before full training.

**8. `dataset.py` frame decoding is fragile.** When `stream.frames == 0` it exhausts the container counting frames, then `container.seek(0)` — seeking after full decode doesn't reliably reset a PyAV container. It also decodes every frame sequentially and discards most, which is slow. Consider `decord` or PyAV keyframe seeking. Also `src_rate = frame.sample_rate` reads a loop variable after the loop ends — works, but will break confusingly if the loop body ever changes.

**9. Aspect ratio is squashed** by `img.resize((224, 224))`. Centre-crop first for face data.

**10. `verify_baseline.py`'s success criterion** (`final_loss < 0.1`) is tested against a model asked to memorise **pure Gaussian noise** with `norm_pix_loss` on. Random noise has no structure to exploit, so this may legitimately fail to reach 0.1 even with perfectly correct code. Better: assert that the loss **decreased substantially** from step 0, e.g. `final < 0.3 * initial`.

---

## Part 11 — What to study, in order

You said you're new to AI/ML. Here's a dependency-ordered path. Don't try to learn it all before touching the code — learn each layer, then come back and re-read the corresponding Part above.

### Tier 0 — Prerequisites (1–2 weeks)
- **Linear algebra**: vectors, matrices, matrix multiplication, dot products. *3Blue1Brown's "Essence of Linear Algebra"* is the best 4 hours you can spend.
- **Calculus**: partial derivatives and the chain rule. That's it — the chain rule *is* backpropagation.
- **Python + NumPy**: array shapes, indexing, broadcasting.

### Tier 1 — Neural network fundamentals (2–3 weeks)
- Perceptron → MLP → activation functions (ReLU, GELU)
- **Backpropagation** — meta-topic: *automatic differentiation*
- Loss functions, gradient descent, SGD → Adam
- Overfitting, regularisation, train/val split
- **Resource**: Andrej Karpathy's *"Neural Networks: Zero to Hero"* (YouTube). Build micrograd yourself. Nothing else comes close for building real intuition.

### Tier 2 — PyTorch mechanics (1–2 weeks)
- Tensors, `dtype`, devices, `.to(device)`
- `nn.Module`, `forward()`, `nn.Parameter` vs `register_buffer`
- Autograd, `.backward()`, `optimizer.zero_grad()`
- `Dataset` / `DataLoader` / `collate_fn`
- **`einops`** — `rearrange`, `reduce`, `repeat`. Your codebase leans on it heavily.
- `torch.gather`, `expand`, `view` vs `reshape`, broadcasting rules

### Tier 3 — Computer vision (2 weeks)
- Convolutions: kernel, stride, padding, output-size arithmetic
- Why `kernel == stride` = patching (Part 3)
- CNN → ResNet → the idea of residual connections

### Tier 4 — Transformers ⭐ (3–4 weeks — the most important tier)
- **Self-attention**, Q/K/V, scaled dot-product, softmax
- Multi-head attention
- LayerNorm, residuals, pre-norm vs post-norm, the feed-forward block
- Positional encodings: sinusoidal, learned, relative, RoPE
- **Vision Transformer (ViT)** — read the paper *"An Image is Worth 16x16 Words"*
- **Resources**: Karpathy's *"Let's build GPT"*; Jay Alammar's *"The Illustrated Transformer"*; the *Annotated Transformer*.

### Tier 5 — Self-supervised learning (2 weeks)
- The idea of pretext tasks; pretraining → fine-tuning
- **MAE**: *"Masked Autoencoders Are Scalable Vision Learners"* (He et al., 2021) — **read this paper carefully, your code is a direct descendant**
- **VideoMAE** — tube masking, why 90% works for video
- BERT (where masked modelling started), BEiT, SimMIM
- Contrastive learning: SimCLR, InfoNCE, CLIP — relevant because you have `losses/contrastive.py`

### Tier 6 — Audio (2 weeks)
- Sampling rate, Nyquist, quantisation
- Fourier transform → STFT → spectrogram
- Mel scale, log-mel, MFCC
- `torchaudio` transforms
- **AST** (Audio Spectrogram Transformer), **AudioMAE**

### Tier 7 — Multimodal / your actual project (ongoing)
- **CAV-MAE** (*"Contrastive Audio-Visual Masked Autoencoder"*, ICLR 2023) — the closest published work to what you're building. Read it, and read its code.
- AV-HuBERT — audio-visual speech representation learning on LRS3
- Cross-attention vs concatenated self-attention for fusion
- Modality embeddings, modality imbalance, modality collapse

### Tier 8 — Training at scale (as needed)
- LR warmup + cosine annealing; effective batch size
- Mixed precision (AMP), gradient accumulation, gradient checkpointing
- Distributed training (DDP)
- Experiment tracking (Weights & Biases — your config has a `use_wandb` flag)

---

## Part 12 — A suggested order of operations

1. Fix bugs **1, 2, 3** from Part 10. Nothing runs until then.
2. Run `verify_baseline.py` on CPU with `batch_size: 2`. Loosen its pass criterion per bug 10. You want to see the loss drop clearly.
3. Add print statements of `.shape` at every step of `CrossModalMAE.forward`. Confirm with your own eyes: `(2, 3, 16, 224, 224)` → `(2, 1568, 384)` → `(2, 940, 384)` → `(2, 1568, 1536)`. Shape-tracing is the core debugging skill in this field.
4. Fix bugs **4** and **5** (audio length, modality embedding).
5. Get 10 real video files, build a file list, and check `AVDataset[0]` returns the shapes you expect.
6. Overfit those 10 clips to near-zero loss. If that works, the pipeline is sound.
7. Write `utils/visualization.py` to render reconstructions. Seeing blurry-but-recognisable faces emerge is the real proof, and it's the moment this all clicks.
8. Only then scale to the full dataset with `full_train.yaml`.

---

One last thing worth internalising: nearly every line in this codebase is a shape transformation. If you can say out loud what the tensor shape is before and after each line, and why, you understand the model. That skill transfers to every architecture you will ever read.
