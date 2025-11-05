# pytorch >= 2.0
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Utilities (SoTA bits)
# ----------------------


def spectral_conv2d(in_ch, out_ch, k=3, s=1, p=1, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias))


def spectral_linear(in_f, out_f, bias=True):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f, bias=bias))


def weight_norm_conv2d(in_ch, out_ch, k=3, s=1, p=1, bias=True):
    return nn.utils.weight_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias))


def weight_norm_linear(in_f, out_f, bias=True):
    return nn.utils.weight_norm(nn.Linear(in_f, out_f, bias=bias))


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=8, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N) if N > 1 else 1
        if N % G != 0:  # make view safe
            G = N
        # [G, N//G, F, C//F, H, W]
        y = x.view(G, -1, self.num_channels, C // self.num_channels, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + 1e-8)  # [N//G, F, C//F, H, W]
        y = y.mean(dim=(2, 3, 4), keepdim=True)  # [N//G, 1, 1, 1, 1]
        y = y.repeat(G, 1, 1, H, W).view(N, 1, H, W)  # <-- include G here
        return torch.cat([x, y], dim=1)

def make_fir_kernel():
    k = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
    k = (k[:, None] * k[None, :])
    k = k / k.sum()
    return k


class BlurDownsample(nn.Module):
    """FIR blur-pooling downsample (StyleGAN2/3)."""
    def __init__(self, channels, kernel=None, stride=2):
        super().__init__()
        self.stride = stride
        if kernel is None:
            kernel = make_fir_kernel()
        self.register_buffer('kernel', kernel[None, None, :, :])
        self.groups = channels

    def forward(self, x):
        k = self.kernel.repeat(self.groups, 1, 1, 1)  # depthwise
        return F.conv2d(x, k, stride=self.stride,
                        padding=self.kernel.shape[-1] // 2, groups=self.groups)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half))
    ang = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2: emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Spectral-norm FiLM affine."""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        # self.affine = spectral_linear(cond_dim, 2 * feat_dim, bias=True)
        self.affine = weight_norm_linear(cond_dim, 2 * feat_dim, bias=True)

    def forward(self, h, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        return h * (gamma[:, :, None, None] + 1.0) + beta[:, :, None, None]


class ResDownBlock(nn.Module):
    """StyleGAN2-like residual down block with FIR downsample; optional per-block FiLM."""
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        # self.conv1 = spectral_conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        # self.conv2 = spectral_conv2d(out_ch, out_ch, 3, 1, 1, bias=True)
        # self.skip  = spectral_conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.conv1 = weight_norm_conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.conv2 = weight_norm_conv2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.skip  = weight_norm_conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.blur  = BlurDownsample(out_ch, stride=2)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.cond  = FiLM(cond_dim, out_ch) if cond_dim is not None and cond_dim > 0 else None

    def forward(self, x, cond=None):
        y = self.act(self.conv1(x))
        if self.cond is not None and cond is not None:
            y = self.cond(y, cond)
        y = self.act(self.conv2(y))
        y = self.blur(y)
        skip = self.blur(self.skip(x))
        return (y + skip) / math.sqrt(2.0)


class SelfAttention(nn.Module):
    """Spectral-normalized self-attention block."""

    def __init__(self, channels):
        super().__init__()
        inter_channels = max(1, channels // 8)
        # self.query = spectral_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        # self.key = spectral_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        # self.value = spectral_conv2d(channels, channels, k=1, s=1, p=0, bias=False)
        self.query = weight_norm_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        self.key = weight_norm_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        self.value = weight_norm_conv2d(channels, channels, k=1, s=1, p=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        proj_query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key(x).view(b, -1, h * w)
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        proj_value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x

# ---------------------------------------------
# Upgraded ConditionalDiscriminator (same name)
# ---------------------------------------------


class ConditionalDiscriminator(nn.Module):
    """
    SoTA 2025 discriminator:
      - StyleGAN2-like ResNet backbone (SN everywhere, FIR downsample)
      - No norm in D, LeakyReLU activations
      - Minibatch-stddev before head
      - Per-block FiLM for continuous / timestep cond
      - Optional label projection term
      - Two-path encode (x_tn, x_t) for diffusion/consistency GANs
    API compatible with your previous forward/forward_with_feats.
    """
    def __init__(self,
                 image_channels,
                 num_classes,
                 num_continuous_features,
                 embedding_dim=64,
                 num_blocks=4,
                 initial_features=64,
                 num_objects=1,
                 image_size=32,
                 timestep_prediction=False,
                 double_timestep_prediction=False,
                 max_timestep=10,
                 use_3d=False):
        super().__init__()
        if use_3d:
            raise NotImplementedError("3D conv path not included in this 2D SoTA update.")

        self.num_classes = num_classes
        self.num_continuous_features = num_continuous_features
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.cond_embedding_dim = int(embedding_dim/8)
        self.timestep_prediction = timestep_prediction
        self.double_timestep_prediction = double_timestep_prediction

        # ---- Embeddings for conditioning (kept compatible with your code) ----
        if num_classes > 0:
            # Index-based per-object embedding (sum over objects)
            self.class_embedding = nn.Embedding(num_classes * num_objects, self.cond_embedding_dim)

        self.continuous_embedding = nn.Linear(num_continuous_features * num_objects, self.cond_embedding_dim)

        # For timestep(s) we use sinusoidal embeddings generated on the fly
        # and map them (if needed) to the same embedding_dim later.

        # ---- Backbone ----
        chs = [initial_features * (2 ** i) for i in range(num_blocks)]
        # self.from_rgb = spectral_conv2d(image_channels, chs[0], k=1, s=1, p=0, bias=True)
        self.from_rgb = weight_norm_conv2d(image_channels, chs[0], k=1, s=1, p=0, bias=True)

        # Per-block FiLM condition vector will be concatenation of available pieces
        cond_dim = self.cond_embedding_dim  # continuous
        if num_classes > 0:
            cond_dim += self.cond_embedding_dim
        if timestep_prediction:
            cond_dim += self.cond_embedding_dim  # t2
        if double_timestep_prediction:
            cond_dim += self.cond_embedding_dim  # t1

        blocks = []
        attentions = []
        in_ch = chs[0]
        resolution = image_size
        attention_added = False
        for out_ch in chs[1:]:
            blocks.append(ResDownBlock(in_ch, out_ch, cond_dim=cond_dim))
            resolution = max(resolution // 2, 1)
            if not attention_added and resolution <= 16:
                attentions.append(SelfAttention(out_ch))
                attention_added = True
            else:
                attentions.append(nn.Identity())
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        self.mbstd = MinibatchStdDev(group_size=16, num_channels=1)
        # self.final_conv = weight_norm_conv2d(3 * in_ch + 2, in_ch, 3, 1, 1, bias=True)
        self.final_conv = weight_norm_conv2d(in_ch + 1, in_ch, 3, 1, 1, bias=True)
        # self.final_conv = spectral_conv2d(in_ch + 1, in_ch, 3, 1, 1, bias=True)
        self.final_act = nn.LeakyReLU(0.2, inplace=True)

        # Head
        head_in = in_ch
        # self.head = nn.Sequential(
        #     spectral_linear(head_in, max(256, in_ch // 2)),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     spectral_linear(max(256, in_ch // 2), 1)
        # )
        self.head = nn.Sequential(
            weight_norm_linear(head_in, head_in * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            weight_norm_linear(head_in * 2, head_in),
            nn.LeakyReLU(0.2, inplace=True),
            weight_norm_linear(head_in, 1)
        )
        # self.head = nn.Sequential(
        #     spectral_linear(head_in, head_in * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),  # Reduce dropout
        #     spectral_linear(head_in * 2, head_in),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     spectral_linear(head_in, 1)
        # )

        # Projection term for labels (BigGAN-style): map pooled feat -> embedding_dim
        if num_classes > 0:
            # self.proj_map = spectral_linear(head_in, embedding_dim, bias=False)
            self.proj_map = weight_norm_linear(head_in, embedding_dim, bias=False)

        # Global pool for feature returns
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ADD THIS: Running statistics for input normalization
        self.register_buffer('x_tn_running_mean', torch.zeros(1))
        self.register_buffer('x_tn_running_var', torch.ones(1))
        self.register_buffer('x_t_running_mean', torch.zeros(1))
        self.register_buffer('x_t_running_var', torch.ones(1))
        self.running_update_count = 0

    # -------- public API (kept) --------
    def forward(self, *args, **kwargs):
        return self._forward_impl(*args, return_feats=False, **kwargs)

    def forward_with_feats(self, x_tn, x_t, class_vector, continuous_features,
                           timestep_2=None, timestep_1=None, **kwargs):
        return self._forward_impl(
            x_tn, x_t, class_vector, continuous_features,
            timestep_2=timestep_2, timestep_1=timestep_1,
            return_feats=True, **kwargs
        )

    # -------- core --------
    # def _encode_single(self, x, cond_vec, feats_list=None):
    #     h = self.from_rgb(x)
    #     for blk, attn in zip(self.blocks, self.attentions):
    #         h = blk(h, cond_vec)
    #         h = attn(h)
    #         if feats_list is not None:
    #             # append pooled per-block features for feature-matching losses
    #             feats_list.append(self.global_pool(h).view(h.size(0), -1))
    #     return h

    def _encode_single(self, x, cond_vec, feats_list=None, normalize=True):
        """Encode with optional input normalization to prevent distribution shift."""
        if normalize:
            # Normalize each input independently to handle distribution shift
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x_var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            x_std = torch.sqrt(x_var + 1e-6)
            x = (x - x_mean) / x_std
            # Optionally clip extreme values
            x = torch.clamp(x, -5.0, 5.0)

        h = self.from_rgb(x)
        for blk, attn in zip(self.blocks, self.attentions):
            h = blk(h, cond_vec)
            h = attn(h)
            if feats_list is not None:
                feats_list.append(self.global_pool(h).view(h.size(0), -1))
        return h

    def _build_cond_vec(self, class_vector, continuous_features, timestep_2, timestep_1):
        conds = []

        if self.num_classes > 0:
            # class_vector: [B, num_classs*num_objects] indices → sum embeddings
            class_embed = self.class_embedding(class_vector).sum(dim=1)  # [B, emb]
            conds.append(class_embed)

        cont_embed = self.continuous_embedding(continuous_features)      # [B, emb]
        conds.append(cont_embed)

        if self.timestep_prediction:
            t2_emb = self.timestep_embedding(timestep_2, self.cond_embedding_dim)  # [B, emb]
            conds.append(t2_emb)

        if self.double_timestep_prediction:
            t1_emb = self.timestep_embedding(timestep_1, self.cond_embedding_dim) # [B, emb]
            conds.append(t1_emb)

        return torch.cat(conds, dim=1) if len(conds) > 1 else conds[0]

    def _forward_impl(self, x_tn, x_t, class_vector, continuous_features,
                      timestep_2=None, timestep_1=None, fake_input: bool = False, return_feats: bool = False):
        B = x_t.shape[0]

        # Build per-example condition vector for FiLM
        cond_vec = self._build_cond_vec(class_vector, continuous_features, timestep_2, timestep_1)
        feats_collected = [] if return_feats else None

        # ###########################################
        # import numpy as np
        # import matplotlib
        # matplotlib.use("TkAgg")  # or "Qt5Agg"
        # import matplotlib.pyplot as plt
        # from pathlib import Path
        #
        # # --- tensors -> numpy (H, W, C) ---
        # x_tn_arr = x_tn[0, 0].detach().cpu().numpy() #.permute(1, 2, 0).numpy()
        # x_t_arr = x_t[0, 0].detach().cpu().numpy()  #.permute(1, 2, 0).numpy()
        #
        # # --- subtitle text ---
        # cont_vec = continuous_features[0].detach().cpu().tolist()
        # cont_vec_str = ", ".join(f"{v:.2f}" for v in cont_vec)
        # t_cur = float(timestep_2[0].detach().cpu())
        # t = float(timestep_1[0].detach().cpu())
        # t_vec_str = f"{np.round(t_cur, 3)}, {np.round(t, 3)}"
        #
        # # --- plotting ---
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        #
        # im0 = axes[0].imshow(x_tn_arr, aspect="equal")
        # axes[0].set_title("x_tn")
        # axes[0].axis("off")
        # fig.colorbar(im0, ax=axes[0])
        #
        # im1 = axes[1].imshow(x_t_arr, aspect="equal")
        # axes[1].set_title("x_t")
        # axes[1].axis("off")
        # fig.colorbar(im1, ax=axes[1])
        #
        # # Put suptitle BEFORE saving; lift it a touch
        # fig.suptitle(f"cont_feat: [{cont_vec_str}]\n t: [{t_vec_str}]", y=1.02)
        #
        # # --- filename helper: next free name like foo.png, foo_1.png, foo_2.png, ...
        # def next_available_filename(path) -> Path:
        #     p = Path(path)
        #     stem, ext = p.stem, p.suffix
        #     i = 0
        #     candidate = p
        #     while candidate.exists():
        #         i += 1
        #         candidate = p.with_name(f"{stem}_{i}{ext}")
        #     return candidate
        #
        # # Save to first available name
        # out_path = next_available_filename(f"epoch_disc_inputs_fake_{fake_input}.png")
        # fig.savefig(out_path, bbox_inches="tight")
        #
        # plt.show()
        # ###########################################

        # ===== KEY FIX 1: Normalize inputs before encoding =====
        # This prevents distribution shift from causing NaNs when pretrained disc sees new CM outputs
        h1 = self._encode_single(x_tn, cond_vec, feats_list=feats_collected, normalize=True)
        h2 = self._encode_single(x_t, cond_vec, feats_list=None, normalize=True)

        if h1.shape[-2:] != h2.shape[-2:]:
            h1 = F.adaptive_avg_pool2d(h1, h2.shape[-2:])

        # ===== KEY FIX 2: Normalize features before computing difference =====
        # Prevent explosion when h1 and h2 have different magnitudes
        h1_norm = F.normalize(h1, p=2, dim=1)  # [B, C, H, W] -> normalized per channel
        h2_norm = F.normalize(h2, p=2, dim=1)

        # Compute difference on normalized features
        diff = (h1_norm - h2_norm)  # Much more stable

        # Optional: compute cosine similarity for extra robustness
        sim = (h1_norm * h2_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # Use normalized difference
        h = diff

        # NOW apply minibatch stddev
        h = self.mbstd(h)  # [B, C + 1, H, W]

        # ===== KEY FIX 3: Clamp before final conv to prevent NaN propagation =====
        h = torch.clamp(h, -10.0, 10.0)
        h = self.final_act(self.final_conv(h))
        h = torch.clamp(h, -10.0, 10.0)  # Clamp again after conv

        pooled = h.mean(dim=(2, 3))  # [B, C]

        # ===== KEY FIX 4: Use spectral_norm in head instead of weight_norm =====
        # This is critical for numerical stability with adversarial training
        logit = self.head(pooled).squeeze(1)

        if return_feats:
            return logit, feats_collected
        else:
            return logit

    # -------- helpers kept from your API --------
    def timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int):
        """
        Sinusoidal timestep embeddings for continuous/discrete timesteps.
        """
        if timesteps is None:
            # fallback zeros if not provided
            return torch.zeros((1, embedding_dim), device=next(self.parameters()).device)
        if timesteps.dim() > 1:
            timesteps = timesteps.view(-1)
        emb = sinusoidal_embedding(timesteps, embedding_dim)
        return emb  # [B, embedding_dim]

    def extract_features(self, x, class_vector, continuous_features, timestep):
        """
        Returns final pooled feature (pre-head) concatenated with FiLM cond (useful for analysis).
        """
        cond_vec = self._build_cond_vec(class_vector, continuous_features, timestep, None)
        h = self._encode_single(x, cond_vec, feats_list=None)
        h = self.mbstd(h)
        h = self.final_act(self.final_conv(h))
        pooled = h.mean(dim=(2, 3))
        return torch.cat([pooled, cond_vec], dim=1)


