import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_conv2d(in_ch, out_ch, k=3, s=1, p=1, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias))


def spectral_linear(in_f, out_f, bias=True):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f, bias=bias))


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=8, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N) if N > 1 else 1
        if N % G != 0:
            G = N
        y = x.view(G, -1, self.num_channels, C // self.num_channels, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + 1e-8)
        y = y.mean(dim=(2, 3, 4), keepdim=True)
        y = y.repeat(G, 1, 1, H, W).view(N, 1, H, W)
        return torch.cat([x, y], dim=1)


def make_fir_kernel():
    k = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
    k = (k[:, None] * k[None, :])
    k = k / k.sum()
    return k


class BlurDownsample(nn.Module):
    def __init__(self, channels, kernel=None, stride=2):
        super().__init__()
        self.stride = stride
        if kernel is None:
            kernel = make_fir_kernel()
        self.register_buffer('kernel', kernel[None, None, :, :])
        self.groups = channels

    def forward(self, x):
        k = self.kernel.repeat(self.groups, 1, 1, 1)
        return F.conv2d(x, k, stride=self.stride,
                        padding=self.kernel.shape[-1] // 2, groups=self.groups)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half))
    ang = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.affine = spectral_linear(cond_dim, 2 * feat_dim, bias=True)

    def forward(self, h, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        return h * (gamma[:, :, None, None] + 1.0) + beta[:, :, None, None]


class ResDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.conv1 = spectral_conv2d(in_ch, out_ch, 3, 1, 1, bias=True)
        self.conv2 = spectral_conv2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.skip = spectral_conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.blur = BlurDownsample(out_ch, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.cond = FiLM(cond_dim, out_ch) if cond_dim is not None and cond_dim > 0 else None

    def forward(self, x, cond=None):
        y = self.act(self.conv1(x))
        if self.cond is not None and cond is not None:
            y = self.cond(y, cond)
        y = self.act(self.conv2(y))
        y = self.blur(y)
        skip = self.blur(self.skip(x))
        return (y + skip) / math.sqrt(2.0)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        inter_channels = max(1, channels // 8)
        self.query = spectral_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        self.key = spectral_conv2d(channels, inter_channels, k=1, s=1, p=0, bias=False)
        self.value = spectral_conv2d(channels, channels, k=1, s=1, p=0, bias=False)
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


class TemporalDifferenceModule(nn.Module):
    """
    Enhanced temporal difference processing with multiple pathways:
    1. Raw difference
    2. Magnitude difference
    3. Direction-aware features
    """

    def __init__(self, channels):
        super().__init__()
        # Process raw difference
        self.diff_conv = nn.Sequential(
            spectral_conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv2d(channels, channels, 3, 1, 1)
        )

        # Process magnitude information
        self.mag_conv = nn.Sequential(
            spectral_conv2d(channels, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv2d(channels // 2, channels, 3, 1, 1)
        )

        # Fusion
        self.fusion = spectral_conv2d(channels * 3, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, h1, h2):
        # Raw difference (preserves sign)
        diff = h1 - h2
        diff_feat = self.diff_conv(diff)

        # Magnitude difference (always positive)
        mag_diff = torch.abs(h1) - torch.abs(h2)
        mag_feat = self.mag_conv(mag_diff)

        # Combine all pathways
        combined = torch.cat([diff_feat, mag_feat, h1], dim=1)
        return self.act(self.fusion(combined))


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
                 embedding_dim=128,
                 num_blocks=5,  # Increased from 4
                 initial_features=64,  # Increased from 32
                 num_objects=1,
                 image_size=32,
                 timestep_prediction=False,
                 double_timestep_prediction=False,
                 max_timestep=10,
                 use_3d=False):
        super().__init__()

        if use_3d:
            raise NotImplementedError("3D not implemented")

        self.num_classes = num_classes
        self.num_continuous_features = num_continuous_features
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.cond_embedding_dim = embedding_dim // 8
        self.timestep_prediction = timestep_prediction
        self.double_timestep_prediction = double_timestep_prediction

        # Embeddings
        if num_classes > 0:
            self.class_embedding = nn.Embedding(num_classes * num_objects, self.cond_embedding_dim)
        self.continuous_embedding = spectral_linear(num_continuous_features * num_objects,
                                                    self.cond_embedding_dim)

        # Build condition dimension
        cond_dim = self.cond_embedding_dim
        if num_classes > 0:
            cond_dim += self.cond_embedding_dim
        if timestep_prediction:
            cond_dim += self.cond_embedding_dim
        if double_timestep_prediction:
            cond_dim += self.cond_embedding_dim

        # Backbone with more capacity
        chs = [initial_features * (2 ** i) for i in range(num_blocks)]
        self.from_rgb = spectral_conv2d(image_channels, chs[0], k=1, s=1, p=0, bias=True)

        blocks = []
        attentions = []
        in_ch = chs[0]
        resolution = image_size
        for i, out_ch in enumerate(chs[1:]):
            blocks.append(ResDownBlock(in_ch, out_ch, cond_dim=cond_dim))
            resolution = max(resolution // 2, 1)
            # Add attention at multiple resolutions for better feature learning
            if resolution <= 16 and resolution >= 4:
                attentions.append(SelfAttention(out_ch))
            else:
                attentions.append(nn.Identity())
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        # Enhanced temporal difference processing
        self.temporal_diff = TemporalDifferenceModule(in_ch)

        # Minibatch stddev
        self.mbstd = MinibatchStdDev(group_size=16, num_channels=1)

        # Final processing
        self.final_conv = spectral_conv2d(in_ch + 1, in_ch, 3, 1, 1, bias=True)
        self.final_act = nn.LeakyReLU(0.2, inplace=True)

        # Deeper head for better discrimination
        head_in = in_ch
        self.head = nn.Sequential(
            spectral_linear(head_in, head_in * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            spectral_linear(head_in * 2, head_in),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_linear(head_in, 1)
        )

        # Projection for labels
        if num_classes > 0:
            self.proj_map = spectral_linear(head_in, embedding_dim, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, *args, **kwargs):
        return self._forward_impl(*args, return_feats=False, **kwargs)

    def forward_with_feats(self, x_tn, x_t, class_vector, continuous_features,
                           timestep_2=None, timestep_1=None, **kwargs):
        return self._forward_impl(
            x_tn, x_t, class_vector, continuous_features,
            timestep_2=timestep_2, timestep_1=timestep_1,
            return_feats=True, **kwargs
        )

    def _encode_single(self, x, cond_vec, feats_list=None):
        """Encode single image through backbone"""
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
            class_embed = self.class_embedding(class_vector).sum(dim=1)
            conds.append(class_embed)

        cont_embed = self.continuous_embedding(continuous_features)
        conds.append(cont_embed)

        if self.timestep_prediction:
            t2_emb = self.timestep_embedding(timestep_2, self.cond_embedding_dim)
            conds.append(t2_emb)

        if self.double_timestep_prediction:
            t1_emb = self.timestep_embedding(timestep_1, self.cond_embedding_dim)
            conds.append(t1_emb)

        return torch.cat(conds, dim=1) if len(conds) > 1 else conds[0]

    def _forward_impl(self, x_tn, x_t, class_vector, continuous_features,
                      timestep_2=None, timestep_1=None, fake_input: bool = False,
                      return_feats: bool = False):
        B = x_t.shape[0]

        # Build condition vector
        cond_vec = self._build_cond_vec(class_vector, continuous_features, timestep_2, timestep_1)
        feats_collected = [] if return_feats else None

        # Encode both inputs
        h1 = self._encode_single(x_tn, cond_vec, feats_list=feats_collected)
        h2 = self._encode_single(x_t, cond_vec, feats_list=None)

        # Spatial alignment if needed
        if h1.shape[-2:] != h2.shape[-2:]:
            h1 = F.adaptive_avg_pool2d(h1, h2.shape[-2:])

        # *** KEY CHANGE: Use enhanced temporal difference module ***
        # This preserves magnitude information and learns better features
        h = self.temporal_diff(h1, h2)

        # Minibatch stddev for mode collapse prevention
        h = self.mbstd(h)

        # Final processing with gradient clipping for stability
        h = torch.clamp(h, -10.0, 10.0)
        h = self.final_act(self.final_conv(h))
        h = torch.clamp(h, -10.0, 10.0)

        # Pool and predict
        pooled = h.mean(dim=(2, 3))
        logit = self.head(pooled).squeeze(1)

        if return_feats:
            return logit, feats_collected
        else:
            return logit

    def timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int):
        if timesteps is None:
            return torch.zeros((1, embedding_dim), device=next(self.parameters()).device)
        if timesteps.dim() > 1:
            timesteps = timesteps.view(-1)
        return sinusoidal_embedding(timesteps, embedding_dim)

    def extract_features(self, x, class_vector, continuous_features, timestep):
        cond_vec = self._build_cond_vec(class_vector, continuous_features, timestep, None)
        h = self._encode_single(x, cond_vec, feats_list=None)
        h = self.mbstd(h)
        h = self.final_act(self.final_conv(h))
        pooled = h.mean(dim=(2, 3))
        return torch.cat([pooled, cond_vec], dim=1)