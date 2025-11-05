import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# Normalization Strategy
# ===========================

def adaptive_conv2d(in_ch, out_ch, k=3, s=1, p=1, bias=True, use_spectral=False):
    """Adaptive normalization: weight_norm for simple data, spectral for complex."""
    conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
    if use_spectral:
        return nn.utils.spectral_norm(conv)
    else:
        return nn.utils.weight_norm(conv)


def adaptive_linear(in_f, out_f, bias=True, use_spectral=False):
    """Adaptive normalization for linear layers."""
    linear = nn.Linear(in_f, out_f, bias=bias)
    if use_spectral:
        return nn.utils.spectral_norm(linear)
    else:
        return nn.utils.weight_norm(linear)


# ===========================
# Building Blocks
# ===========================

class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation (StyleGAN2)."""

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


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""

    def __init__(self, cond_dim, feat_dim, use_spectral=False):
        super().__init__()
        self.affine = adaptive_linear(cond_dim, 2 * feat_dim, bias=True, use_spectral=use_spectral)

    def forward(self, h, cond):
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        return h * (gamma[:, :, None, None] + 1.0) + beta[:, :, None, None]


class ResDownBlock(nn.Module):
    """Residual downsampling block with optional FiLM conditioning."""

    def __init__(self, in_ch, out_ch, cond_dim=None, use_spectral=False):
        super().__init__()
        self.conv1 = adaptive_conv2d(in_ch, out_ch, 3, 1, 1, use_spectral=use_spectral)
        self.conv2 = adaptive_conv2d(out_ch, out_ch, 3, 1, 1, use_spectral=use_spectral)
        self.skip = adaptive_conv2d(in_ch, out_ch, 1, 1, 0, use_spectral=use_spectral)
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.cond = FiLM(cond_dim, out_ch, use_spectral) if cond_dim else None

    def forward(self, x, cond=None):
        y = self.act(self.conv1(x))
        if self.cond is not None and cond is not None:
            y = self.cond(y, cond)
        y = self.act(self.conv2(y))
        y = self.downsample(y)
        skip = self.downsample(self.skip(x))
        return (y + skip) / math.sqrt(2.0)


class EfficientAttention(nn.Module):
    """Efficient multi-head self-attention."""

    def __init__(self, channels, num_heads=4, use_spectral=False):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = adaptive_conv2d(channels, channels * 3, 1, 1, 0, bias=False, use_spectral=use_spectral)
        self.proj = adaptive_conv2d(channels, channels, 1, 1, 0, use_spectral=use_spectral)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, HW]

        # Efficient attention with PyTorch 2.0+
        q = q.transpose(-2, -1)  # [B, heads, HW, head_dim]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(-2, -1).reshape(B, C, H, W)

        return self.gamma * self.proj(attn) + x


# ===========================
# Dual-Path Discriminator
# ===========================

class DualContrastiveHead(nn.Module):
    """Contrastive head for (x_tn, x_t) pairs in consistency models."""

    def __init__(self, feature_dim, projection_dim=256, temperature=0.1, use_spectral=False):
        super().__init__()
        self.projector = nn.Sequential(
            adaptive_linear(feature_dim, projection_dim, use_spectral=use_spectral),
            nn.ReLU(inplace=True),
            adaptive_linear(projection_dim, projection_dim, use_spectral=use_spectral)
        )
        self.temperature = temperature

    def forward(self, h1, h2):
        """
        h1, h2: [B, D] feature vectors
        Returns: contrastive loss (scalar)
        """
        z1 = F.normalize(self.projector(h1), dim=1)
        z2 = F.normalize(self.projector(h2), dim=1)

        # Contrastive loss: matching pairs should have high similarity
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class ConditionalDiscriminator(nn.Module):
    """
    Progressive conditional discriminator for geometry → real image training.

    Features:
    - Adaptive normalization (weight_norm → spectral_norm)
    - Dual-path encoding for (x_tn, x_t) pairs
    - Optional contrastive loss for consistency models
    - Multi-resolution features for feature matching
    - No input normalization (learns distribution directly)
    - Hinge loss compatible
    """

    def __init__(
            self,
            image_channels=1,
            num_classes=4,
            num_continuous_features=4,
            embedding_dim=128,
            num_blocks=4,
            initial_features=64,
            num_objects=1,
            image_size=256,
            timestep_prediction=True,
            double_timestep_prediction=True,
            use_spectral_norm=False,  # Start False, enable for real images
            use_contrastive=False,
            contrastive_weight=0.5,
            **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_continuous_features = num_continuous_features
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.cond_embedding_dim = max(16, embedding_dim // 8)
        self.timestep_prediction = timestep_prediction
        self.double_timestep_prediction = double_timestep_prediction
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.use_spectral_norm = use_spectral_norm

        # Conditioning embeddings
        if num_classes > 0:
            self.class_embedding = nn.Embedding(
                num_classes * num_objects,
                self.cond_embedding_dim
            )

        self.continuous_embedding = nn.Linear(
            num_continuous_features * num_objects,
            self.cond_embedding_dim
        )

        # Build condition dimension
        cond_dim = self.cond_embedding_dim  # continuous
        if num_classes > 0:
            cond_dim += self.cond_embedding_dim  # class
        if timestep_prediction:
            cond_dim += self.cond_embedding_dim  # t2
        if double_timestep_prediction:
            cond_dim += self.cond_embedding_dim  # t1

        # Encoder backbone
        chs = [initial_features * (2 ** i) for i in range(num_blocks)]
        self.from_rgb = adaptive_conv2d(
            image_channels, chs[0], k=3, s=1, p=1,
            use_spectral=use_spectral_norm
        )

        blocks = []
        attentions = []
        in_ch = chs[0]
        resolution = image_size

        for i, out_ch in enumerate(chs[1:], 1):
            blocks.append(ResDownBlock(in_ch, out_ch, cond_dim, use_spectral_norm))
            resolution = resolution // 2

            # Add attention at medium resolution
            if resolution <= 32 and resolution > 8:
                attentions.append(EfficientAttention(out_ch, num_heads=4, use_spectral=use_spectral_norm))
            else:
                attentions.append(nn.Identity())

            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        # Feature processing
        self.mbstd = MinibatchStdDev(group_size=16, num_channels=1)
        self.final_conv = adaptive_conv2d(
            in_ch + 1, in_ch, 3, 1, 1,
            use_spectral=use_spectral_norm
        )
        self.final_act = nn.LeakyReLU(0.2, inplace=True)

        # Discriminator head (no dropout!)
        self.head = nn.Sequential(
            adaptive_linear(in_ch, in_ch, use_spectral=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            adaptive_linear(in_ch, 1, use_spectral=use_spectral_norm)
        )

        # Contrastive head for consistency model training
        if use_contrastive:
            self.contrastive_head = DualContrastiveHead(
                in_ch, projection_dim=256,
                use_spectral=use_spectral_norm
            )

        # Global pooling for features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, *args, **kwargs):
        """Standard forward without feature extraction."""
        return self._forward_impl(*args, return_feats=False, **kwargs)

    def forward_with_feats(self, x_tn, x_t, class_vector, continuous_features,
                           timestep_2=None, timestep_1=None, **kwargs):
        """Forward with intermediate features for feature matching loss."""
        return self._forward_impl(
            x_tn, x_t, class_vector, continuous_features,
            timestep_2=timestep_2, timestep_1=timestep_1,
            return_feats=True, **kwargs
        )

    def _encode_single(self, x, cond_vec, feats_list=None):
        """
        Encode single image path.

        CRITICAL: No input normalization!
        Let discriminator learn the true data distribution.
        """
        h = self.from_rgb(x)

        for blk, attn in zip(self.blocks, self.attentions):
            h = blk(h, cond_vec)
            h = attn(h)

            if feats_list is not None:
                # Collect multi-scale features for feature matching
                feats_list.append(self.global_pool(h).view(h.size(0), -1))

        return h

    def _build_cond_vec(self, class_vector, continuous_features, timestep_2, timestep_1):
        """Build conditioning vector from all available signals."""
        conds = []

        if self.num_classes > 0 and class_vector is not None:
            class_embed = self.class_embedding(class_vector).sum(dim=1)
            conds.append(class_embed)

        if continuous_features is not None:
            cont_embed = self.continuous_embedding(continuous_features)
            conds.append(cont_embed)

        if self.timestep_prediction and timestep_2 is not None:
            t2_emb = self._timestep_embedding(timestep_2, self.cond_embedding_dim)
            conds.append(t2_emb)

        if self.double_timestep_prediction and timestep_1 is not None:
            t1_emb = self._timestep_embedding(timestep_1, self.cond_embedding_dim)
            conds.append(t1_emb)

        return torch.cat(conds, dim=1) if len(conds) > 1 else conds[0]

    def _forward_impl(self, x_tn, x_t, class_vector, continuous_features,
                      timestep_2=None, timestep_1=None, return_feats=False, **kwargs):
        """
        Core forward implementation.

        Args:
            x_tn: Consistency model output at timestep n
            x_t: Target image at timestep t
            class_vector: Shape class indices [B, num_objects]
            continuous_features: Position, size, rotation [B, num_continuous * num_objects]
            timestep_2, timestep_1: Timesteps for dual timestep prediction
            return_feats: Whether to return intermediate features

        Returns:
            If return_feats=False: logits [B]
            If return_feats=True: (logits [B], features [list], contrastive_loss [scalar or None])
        """
        B = x_t.shape[0]

        # Build conditioning vector
        cond_vec = self._build_cond_vec(class_vector, continuous_features, timestep_2, timestep_1)

        # Encode both paths
        feats_collected = [] if return_feats else None

        h1 = self._encode_single(x_tn, cond_vec, feats_list=feats_collected)
        h2 = self._encode_single(x_t, cond_vec, feats_list=None)

        # Align spatial dimensions if needed
        if h1.shape[-2:] != h2.shape[-2:]:
            h1 = F.adaptive_avg_pool2d(h1, h2.shape[-2:])

        # Compute difference (no normalization for binary/simple images)
        diff = h1 - h2

        # Apply minibatch stddev
        h = self.mbstd(diff)

        # Final processing
        h = self.final_act(self.final_conv(h))
        pooled = h.mean(dim=(2, 3))  # [B, C]

        # Discriminator logits
        logit = self.head(pooled).squeeze(1)  # [B]

        # Compute contrastive loss if enabled
        contrastive_loss = None
        if self.use_contrastive and return_feats:
            h1_pooled = self.global_pool(h1).view(B, -1)
            h2_pooled = self.global_pool(h2).view(B, -1)
            contrastive_loss = self.contrastive_head(h1_pooled, h2_pooled)

        if return_feats and self.use_contrastive:
            return logit, feats_collected, contrastive_loss
        elif return_feats:
            return logit, feats_collected
        else:
            return logit

    def _timestep_embedding(self, timesteps, embedding_dim):
        """Sinusoidal timestep embeddings."""
        if timesteps is None:
            device = next(self.parameters()).device
            return torch.zeros((1, embedding_dim), device=device)

        if timesteps.dim() > 1:
            timesteps = timesteps.view(-1)

        half = embedding_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device) / max(1, half)
        )
        ang = timesteps.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

        if embedding_dim % 2:
            emb = F.pad(emb, (0, 1))

        return emb

    def switch_to_spectral_norm(self):
        """
        Switch from weight_norm to spectral_norm for fine-tuning on real images.
        Call this after Phase 2 training before loading real image data.
        """
        print("Switching discriminator to spectral normalization...")
        self.use_spectral_norm = True

        # Rebuild all normalized layers
        # Note: This is a simplified version. In practice, you'd want to
        # preserve learned weights when switching norms
        self._rebuild_with_spectral_norm()

    def _rebuild_with_spectral_norm(self):
        """Helper to rebuild layers with spectral norm (implement if needed)."""
        # This would require carefully transferring weights
        # For simplicity, just retrain when switching norms
        pass


# ===========================
# Training Utilities
# ===========================

def hinge_loss_discriminator(real_pred, fake_pred):
    """Hinge loss for discriminator (works well with simple images)."""
    loss_real = torch.mean(F.relu(1.0 - real_pred))
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    return loss_real + loss_fake


def hinge_loss_generator(fake_pred):
    """Hinge loss for generator."""
    return -torch.mean(fake_pred)


def r1_penalty(discriminator, x_tn_real, x_t_real, class_vector,
               continuous_features, timestep_2, timestep_1, gamma=10.0):
    """
    R1 gradient penalty for discriminator regularization.
    Lighter than full spectral norm, works well with weight norm.
    """
    x_tn_real.requires_grad_(True)
    x_t_real.requires_grad_(True)

    real_pred = discriminator(
        x_tn_real, x_t_real, class_vector,
        continuous_features, timestep_2, timestep_1
    )

    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=[x_tn_real, x_t_real],
        create_graph=True
    )[0]

    penalty = grad_real.pow(2).sum(dim=(1, 2, 3)).mean()
    return gamma * penalty


# ===========================
# Example Training Loop
# ===========================

def example_training_step(generator, discriminator, real_batch, optimizer_g, optimizer_d):
    """
    Example training step showing how to use the discriminator.
    """
    x_real, class_vec, cont_feat = real_batch
    B = x_real.shape[0]
    device = x_real.device

    # Sample timesteps for consistency model
    t1 = torch.randint(0, 10, (B,), device=device).float()
    t2 = torch.randint(0, 10, (B,), device=device).float()

    # ===== Train Discriminator =====
    optimizer_d.zero_grad()

    # Generate fake samples
    with torch.no_grad():
        x_fake = generator(class_vec, cont_feat, t1)

    # Discriminator predictions
    real_pred = discriminator(x_real, x_real, class_vec, cont_feat, t2, t1)
    fake_pred = discriminator(x_fake, x_real, class_vec, cont_feat, t2, t1)

    # Hinge loss
    d_loss = hinge_loss_discriminator(real_pred, fake_pred)

    # R1 penalty (every N iterations)
    if torch.rand(1).item() < 0.1:  # 10% of iterations
        r1_loss = r1_penalty(
            discriminator, x_real, x_real, class_vec, cont_feat, t2, t1, gamma=10.0
        )
        d_loss += r1_loss

    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer_d.step()

    # ===== Train Generator =====
    optimizer_g.zero_grad()

    x_fake = generator(class_vec, cont_feat, t1)
    fake_pred, feats_fake, contrastive_loss = discriminator.forward_with_feats(
        x_fake, x_real, class_vec, cont_feat, t2, t1
    )

    # Generator hinge loss
    g_loss = hinge_loss_generator(fake_pred)

    # Add contrastive loss if available
    if contrastive_loss is not None:
        g_loss += discriminator.contrastive_weight * contrastive_loss

    # Optional: Feature matching loss
    with torch.no_grad():
        _, feats_real, _ = discriminator.forward_with_feats(
            x_real, x_real, class_vec, cont_feat, t2, t1
        )

    feat_match_loss = sum(
        F.l1_loss(ff, fr.detach())
        for ff, fr in zip(feats_fake, feats_real)
    )
    g_loss += 0.1 * feat_match_loss

    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer_g.step()

    return {
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
        'real_pred': real_pred.mean().item(),
        'fake_pred': fake_pred.mean().item(),
    }