"""
unet.py
-------
Architecture U-Net pour le modele de diffusion (DDPM).

Le U-Net est le reseau de debruitage au coeur du DDPM.
Il prend en entree une image bruitee + un encoding du timestep,
et predit le bruit a retirer.

Architecture basee sur le papier de Ho et al. (2020) :
  - Encodeur descendant avec blocs residuels
  - Bottleneck avec attention
  - Decodeur montant avec skip connections
  - Embedding sinusoidal du timestep (comme les Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------------
# Embedding du timestep
# ------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """
    Embedding sinusoidal du timestep t.

    Meme principe que l'encoding positionnel des Transformers :
    on transforme un scalaire t en un vecteur haute dimension
    avec des sinus et cosinus a differentes frequences.
    Cela permet au reseau de distinguer facilement les niveaux de bruit.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ------------------------------------------------------------------
# Blocs de base
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Bloc residuel avec injection du timestep.

    Deux convolutions 3x3 avec GroupNorm et SiLU.
    Le timestep est injecte entre les deux convolutions via
    une projection lineaire (permet au reseau d'adapter son
    comportement en fonction du niveau de bruit).
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection avec projection si les dimensions changent
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Injection du timestep
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    Self-attention sur les feature maps.

    On l'utilise seulement aux resolutions les plus basses (ex: 16x16)
    pour capturer les dependances globales sans exploser en memoire.
    Complexite O(n^2) ou n = H*W, donc prohibitif a haute resolution.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        normed = self.norm(x)

        qkv = self.qkv(normed).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention : Q * K^T / sqrt(d)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Reduction de resolution par convolution stride 2."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Augmentation de resolution par interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ------------------------------------------------------------------
# U-Net complet
# ------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net pour le debruitage dans le DDPM.

    Architecture en U avec skip connections :
      Encodeur: [Conv -> ResBlock -> ResBlock -> (Attention) -> Down] x n_levels
      Bottleneck: ResBlock -> Attention -> ResBlock
      Decodeur: [Up -> Concat skip -> ResBlock -> ResBlock -> (Attention)] x n_levels

    Le timestep est injecte dans chaque ResBlock via un embedding sinusoidal.
    L'attention est ajoutee seulement aux niveaux specifies (typiquement
    les resolutions les plus basses) pour economiser la memoire.
    """

    def __init__(self, in_channels=3, base_channels=64,
                 channel_mults=(1, 2, 4), n_res_blocks=2,
                 attention_levels=None):
        super().__init__()

        if attention_levels is None:
            attention_levels = [len(channel_mults) - 1]

        time_emb_dim = base_channels * 4

        # Embedding du timestep
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Convolution initiale
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encodeur
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()

            for _ in range(n_res_blocks):
                block_list.append(ResBlock(in_ch, out_ch, time_emb_dim))
                if level in attention_levels:
                    attn_list.append(SelfAttention(out_ch))
                else:
                    attn_list.append(nn.Identity())
                in_ch = out_ch
                channels.append(in_ch)

            self.down_blocks.append(nn.ModuleDict({
                'blocks': block_list,
                'attns': attn_list,
            }))

            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample(in_ch))
                channels.append(in_ch)
            else:
                self.down_samples.append(nn.Identity())

        # Bottleneck
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = SelfAttention(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)

        # Decodeur
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            out_ch = base_channels * channel_mults[level]
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()

            for i in range(n_res_blocks + 1):
                skip_ch = channels.pop()
                block_list.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim))
                if level in attention_levels:
                    attn_list.append(SelfAttention(out_ch))
                else:
                    attn_list.append(nn.Identity())
                in_ch = out_ch

            self.up_blocks.append(nn.ModuleDict({
                'blocks': block_list,
                'attns': attn_list,
            }))

            if level > 0:
                self.up_samples.append(Upsample(in_ch))
            else:
                self.up_samples.append(nn.Identity())

        # Convolution finale
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: image bruitee (B, C, H, W)
            t: timestep (B,) entier

        Returns:
            prediction du bruit (B, C, H, W)
        """
        t_emb = self.time_embed(t)
        h = self.conv_in(x)
        skips = [h]

        # Encodeur
        for level, (block_dict, downsample) in enumerate(
            zip(self.down_blocks, self.down_samples)
        ):
            for block, attn in zip(block_dict['blocks'], block_dict['attns']):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)

            if level < len(self.down_blocks) - 1:
                h = downsample(h)
                skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decodeur
        for level, (block_dict, upsample) in enumerate(
            zip(self.up_blocks, self.up_samples)
        ):
            for block, attn in zip(block_dict['blocks'], block_dict['attns']):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)

            if level < len(self.up_blocks) - 1:
                h = upsample(h)

        return self.conv_out(h)
