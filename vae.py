from functools import partial
import itertools
from typing import Optional, Any

import numpy as np

import jax.numpy as jnp
from jax import random
from jax.util import safe_map

from flax import linen as nn
from einops import repeat, rearrange
from vae_helpers import (Conv1x1, Conv3x3, gaussian_sample, resize, parse_layer_string, pad_channels, get_width_settings,
                         gaussian_kl, Attention, recon_loss, sample, normalize, checkpoint, lecun_normal, has_attn, Block, EncBlock, identity)
import hps
from vqvae import Discriminator
map = safe_map

class Encoder(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        assert widths[str(int(x.shape[1]))] == H.width
        x = Conv3x3(H.width, dtype=H.dtype)(x)
        blocks = parse_layer_string(H.enc_blocks)
        n_blocks = len(blocks)
        activations = {}
        activations[x.shape[1]] = x  # Spatial dimension
        for res, down_rate in blocks:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            width = widths.get(str(res), H.width)
            block = EncBlock(H, res, width, down_rate or 1, use_3x3, last_scale=np.sqrt(1 / n_blocks))
            x = checkpoint(block.__call__, H, (x,))
            new_res = x.shape[1]
            new_width = widths.get(str(new_res), H.width)
            x = x if (x.shape[3] == new_width) else pad_channels(x, new_width)
            activations[new_res] = x
        return activations


class DecBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    mixin: Optional[int]
    n_blocks: int
    bias_cond: bool

    def setup(self):
        H = self.H
        width = self.width = get_width_settings(
            H.custom_width_str).get(str(self.res), H.width)
        use_3x3 = self.res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim if H.zdim != -1 else width
        self.enc   = Block(H, cond_width, self.zdim * 2,
                           use_3x3=use_3x3)
        self.prior = Block(H, cond_width, self.zdim * 2 + width,
                           use_3x3=use_3x3, last_scale=0.)
        self.z_proj = Conv1x1(
            width, kernel_init=lecun_normal(np.sqrt(1 / self.n_blocks)),
            dtype=self.H.dtype)
        self.resnet = Block(H, cond_width, width, residual=True,
                            use_3x3=use_3x3,
                            last_scale=np.sqrt(1 / self.n_blocks))
        self.z_fn = lambda x: self.z_proj(x)
        self.pre_layer = Attention(H) if has_attn(self.res, H) else identity
        self.bias_x = None
        self.expand_cond = self.mixin is not None
        if self.bias_cond:
            self.bias_x = self.param('bias_'+str(self.res), lambda key, shape: jnp.zeros(shape, dtype=H.dtype), (self.res, self.res, width))

    def sample(self, x, acts, rng):
        x = jnp.broadcast_to(x, acts.shape)
        qm, qv = jnp.split(self.enc(jnp.concatenate([x, acts], 3)), 2, 3)
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        z = gaussian_sample(qm, jnp.exp(qv), rng)
        kl = gaussian_kl(qm, pm, qv, pv)
        #print('sample', jnp.isnan(kl.mean()), jnp.isfinite(kl.mean()))
        return z, x + xpp, kl

    def sample_uncond(self, x, rng, t=None, lvs=None):
        pm, pv, xpp = jnp.split(self.prior(x), [self.zdim, 2 * self.zdim], 3)
        return (gaussian_sample(pm, jnp.exp(pv) * (t or 1), rng)
                if lvs is None else lvs, x + xpp)

    def add_bias(self, x, batch):
        bias = repeat(self.bias_x, '... -> b ...', b=batch)
        if x is None:
            return bias
        else:
            return x + bias 
    
    def forward(self, acts, rng, x=None):
        if self.expand_cond:
            # Assume width increases monotonically with depth
            x = resize(x[..., :acts.shape[3]], (self.res, self.res))
        if self.bias_cond:
            x = self.add_bias(x, acts.shape[0])
        x = self.pre_layer(x)
        #print('call', jnp.isnan(x.mean()), jnp.isfinite(x.mean()))
        z, x, kl = self.sample(x, acts, rng)
        x = self.resnet(x + self.z_fn(z))
        return z, x, kl
    
    def __call__(self, acts, rng, x=None):
        z, x, kl = self.forward(acts, rng, x=x)
        return x, dict(kl=kl)
    
    def get_latents(self, acts, rng, x=None):
        z, x, kl = self.forward(acts, rng, x=x)
        return x, dict(kl=kl, z=z)
    
    def forward_uncond(self, rng, n, t=None, lvs=None, x=None):
        if self.expand_cond:
            # Assume width increases monotonically with depth
            x = resize(x[..., :self.width], (self.res, self.res))
        if self.bias_cond:
            x = self.add_bias(x, n)
        x = self.pre_layer(x)
        z, x = self.sample_uncond(x, rng, t, lvs)
        return self.resnet(x + self.z_fn(z))

class Decoder(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            bias_cond = (mixin is not None and res <= H.no_bias_above) or (idx == 0)
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks), bias_cond=bias_cond))
            resos.add(res)
        self.dec_blocks = dec_blocks        
        self.gain = self.param('gain', lambda key, shape: jnp.ones(shape, dtype=self.H.dtype), (H.width,))
        self.bias = self.param('bias', lambda key, shape: jnp.zeros(shape, dtype=self.H.dtype), (H.width,))
        self.out_conv = Conv1x1(H.n_channels, dtype=self.H.dtype)
        self.final_fn = lambda x: self.out_conv(x * self.gain + self.bias)

    def __call__(self, activations, rng, get_latents=False):
        stats = []
        for idx, block in enumerate(self.dec_blocks):
            rng, block_rng = random.split(rng)
            acts = activations[block.res]
            f = block.__call__ if not get_latents else block.get_latents
            if idx == 0:
                x, block_stats = checkpoint(f, self.H, (acts, block_rng))
            else:
                x, block_stats = checkpoint(f, self.H, (acts, block_rng, x))
            stats.append(block_stats)
        return self.final_fn(x), stats

    def forward_uncond(self, n, rng, t=None):
        x = None
        for idx, block in enumerate(self.dec_blocks):
            t_block = t[idx] if isinstance(t, list) else t
            rng, block_rng = random.split(rng)
            x = block.forward_uncond(block_rng, n, t_block, x=x)
        return self.final_fn(x)

    def forward_manual_latents(self, n, latents, rng, t=None):
        x = None
        for idx, (block, lvs) in enumerate(itertools.zip_longest(self.dec_blocks, latents)):
            rng, block_rng = random.split(rng)
            x = block.forward_uncond(block_rng, n, t, lvs, x=x)
        return self.final_fn(x)

class VDVAE(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)
        self.discriminator = Discriminator(self.H)

    def __call__(self, x, rng=None, **kwargs):
        x_target = jnp.array(x) # is this clone?
        rng, uncond_rng = random.split(rng)
        px_z, stats = self.decoder(self.encoder(x), rng)
        ndims = np.prod(x.shape[1:])
        kl = sum((s['kl']/ ndims).sum((1, 2, 3)).mean() for s in stats)
        if not self.H.gan:
            loss = recon_loss(px_z, x_target)
            return dict(loss=loss + kl, recon_loss=loss, kl=kl), None
        else:
            uncond_sample = self.forward_uncond_samples(x_target.shape[0], uncond_rng) if self.H.uncond_sample else None
            gan_loss, contra = self.discriminator(x_target, px_z, uncond_sample) 
            loss = -kl + self.H.gan_coeff * gan_loss
            return dict(loss=loss, gan_loss=gan_loss, kl=kl), contra

    def forward_get_latents(self, x, rng):
        return self.decoder(self.encoder(x), rng, get_latents=True)[-1]

    def forward_uncond_samples(self, size, rng, t=None):
        return sample(self.decoder.forward_uncond(size, rng, t=t))

    def forward_samples_set_latents(self, size, latents, rng, t=None):
        return sample(self.decoder.forward_manual_latents(size, latents, rng, t=t))
