import numpy as np
from functools import partial
import os
import time
from jax import grad, lax
import jax.numpy as jnp
from flax import jax_utils
from flax import linen as nn

import hps 
from vae_helpers import sample, get_width_settings, pad_channels
from vae_helpers import astype
from quantizer import VectorQuantizerEMA
from einops import rearrange, repeat
from specnorm import SNConv2D as SNConv
from specnorm import Conv2D

SNConv1x1 = partial(SNConv, kernel_size=(1, 1))
_downsample = lambda x: nn.avg_pool(x, (2, 2), (2, 2))                
activation = lambda x: nn.leaky_relu(x, negative_slope=0.2)

def discriminator_widths(H):
    return [int(width) for width in get_width_settings(H.custom_width_str).values()]
    
def encoder_widths(H):
    return [int(width) for res, width in get_width_settings(H.custom_width_str).items() if int(res) >= H.vq_res]
    
def decoder_widths(H):
    widths = [int(width) for res, width in get_width_settings(H.custom_width_str).items() if int(res) >= H.vq_res]
    widths.reverse()
    return widths

class Block(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x, update_stats, rng):
        t = x

        x = Conv(features=self.features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
        x = activation(x)

        x = Conv(features=self.features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
        x = activation(x)

        t = Conv(features=self.features, kernel_size=(1, 1))(t, update_stats=training, rng=rng)
        x = (x + t) * (1 / np.sqrt(2))
        return x


class DiscriminatorBody(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x, training=False, **kwargs):
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        Conv = partial(SNConv, padding='SAME', use_bias=False) 
        acts = {}
        features_list = discriminator_widths(self.H)
        for idx, features in enumerate(features_list):
            downsample = _downsample if idx < len(features_list) - 1 else lambda x: x
            
            # regular blocks
            for _ in range(self.H.blocks_per_res - 1):
                x = Block(features=features)(x, update_stats=training, rng=rng)
            
            if x.shape[1] in [32, 16, 8, 4]:
                acts[x.shape[1]] = x 
            
            # downsampling block
            t = x
            x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
            x = activation(x)
            
            if idx == len(features_list) - 1:
                x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
            else:
                x = Conv(features=features, kernel_size=(4, 4), strides=(2, 2))(x, update_stats=training, rng=rng)
            x = activation(x)

            t = Conv(features=features, kernel_size=(1, 1))(t, update_stats=training, rng=rng)
            t = downsample(t)
            x = (x + t) * (1 / np.sqrt(2))

        acts[1] = SNConv(features=1, kernel_size=(1, 1), strides=(4, 4), padding='VALID', use_bias=False)(x, update_stats=training, rng=rng)
        return acts    

class Discriminator(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        self.discriminator = DiscriminatorBody(H)
        convert = lambda x: [val for val in x.split(',')]
        inner_width = discriminator_widths(H)[0]
        self.gan_resos, self.contra_resos = map(convert, (self.H.gan_resos, self.H.contra_resos))
        self.gan_head = {res: SNConv1x1(1) for res in self.gan_resos}
        self.contra_head_global = {res: SNConv1x1(32) for res in self.contra_resos if res != 1}
        self.contra_head_global2 = {res: SNConv1x1(inner_width) for res in self.contra_resos if res != 1}
        self.contra_head_patch = {res: SNConv1x1(inner_width) for res in self.contra_resos}

    def __call__(self, data, training=False, **kwargs):
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        acts = self.discriminator(data, training=training, rng=rng)
        
        gan_embs = []
        for res in self.gan_resos: 
            gan_embs += [self.gan_head[res](acts[int(res)], update_stats=training, rng=rng)]
            
        patch_embs, global_embs = [], []
        for res in self.contra_resos: 
            patch_embs += [self.contra_head_patch[res](acts[int(res)], update_stats=training, rng=rng)]
            if int(res) != 1:
                emb_global = self.contra_head_global[res](acts[int(res)], update_stats=training, rng=rng)
                emb_global = rearrange(emb_global, 'b h w c -> b 1 1 (h w c)')   
                emb_global = self.contra_head_global2[res](emb_global, update_stats=training, rng=rng)
                global_embs += [emb_global]
        
        def normalize(x):
            return x / jnp.linalg.norm(x+1e-6, ord=2, axis=-1, keepdims=True)
        
        patch_embs = tuple(map(normalize, patch_embs))
        global_embs = tuple(map(normalize, global_embs))        
        return gan_embs, patch_embs, global_embs 
    
    def gan_split(self, gan_embs):
        reals, fakes = [], []
        for emb in gan_embs: 
            if self.H.uncond_sample:
                raise NotImplementedError
                real, fake, _ = jnp.split(emb, 3, axis=0)
            else:
                real, fake = jnp.split(emb, 2, axis=0)
            real, fake = map(lambda x: rearrange(x, 'b h w 1 -> (b h w)'), (real, fake))
            reals += [real]
            fakes += [fake]
        return reals, fakes
    
    def contra_loss(self, x, patch): 
        if self.H.uncond_sample:
            real, fake, _ = jnp.split(x, 3, axis=0)
        else:
            real, fake = jnp.split(x, 2, axis=0)
        real, fake = map(lambda x: lax.all_gather(x, 'batch'), (real, fake))
        if patch:
            f = lambda x: x.reshape((-1, min(1024, np.prod(x.shape[:-1])), x.shape[-1]))
            real, fake = map(f, (real, fake))
            x = jnp.einsum('aid,ajd->ija', real, fake)
            x = nn.log_softmax(x, axis=0)
            loss = - jnp.einsum('ii...->', x) / (x.shape[0] * x.shape[2])
            return loss
        else:
            x = jnp.einsum('aihwd,bjhwd->abijhw', real, fake)
            x = rearrange(x, 'dev_real dev_fake batch_real batch_fake ... -> (dev_real batch_real) (dev_fake batch_fake) ...')
            x = nn.log_softmax(x, axis=0)
            loss = - jnp.einsum('ii...->', x) / (x.shape[0] * np.prod(x.shape[2:]))
            return loss
        
class Encoder(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x, training=False, **kwargs):
        Conv = partial(SNConv, padding='SAME', use_bias=False) 
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        x = Conv(features=64, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
        features_list = encoder_widths(self.H)
        
        for idx, features in enumerate(features_list):
            downsample = _downsample if idx < len(features_list) - 1 else lambda x: x
            
            # regular blocks
            for _ in range(self.H.blocks_per_res - 1):
                x = Block(features=features)(x, update_stats=training, rng=rng)
            
            # downsampling block
            t = x
            x = activation(x)
            x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
            
            x = activation(x)
            if idx < len(features_list) - 1:
                x = Conv(features=features, kernel_size=(4, 4), strides=(2, 2))(x, update_stats=training, rng=rng)
            else:
                x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)             
            t = Conv(features=features, kernel_size=(1, 1))(t, update_stats=training, rng=rng)
            t = downsample(t)
            x = (x + t) * (1 / np.sqrt(2))

        return x

class Decoder(nn.Module):
    H: hps.Hyperparams

    @nn.compact
    def __call__(self, x, training=False, **kwargs):
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        Conv = partial(Conv2D, padding='SAME', use_bias=False) 
        _upsample = lambda x: repeat(x, 'b h w c -> b (h x) (w y) c', x=2, y=2)
        features_list = decoder_widths(self.H)
        for idx, features in enumerate(features_list):
            upsample = _upsample if idx < len(features_list) - 1 else lambda x: x
            
            # regular blocks
            for _ in range(self.H.blocks_per_res - 1):
                x = Block(features=features)(x, update_stats=training, rng=rng)
            
            # downsampling block
            t = x
            x = activation(x)
            x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
            
            x = activation(x)
            x = Conv(features=features, kernel_size=(3, 3))(x, update_stats=training, rng=rng)
            
            x = upsample(x + t[..., :x.shape[-1]])
                    
        x = Conv(features=self.H.n_channels, kernel_size=(1, 1))(x, update_stats=training, rng=rng)
        return x
    
class Generator(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        vq_dim = widths[str(H.vq_res)]
        self.encoder = Encoder(H)
        self.quantizer = VectorQuantizerEMA(
                          embedding_dim=vq_dim,
                          num_embeddings=H.codebook_size,
                          commitment_cost=0.25,
                          decay=0.99,
                          cross_replica_axis='batch') # we set dtype = float32 here
        self.decoder = Decoder(H)

    def __call__(self, x, training=False, **kwargs):
        x_target = jnp.array(x)
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        x = self.encoder(x, train=training, rng=rng)
        quant_dict = self.quantizer(x.astype(jnp.float32), training, rng=rng)
        kl = astype(quant_dict['loss'], self.H)
        px_z = self.decoder(astype(quant_dict['quantize'], self.H), train=training, rng=rng)
        return kl, px_z

    def forward_get_latents(self, x):
        x = self.encoder(x).astype(jnp.float32)
        return self.quantizer(x, is_training=False)['encoding_indices'].astype(jnp.int32)

    def forward_samples_set_latents(self, latents):
        latents = self.quantizer(None, is_training=False, encoding_indices=latents)
        px_z = self.decoder(astype(latents, self.H))
        return sample(px_z)
