from functools import partial
import jax.numpy as jnp
from flax import linen as nn
from jax import random, lax
from vae_helpers import parse_layer_string, pad_channels, get_width_settings, Conv1x1, Conv3x3, EncBlock, checkpoint, astype, recon_loss, sample
from einops import repeat, rearrange
from quantizer import VectorQuantizerEMA
import hps
import numpy as np

class BasicUnit(nn.Module):
    H: hps.Hyperparams
    module_type: str 
    min_res: int = 1

    @nn.compact
    def __call__(self, x, train=False):
        H = self.H
        module_type = self.module_type
        disc = False
        if module_type == "encoder":
            block_str = H.enc_blocks
            up = False
        elif module_type == "discriminator":
            block_str = H.dis_blocks
            up = False
            disc = True
        elif module_type == "decoder":
            block_str = H.dec_blocks
            up = True
        else:
            raise NotImplementedError
        widths = get_width_settings(H.custom_width_str)
        blocks = parse_layer_string(block_str)
        x = Conv3x3(widths[str(blocks[0][0])], dtype=H.dtype)(x)
        for res, down_rate in blocks:
            if res < self.min_res:
                continue
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            width = widths.get(str(res), H.width)
            block = EncBlock(H, res, width, down_rate or 1, use_3x3, last_scale=np.sqrt(1 / len(blocks)), up=up, disc=disc)
            x = checkpoint(partial(block.__call__, train=train), H, (x,)) #TODO: needs to be fixed for batchnorm
            new_res = x.shape[1]
            new_width = widths.get(str(new_res), H.width)
            if x.shape[3] < new_width:
                x = pad_channels(x, new_width)
            elif x.shape[3] > new_width:
                x = x[..., :new_width]
        if module_type == 'decoder':
            x = Conv1x1(H.n_channels, dtype=H.dtype)(x)
        return x


class VQVAE(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        widths = get_width_settings(H.custom_width_str)
        vq_dim = widths[str(H.vq_res)]
        self.encoder = BasicUnit(H, 'encoder', min_res=H.vq_res)
        self.quantizer = VectorQuantizerEMA(
                          embedding_dim=vq_dim,
                          num_embeddings=H.codebook_size,
                          commitment_cost=0.25,
                          decay=0.99,
                          cross_replica_axis='batch') # we set dtype = float32 here
        self.decoder = BasicUnit(H, 'decoder', min_res=H.vq_res)
        if H.gan:
            self.discriminator = Discriminator(H)

    def __call__(self, x, is_training=False, **kwargs):
        x_target = jnp.array(x)
        x = self.encoder(x, train=is_training)
        input_dtype = x.dtype
        rng = kwargs['rng'] if 'rng' in kwargs.keys() else None
        quant_dict = self.quantizer(x.astype(jnp.float32), is_training, rng=rng)
        kl = astype(quant_dict['loss'], self.H)
        px_z = self.decoder(astype(quant_dict['quantize'], self.H), train=is_training)
        if not self.H.gan:
            loss = recon_loss(px_z, x_target)
            return dict(loss=loss + kl, recon_loss=loss, kl=kl), None
        else:
            gan_loss, contra = self.discriminator(x_target, px_z, train=is_training) 
            loss = -kl + self.H.gan_coeff * gan_loss
            return dict(loss=loss, gan_loss=gan_loss, kl=kl), contra

    def forward_get_latents(self, x):
        x = self.encoder(x).astype(jnp.float32)
        return self.quantizer(x, is_training=False)['encoding_indices'].astype(jnp.int32)

    def forward_samples_set_latents(self, latents):
        latents = self.quantizer(None, is_training=False, encoding_indices=latents)
        px_z = self.decoder(astype(latents, self.H))
        return sample(px_z)

class Discriminator(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        self.discriminator = BasicUnit(H, 'discriminator')
        self.gan_head = Conv1x1(1, dtype=H.dtype)
        self.contra_head = Conv1x1(H.width, dtype=H.dtype)

    def __call__(self, real, fake, uncond_sample=None, eps=1e-5, train=False):           
        def normalize(x):
            return x / jnp.linalg.norm(x+eps, ord=2, axis=-1, keepdims=True)
        batch = [real, fake, uncond_sample] if self.H.uncond_sample else [real, fake]
        out = self.discriminator(jnp.concatenate(batch), train=train)
        gan = rearrange(self.gan_head(out), 'b 1 1 1 -> b')
        contra = rearrange(normalize(self.contra_head(out)), 'b 1 1 c -> b c')
        def bce_logits_loss(logit, label): 
            return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))
        
        real_labels = jnp.ones((real.shape[0],), dtype=jnp.int32)
        fake_labels = jnp.zeros((real.shape[0],), dtype=jnp.int32)
        labels = [real_labels, fake_labels]
        if self.H.uncond_sample:
            labels += [fake_labels]
        loss = bce_logits_loss(gan, jnp.concatenate(labels)).mean()
        return loss, contra        
    
    def contra_loss(self, x): 
        if self.H.uncond_sample:
            real, fake, _ = jnp.split(x, 3, axis=0)
        else:
            real, fake = jnp.split(x, 2, axis=0)
        real, fake = map(lambda x: lax.all_gather(x, 'batch'), (real, fake))
        x = jnp.einsum('aid,bjd->abij', real, fake)
        x = rearrange(x, 'dev_real dev_fake batch_real batch_fake -> (dev_real batch_real) (dev_fake batch_fake)')
        x = nn.log_softmax(x, axis=0)
        loss = - jnp.einsum('ii->', x) / x.shape[0]
        return loss
