from functools import partial
import jax 
import jax.numpy as jnp
import flax
from flax import linen as nn
from jax import random
from jax import image
from flax.core import freeze, unfreeze
from einops import repeat
import hps
identity = lambda x: x

def gaussian_kl(mu1, mu2, logsigma1, logsigma2):
    return (-0.5 + logsigma2 - logsigma1
            + 0.5 * (jnp.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2)
            / (jnp.exp(logsigma2) ** 2))

def gaussian_sample(mu, sigma, rng):
    return mu + sigma * random.normal(rng, mu.shape)

Conv1x1 = partial(nn.Conv, kernel_size=(1, 1), strides=(1, 1))
Conv3x3 = partial(nn.Conv, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

def resize(img, shape):
    n, _, _, c = img.shape
    return image.resize(img, (n,) + shape + (c,), 'nearest')

def recon_loss(px_z, x):
    return jnp.abs(px_z - x).mean()

def sample(px_z):
    return jnp.round((jnp.clip(px_z, -1, 1) + 1) * 127.5).astype(jnp.uint8)

class Attention(nn.Module):
    H: hps.Hyperparams

    def setup(self):
        H = self.H
        self.attention = nn.SelfAttention(num_heads=H.num_heads, dtype=H.dtype)

    def __call__(self, x):
        res = x
        x = self.attention(normalize(x, self.H)) * np.sqrt(1 / x.shape[-1])
        return x + res

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, count = ss.split('x')
            layers.extend(int(count) * [(int(res), None)])
        elif 'm' in ss:
            res, mixin = ss.split('m')
            layers.append((int(res), int(mixin)))
        elif 'd' in ss:
            res, down_rate = ss.split('d')
            layers.append((int(res), int(down_rate)))
        else:
            res = int(ss)
            layers.append((res, 1))
    return layers

def pad_channels(t, new_width):
    return jnp.pad(t, (t.ndim - 1) * [(0, 0)] + [(0, new_width - t.shape[-1])])

def get_width_settings(s):
    mapping = {}
    if s:
        for ss in s.split(','):
            k, v = ss.split(':')
            mapping[k] = int(v)
    return mapping

def normalize(x, type=None, train=False):
    if type == 'group':
        return nn.GroupNorm()(x)
    elif type == 'batch':
        return nn.BatchNorm(use_running_average=not train, axis_name='batch')(x)
    else:
        return x

def checkpoint(fun, H, args):
    if H.checkpoint:
        return jax.checkpoint(fun)(*args)
    else:
        return fun(*args)

def astype(x, H):
    if H.dtype == 'bfloat16':
        return x.astype(jnp.bfloat16)
    elif H.dtype == 'float32':
        return x.astype(jnp.float32)
    else:
        raise NotImplementedError


# Want to be able to vary the scale of initialized parameters
def lecun_normal(scale):
    return nn.initializers.variance_scaling(
        scale, 'fan_in', 'truncated_normal')

class Block(nn.Module):
    H: hps.Hyperparams
    middle_width: int
    out_width: int
    down_rate: int = 1
    residual: bool = False
    use_3x3: bool = True
    last_scale: bool = 1.
    up: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        H = self.H
        residual = self.residual
        Conv1x1_ = partial(Conv1x1, dtype=H.dtype)
        Conv3x3_ = partial(Conv3x3 if self.use_3x3 else Conv1x1, dtype=H.dtype)
        if H.block_type == 'bottleneck':
            x_ = Conv1x1_(self.middle_width)(nn.gelu(x))
            x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
            x_ = Conv3x3_(self.middle_width)(nn.gelu(x_))
            x_ = Conv1x1_(
                self.out_width, kernel_init=lecun_normal(self.last_scale))(
                    nn.gelu(x_))
        elif H.block_type == 'diffusion':
            middle_width = int(self.middle_width / H.bottleneck_multiple)
            x_ = Conv3x3_(middle_width)(nn.gelu(x))
            x_ = Conv3x3_(
                self.out_width, kernel_init=lecun_normal(self.last_scale))(
                    nn.gelu(x_))
                
        out = x + x_ if residual else x_
        if self.down_rate > 1:
            if self.up:
                out = repeat(out, 'b h w c -> b (h x) (w y) c', x=self.down_rate, y=self.down_rate)
            else:
                window_shape = 2 * (self.down_rate,)
                out = nn.avg_pool(out, window_shape, window_shape)
        return out

def has_attn(res_, H):
    attn_res = [int(res) for res in H.attn_res.split(',') if len(res) > 0]
    return res_ in attn_res
    
class EncBlock(nn.Module):
    H: hps.Hyperparams
    res: int
    width: int
    down_rate: int = 1
    use_3x3: bool = True
    last_scale: bool = 1.
    up: bool = False

    def setup(self):
        H = self.H
        width, use_3x3 = self.width, self.use_3x3        
        middle_width = int(width * H.bottleneck_multiple)
        self.pre_layer = Attention(H) if has_attn(self.res, H) else identity
        self.block1 = Block(H, middle_width, width, self.down_rate or 1, True, use_3x3, up=self.up)
        
    def __call__(self, x, train=True):
        return self.block1(self.pre_layer(x), train=train)
