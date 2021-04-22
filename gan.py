import numpy as np
from functools import partial
import jax
from jax import tree_multimap
from jax import grad, lax, pmap
import jax.numpy as jnp
from flax import jax_utils
from flax import linen as nn

from vae_helpers import sample, recon_loss
from train_helpers import clip_grad_norm
from gan_helpers import Generator, Discriminator


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=0)
def training_step(H, data, optimizer, ema, state, rng):   
    if H.loss_type == 'NS':
        real_loss_fn = lambda logits: nn.softplus(-logits).mean() 
        fake_loss_fn = lambda logits: nn.softplus(logits).mean() 
    elif H.loss_type == 'HG':
        real_loss_fn = lambda logits: nn.relu(1 - logits).mean()
        fake_loss_fn = lambda logits: nn.relu(1 + logits).mean()
    pmean_fn = lambda x: jax.lax.pmean(x, axis_name='batch')
    
    def contrastive_loss(patch_embs, global_embs):
        loss = sum(map(lambda x: Discriminator(H).contra_loss(x, patch=H.patch_nce), patch_embs))
        loss += sum(map(lambda x: Discriminator(H).contra_loss(x, patch=False), global_embs))
        return loss / (len(patch_embs) + len(global_embs))
    
    def loss_g(params_g, params_d, batch, rng, variables_g, variables_d):
        (kl, fake_batch), variables_g = Generator(H).apply(
            {'params': params_g, **variables_g}, batch, training=True, mutable=list(variables_g.keys()))

        disc_batch = jnp.concatenate([batch, fake_batch])
        (gan_embs, patch_embs, global_embs), _ = Discriminator(H).apply(
            {'params': params_d, **variables_d}, disc_batch, training=True, mutable=list(variables_d.keys()))
                        
        stats = {}
        reals, fakes = Discriminator(H).gan_split(gan_embs)
        if H.loss_type == 'HG':
            assert NotImplementError
            #real_loss_fn = lambda x: -x.mean()
        stats['fake_real_loss'] = sum(map(real_loss_fn, fakes)) / len(fakes)
        stats['contra_loss_g'] = contrastive_loss(patch_embs, global_embs)
        stats['kl'] = kl  
        stats['recon_loss'] = recon_loss(fake_batch, batch)
        loss = sum(stats.values())
        return loss, (variables_g, variables_d, stats)

    def loss_d(params_d, params_g, batch, rng, variables_g, variables_d):
        (kl, fake_batch), _ = Generator(H).apply(
            {'params': params_g, **variables_g}, batch, training=True, mutable=list(variables_g.keys()))

        disc_batch = jnp.concatenate([batch, fake_batch])
        (gan_embs, patch_embs, global_embs), variables_d_ = Discriminator(H).apply(
            {'params': params_d, **variables_d}, disc_batch, training=True, mutable=list(variables_d.keys()))
        reals, fakes = Discriminator(H).gan_split(gan_embs)
        
        stats = {}
        stats['real_loss'] = sum(map(real_loss_fn, reals)) / len(reals)
        stats['fake_loss'] = sum(map(fake_loss_fn, fakes)) / len(fakes)
        stats['contra_loss_d'] = contrastive_loss(patch_embs, global_embs)

        if H.gamma > 0: # speedup with lazy regularization?
            def f(batch, params_d):
                (gan_embs, _, _), _ = Discriminator(H).apply(
                            {'params': params_d, **variables_d}, batch, training=True, mutable=list(variables_d.keys()))
                reals, fakes = Discriminator(H).gan_split(gan_embs)
                return reals[-1].sum()
            real_grad = jax.grad(f)(batch, params_d)
            stats['r1'] = jnp.square(real_grad).sum(axis=(1, 2, 3)).mean() * (0.5 * H.gamma)
        
        variables_d = variables_d_
        loss = sum(stats.values())
        return loss, (variables_g, variables_d, stats)
    
    optimizer_g, optimizer_d = optimizer['G'], optimizer['D']
    variables_g, variables_d = state['G'], state['D']
    
    (disc_loss, (variables_g, variables_d, stats)), grad = jax.value_and_grad(loss_d, has_aux=True)(
        optimizer_d.target, optimizer_g.target, data, rng, variables_g, variables_d)
    
    disc_loss, grad = map(pmean_fn, (disc_loss, grad))
    _, norm = clip_grad_norm(grad, 1)
    optimizer_d = optimizer_d.apply_gradient(grad, learning_rate=H.lr)
    stats_d = {**stats, **dict(disc_loss=disc_loss, disc_norm=norm)}
    #stats = {**stats, **dict(disc_loss=disc_loss, disc_norm=norm)}
    
    (gen_loss, (variables_g, variables_d, stats)), grad = jax.value_and_grad(loss_g, has_aux=True)(
        optimizer_g.target, optimizer_d.target, data, rng, variables_g, variables_d)
    gen_loss, grad = map(pmean_fn, (gen_loss, grad))
    _, norm = clip_grad_norm(grad, 1)
    optimizer_g = optimizer_g.apply_gradient(grad, learning_rate=H.lr)
    stats = {**stats_d, **stats, **dict(gen_loss=gen_loss, gen_norm=norm)}     
    #stats = {**stats, **dict(gen_loss=gen_loss, gen_norm=norm)}     
    
    if ema:
        f = lambda e, p: e * H.ema_rate + (1 - H.ema_rate) * p
        ema['params'] = tree_multimap(f, ema['params'], optimizer_g.target)
        ema['state'] = tree_multimap(f, ema['state'], variables_g)
        
    optimizer['G'], optimizer['D'] = optimizer_g, optimizer_d
    state['G'], state['D'] = variables_g, variables_d

    return optimizer, ema, state, stats
