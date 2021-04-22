import numpy as np
from functools import partial
import os
import time
from train_helpers import (set_up_hyperparams, load_vaes, accumulate_stats,
                           save_model, linear_warmup,
                           clip_grad_norm, p_write_images, get_latents_loop)
from jax import tree_map, tree_multimap, device_get
from jax import grad, lax, pmap
from jax import random
import jax.numpy as jnp
from vae_helpers import sample
from utils import model_fn
from vae_helpers import astype
import input_pipeline
from flax import jax_utils
from gan import training_step as gan_training_step

def training_step(H, data, optimizer, ema, state, rng):
    # this is for vq/vdvae only
    
    def loss_fun(params, state):
        (stats, contra), state = model_fn(H).apply({'params': params, **state}, astype(data, H), rng=rng, is_training=True, mutable=list(state.keys()))
        loss = stats['loss']
        stats = {k: v.astype(jnp.float32) for k, v in stats.items()}
        return loss, (stats, state)

    gradval, (stats, state) = grad(loss_fun, has_aux=True)(optimizer.target, state)
    stats, gradval = lax.pmean((stats, gradval), 'batch')
    
    gradval, grad_norm = clip_grad_norm(gradval, H.grad_clip)
    
    loss_nans = jnp.any(jnp.isnan(stats['loss']))
    kl_nans = jnp.any(jnp.isnan(stats['kl']))
    stats.update(loss_nans=loss_nans, kl_nans=kl_nans)

    learning_rate = H.lr * linear_warmup(H.warmup_iters)(optimizer.state.step)

    # only update if no rank has a nan and if the grad norm is below a specific
    # threshold
    def skip_update(gradval):
        # Only increment the step
        return optimizer.replace(
            state=optimizer.state.replace(step=optimizer.state.step + 1)), ema
    def update(gradval):
        optimizer_ = optimizer.apply_gradient(
            gradval, learning_rate=learning_rate)
        e_decay = H.ema_rate
        if ema:
            ema_ = tree_multimap(
                lambda e, p: e * e_decay + (1 - e_decay) * p, ema, optimizer.target)
        else:
            ema_ = ema
        return optimizer_, ema_
    if H.model == 'vdvae':
        skip = (loss_nans | kl_nans | ((H.skip_threshold != -1)
                                    & ~(grad_norm < H.skip_threshold)))
        optimizer, ema = lax.cond(skip, skip_update, update, gradval)
        stats.update(skipped_updates=skip, grad_norm=grad_norm)
    else:
        optimizer, ema = update(gradval)
        stats.update(grad_norm=grad_norm)
    return optimizer, ema, state, stats
# Would use donate_argnums=(3, 4) here but compilation never finishes
p_training_step = pmap(training_step, 'batch', static_broadcasted_argnums=0)

# def samples_set_latents():
    
def train_loop(H, optimizer, ema, state, logprint):
    rng = random.PRNGKey(H.seed_train)
    if H.gan:
        iterate = int(optimizer['G'].state.step[0])    
    else:
        iterate = int(optimizer.state.step[0])    
    ds_train = input_pipeline.get_ds(H, mode='train')
    ds_valid = input_pipeline.get_ds(H, mode='test')
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    training_step = p_training_step if not H.gan else gan_training_step
    for data in input_pipeline.prefetch(ds_train, n_prefetch=2): # why 2?
        rng, iter_rng = random.split(rng)
        iter_rng = random.split(iter_rng, H.device_count)   
        t0 = time.time()
        optimizer, ema, state, training_stats = training_step(
            H, data['image'], optimizer, ema, state, iter_rng)
        training_stats = device_get(
            tree_map(lambda x: x[0], training_stats))
        training_stats['iter_time'] = time.time() - t0
        stats.append(training_stats)
        if (iterate % H.iters_per_print == 0
                or (iters_since_starting in early_evals) or (iters_since_starting < H.early_evals)):
            logprint(model=H.desc, type='train_loss',
                      lr=H.lr * float(
                          linear_warmup(H.warmup_iters)(iterate)),
                      step=iterate,
                      **accumulate_stats(stats, H.iters_per_print))

        if (iterate % H.iters_per_images == 0
                or (iters_since_starting in early_evals)):
            p_write_images(H, optimizer, ema, state, ds_valid,
                          f'{H.save_dir}/samples-{iterate}.png', logprint)

        iterate += 1
        iters_since_starting += 1
        if iterate % H.iters_per_save == 0:
            logprint(model=H.desc, type='train_loss',
                      step=iterate,
                      **accumulate_stats(stats, H.iters_per_print))
            fp = os.path.join(H.save_dir, 'latest')
            logprint(f'Saving model@ {iterate} to {fp}')
            save_model(fp, optimizer, ema, state, H)

        if iterate % H.iters_per_ckpt == 0:
            save_model(os.path.join(H.save_dir, f'iter-{iterate}'),
                        optimizer, ema, state, H)

def main():
    H, logprint = set_up_hyperparams()
    optimizer, ema, state = load_vaes(H, logprint)
    if H.run_opt == 'train':
        train_loop(H, optimizer, ema, state, logprint)
    elif H.run_opt == 'get_latents_train':
        get_latents_loop(H, optimizer, ema, state, logprint, mode='train')
    elif H.run_opt == 'get_latents_test':
        get_latents_loop(H, optimizer, ema, state, logprint, mode='test')
    elif H.run_opt == 'eval':
        raise NotImplementedError
        
if __name__ == "__main__":
    main()
