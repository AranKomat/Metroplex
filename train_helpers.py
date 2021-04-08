import torch
import numpy as np
import dataclasses
import argparse
import os
import subprocess
import time

from jax.interpreters.xla import DeviceArray
from tensorflow.io import gfile
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, mkdir_p, model_fn

from jax import lax
import jax
from jax import random
import jax.numpy as jnp
from jax.util import safe_map
from flax import jax_utils
from flax.training import checkpoints
from flax.optim import Adam
from functools import partial
from PIL import Image
from jax import lax, pmap
from vae_helpers import astype, sample
import input_pipeline
from einops import rearrange
map = safe_map



def save_model(path, optimizer, ema, state, H):
    optimizer = jax_utils.unreplicate(optimizer)
    checkpoints.save_checkpoint(path, optimizer, optimizer.state.step)
    if ema:
        ema = jax_utils.unreplicate(ema)
        checkpoints.save_checkpoint(path + '_ema', ema, optimizer.state.step)
    if state:
        state = jax_utils.unreplicate(state)
        checkpoints.save_checkpoint(path + '_state', state, optimizer.state.step)
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def load_vaes(H, logprint):
    rng = random.PRNGKey(H.seed_init)
    init_rng, init_eval_rng = random.split(rng)
    init_eval_rng, init_emb_rng = random.split(init_eval_rng)
    init_batch = jnp.zeros((1, H.image_size, H.image_size, H.n_channels))
    variables = model_fn(H).init({'params': init_rng}, init_batch, rng=init_eval_rng)
    state, params = variables.pop('params')
    #print(jax.tree_map(jnp.shape, state))
    del variables
    ema = params if H.ema_rate != 0 else {}
    optimizer = Adam(weight_decay=H.wd, beta1=H.adam_beta1,
                     beta2=H.adam_beta2).create(params)
    if H.restore_path and H.restore_iter > 0:
        logprint(f'Restoring vae from {H.restore_path}')
        optimizer = checkpoints.restore_checkpoint(H.restore_path, optimizer, step=H.restore_iter)
        if ema:
            ema = checkpoints.restore_checkpoint(H.restore_path + '_ema', ema, step=H.restore_iter)
        if state:
            state = checkpoints.restore_checkpoint(H.restore_path + '_state', state, step=H.restore_iter)

    total_params = 0
    for p in jax.tree_flatten(optimizer.target)[0]:
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    optimizer = jax_utils.replicate(optimizer)
    if ema:
        ema = jax_utils.replicate(ema)        
    if state:
        state = jax_utils.replicate(state)
    return optimizer, ema, state

def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['loss_nans', 'kl_nans', 'skipped_updates']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'loss':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['loss'] = np.mean(vals)
            z['loss_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = (stats[-1][k] if len(stats) < frequency
                    else np.mean([a[k] for a in stats[-frequency:]]))
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z

def linear_warmup(warmup_iters):
    return lambda i: lax.min(1., i / warmup_iters)

def setup_save_dirs(H):
    save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(save_dir)
    logdir = os.path.join(save_dir, 'log')
    return dataclasses.replace(
        H,
        save_dir=save_dir,
        logdir=logdir,
    )

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    H = parse_args_and_update_hparams(H, parser, s=s)
    H = setup_save_dirs(H)
    log = logger(H.logdir)
    if H.log_wandb:
        import wandb
        def logprint(*args, pprint=False, **kwargs):
            if len(args) > 0: log(*args)
            wandb.log({k: np.array(x) if type(x) is DeviceArray else x for k, x in kwargs.items()})
        wandb.init(project='vae', entity=H.entity, name=H.name, config=dataclasses.asdict(H))
    else:
        logprint = log
        for i, k in enumerate(sorted(dataclasses.asdict(H))):
            logprint(type='hparam', key=k, value=getattr(H, k))
    np.random.seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    H = dataclasses.replace(
        H,
        seed_init  =H.seed,
        seed_sample=H.seed + 1,
        seed_train =H.seed + 2 + H.host_id,
        seed_eval  =H.seed + 2 + H.host_count + H.host_id,
    )
    return H, logprint

def clip_grad_norm(g, max_norm):
    # Simulates torch.nn.utils.clip_grad_norm_
    g, treedef = jax.tree_flatten(g)
    total_norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g)))
    clip_coeff = jnp.minimum(max_norm / (total_norm + 1e-6), 1)
    g = [clip_coeff * g_ for g_ in g]
    return treedef.unflatten(g), total_norm

def get_latents_step(H, optimizer, ema, state, data, rng):
    params = ema or optimizer.target
    ema_apply = partial(model_fn(H).apply, {'params': params, **state}) 
    forward_get_latents = partial(ema_apply, method=model_fn(H).forward_get_latents)
    zs = forward_get_latents(data, rng)
    return forward_samples_set_latents(zs)

p_get_latents_step = pmap(get_latents_step, 'batch', static_broadcasted_argnums=0)
    
def get_latents_loop(H, optimizer, ema, state, logprint, mode):
    rng = random.PRNGKey(H.seed_train)
    iterate = 0    
    ds = input_pipeline.get_ds(H, mode=mode)
    stats = []
    for data in input_pipeline.prefetch(ds, n_prefetch=2):
        rng, iter_rng = random.split(rng)
        iter_rng = random.split(iter_rng, H.device_count)   
        t0 = time.time()
        latents = p_get_latents_step(H, optimizer, ema, state, data['image'], iter_rng)
        save_latents(latents, data['text'])
        stats.append({'iter_time': time.time() - t0})
        if (iterate % H.iters_per_print == 0
                or (iters_since_starting in early_evals)):
            logprint(model=H.desc, type='get_latents',
                      step=iterate,
                      **accumulate_stats(stats, H.iters_per_print))
        iterate += 1

def write_images(H, optimizer, ema, state, viz_batch):
    rng = random.PRNGKey(H.seed_sample)
    params = ema or optimizer.target
    ema_apply = partial(model_fn(H).apply,
                        {'params': params, **state}) 
    forward_get_latents = partial(ema_apply, method=model_fn(H).forward_get_latents)
    forward_samples_set_latents = partial(
        ema_apply, method=model_fn(H).forward_samples_set_latents)

    batches = [sample(viz_batch)]
    mb = viz_batch.shape[0]
    if H.model == 'vdvae':
        forward_uncond_samples = partial(
        ema_apply, method=model_fn(H).forward_uncond_samples)
        zs = [s['z'] for s in forward_get_latents(viz_batch, rng)]
        lv_points = np.floor(
            np.linspace(
                0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
        for i in lv_points:
            batches.append(forward_samples_set_latents(mb, zs[:i], rng, t=0.1))
        for t in [1.0, 0.9, 0.8]:
            batches.append(forward_uncond_samples(mb, rng, t=t))
    else:
        zs = forward_get_latents(viz_batch)
        batches.append(forward_samples_set_latents(zs))
    im = jnp.stack(batches)
    return im

def p_write_images(H, optimizer, ema, state, ds, fname, logprint):
    for x in input_pipeline.prefetch(ds, n_prefetch=2):
        viz_batch = x['image']
        fun = pmap(write_images, 'batch', static_broadcasted_argnums=0)
        im = np.array(fun(H, optimizer, ema, state, viz_batch))
        im = rearrange(im, 'device height batch ... -> (device batch) height ...')[:H.num_images_visualize]
        im = rearrange(im, 'batch height h w c -> (height h) (batch w) c')
        logprint(f'printing samples to {fname}')
        Image.fromarray(im).save(fname)
        break
