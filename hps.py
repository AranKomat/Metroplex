import dataclasses
import jax
import jax.numpy as jnp
import json 

@dataclasses.dataclass(frozen=True)
class Hyperparams:
    # general
    hps: str = None
    run_opt: str = 'train' # or 'eval', 'get_latents',  
    model: str = 'vdvae' # or 'vqvae'
    desc: str = 'test' # description of run
    device_count: int = jax.local_device_count()
    host_count: int = jax.host_count()
    host_id: int = jax.host_id()
        
    # optimization
    adam_beta1: float = .9
    adam_beta2: float = .9
    lr: float = .0003
    ema_rate: float = 0.
    n_batch: int = 32   
    warmup_iters: float = 100.
    wd: float = 0.
    grad_clip: float = 200.
    skip_threshold: float = 400. # vdvae only        
    dtype: str = "float32"
    checkpoint: bool = True # gradient checkpointing
        
    # training misc.
    iters_per_ckpt: int = 25000
    iters_per_images: int = 10000
    iters_per_print: int = 1000
    iters_per_save: int = 10000
        
    # architecture
    width: int = 512 # width of the highest-res layer (should match with H.dec/enc_blocks below)
    zdim: int = 16 
    pre_layer: bool = False
    norm_type: str = "none"
    no_bias_above: int = 64
    enc_blocks: str = None
    dec_blocks: str = None
    dis_blocks: str = None
    '''
    Example:
    
    VDVAE (m = upsample, d = down)
    "dec_blocks": "1x2,4m1,4x1,8m4,8x1,16m8,16x1,32m16,32x1,64m32,64x1,128m64,128x1,256m128,256x1",
    "enc_blocks": "256x1,256d2,128x1,128d2,64x1,64d2,32x1,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x2",
    
    VQVAE (d = up (decoder) or down (encoder)) 
    "dec_blocks": "32x1,32d2,64x1,64d2,128x1,128d2,256x2",
    "enc_blocks": "256x1,256d2,128x1,128d2,64x1,64d2,32x2",
    
    '''
    custom_width_str: str = ''
    attn_res: str = ''
    bottleneck_multiple: float = 0.25
    vq_res: int = 32
    codebook_size: int = None
    uncond_sample: bool = False
    gan: bool = False
    gan_coeff: float = 1.
    contra_coeff: float = 1.
    block_type: str = "bottleneck"
        
    # visualization
    num_images_visualize: int = 6
    num_variables_visualize: int = 6
    
    # dataset
    n_channels: int = 3
    image_size: int = None
    split_train: str = 'train'
    split_test: str = 'validation'
    data_root: str = './'
    dataset: str = None
    shuffle_buffer: int = 50000
    tfds_data_dir: str = None
    #tfds_data_dir: Optional directory where tfds datasets are stored. If not
    #  specified, datasets are downloaded and in the default tfds data_dir on the
    #  local machine.
    tfds_manual_dir: str = None # Path to manually downloaded dataset
    '''
    The only preprocessing implemented rn is center crop and then resizing. 
    But this can be easily modified if necessary.
    '''
       
    # log 
    logdir: str = None
    log_wandb: bool = False
    project: str = 'vae'
    entity: str = None
    name: str = None

    # save & restore
    save_dir: str = './saved_models'
    restore_path: str = None
    restore_iter: int = 0 # setting this to 0 = new run from scratch
    
    # seed
    seed: int = 0
    seed_eval: int = None
    seed_init: int = None
    seed_sample: int = None
    seed_train: int = None


def parse_args_and_update_hparams(H, parser, s=None):
    parser_dict = vars(parser.parse_args(s))
    json_file = parser_dict['hps']
    with open(json_file) as f:
        json_dict = json.load(f)
    parser_dict.update(json_dict)
    return dataclasses.replace(H, **json_dict)

def add_vae_arguments(parser):
    for f in dataclasses.fields(Hyperparams):
        kwargs = (dict(action='store_true') if f.type is bool and not f.default else
                  dict(default=f.default, type=f.type))
        parser.add_argument(f'--{f.name}', **kwargs, **f.metadata)

    return parser
