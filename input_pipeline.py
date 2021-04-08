import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sys

# this code is inspired by the corresponding code from Vision Transformer repo
# https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py

if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000

def get_dataset_info(dataset, split):
  data_builder = tfds.builder(dataset)
  num_examples = data_builder.info.splits[split].num_examples
  return {
      'num_examples': num_examples,
  }


def get_data(*,
             H,
             mode,
             repeats,
             batch_size,
             shuffle_buffer=MAX_IN_MEMORY,
             tfds_data_dir=None,
             tfds_manual_dir=None):
  """Returns dataset for training/eval.

  Args:
    dataset: Dataset name. Additionally to the requirement that this dataset
      must be in tensorflow_datasets, the dataset must be registered in
      `DATASET_PRESETS` (specifying crop size etc).
    mode: Must be "train" or "test".
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    mixup_alpha: Coefficient for mixup combination. See 
      https://arxiv.org/abs/1710.09412
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    tfds_data_dir: Optional directory where tfds datasets are stored. If not
      specified, datasets are downloaded and in the default tfds data_dir on the
      local machine.
    inception_crop: If set to True, tf.image.sample_distorted_bounding_box()
      will be used. If set to False, tf.image.random_crop() will be used.
  """

  split_map = {'train': H.split_train, 'test': H.split_test} 
  split = split_map[mode]
  crop_size = H.image_size
  data_builder = tfds.builder(H.dataset, data_dir=tfds_data_dir)
  dataset_info = get_dataset_info(H.dataset, split)

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=tfds_manual_dir))
  data = data_builder.as_dataset(
      split=split,
      decoders={'image': tfds.decode.SkipDecoding()},
      shuffle_files=mode == 'train')
  decoder = data_builder.info.features['image'].decode_example

  def crop_center_and_resize(img, size):
    s = tf.shape(img)
    w, h = s[0], s[1]
    c = tf.maximum(w, h)
    wn, hn = h / c, w / c
    result = tf.image.crop_and_resize(tf.expand_dims(img, 0),
                                      [[(1 - wn) / 2, (1 - hn) / 2, wn, hn]],
                                      [0], [size, size])
    return tf.squeeze(result, 0)

  def _pp(data):
    im = decoder(data['image'])
    im = crop_center_and_resize(im, crop_size)
    im = (im - 127.5) / 127.5
    return {'image': im}

  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(min(dataset_info['num_examples'], shuffle_buffer))
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, crop_size, crop_size, 3])
    return data

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)


def prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
  if n_prefetch:
    ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter

def get_ds(H, mode):
    return get_data(
        H=H,
        mode=mode,
        repeats=None,
        batch_size=H.n_batch,
        shuffle_buffer=H.shuffle_buffer,
        tfds_data_dir=H.tfds_data_dir,
        tfds_manual_dir=H.tfds_manual_dir)
