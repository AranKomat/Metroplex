from typing import Any, Optional
from flax import linen as nn
import jax
import jax.numpy as jnp

# inspired from Haiku's corresponding code
# https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/moving_averages.py

class ExponentialMovingAverage(nn.Module):
  shape: list
  dtype: Any = jnp.float32
  decay: float = 0.

  def setup(self):
    shape = self.shape
    dtype = self.dtype
    self.hidden = self.variable("stats", "hidden", lambda: jnp.zeros(shape, dtype=dtype))
    self.average = self.variable("stats", "average", lambda: jnp.zeros(shape, dtype=dtype)) # how to deal with initialized?
    constant = lambda: jnp.zeros(shape, dtype=jnp.int32)
    self.counter = self.variable("stats", "counter", constant)

  def __call__(
      self,
      value: jnp.ndarray,
      update_stats: bool = True,
  ) -> jnp.ndarray:

    counter = self.counter.value + 1
    decay = jax.lax.convert_element_type(self.decay, value.dtype)
    one = jnp.ones([], value.dtype)
    hidden = self.hidden.value * decay + value * (one - decay)

    average = hidden
    average /= (one - jnp.power(decay, counter))
    if update_stats:
      self.counter.value = counter
      self.hidden.value = hidden
      self.average.value = average
    return average

# inspired from Haiku's corresponding code to Flax
# https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/nets/vqvae.py

class VectorQuantizerEMA(nn.Module):
  embedding_dim: int
  num_embeddings: int
  commitment_cost: float
  decay: float
  epsilon: float = 1e-5
  dtype: Any = jnp.float32
  cross_replica_axis: Optional[str] = None  
  initialized: bool = False

  @nn.compact
  def __call__(self, inputs, is_training, rng=None, encoding_indices=None):
    embedding_shape = [self.embedding_dim, self.num_embeddings]
    assert self.dtype == jnp.float32
    ema_cluster_size = ExponentialMovingAverage([self.num_embeddings], self.dtype, decay=self.decay)
    ema_dw = ExponentialMovingAverage(embedding_shape, self.dtype, decay=self.decay)
    initialized = self.has_variable('stats', 'embeddings')
    embeddings = self.variable("stats", "embeddings", nn.initializers.lecun_uniform(), rng, embedding_shape)
    
    def quantize(encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        w = embeddings.value.swapaxes(1, 0)
        w = jax.device_put(w)  # Required when embeddings is a NumPy array.
        return w[(encoding_indices,)]

    if encoding_indices is not None:
        return quantize(encoding_indices)
    
    if not initialized:
        hidden, counter, average = ema_cluster_size.hidden, ema_cluster_size.counter, ema_cluster_size.average
        hidden, counter, average = ema_dw.hidden, ema_dw.counter, ema_dw.average     
        return {
            "quantize": inputs,
            "loss": inputs.mean(),
        }
    
    flat_inputs = jnp.reshape(inputs, [-1, self.embedding_dim])
    distances = (
        jnp.sum(flat_inputs**2, 1, keepdims=True) -
        2 * jnp.matmul(flat_inputs, embeddings.value) +
        jnp.sum(embeddings.value**2, 0, keepdims=True))

    encoding_indices = jnp.argmax(-distances, 1)
    encodings = jax.nn.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)

    encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])
    quantized = quantize(encoding_indices)
    e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - inputs)**2)

    if is_training:
      cluster_size = jnp.sum(encodings, axis=0)
      if self.cross_replica_axis:
        cluster_size = jax.lax.psum(
            cluster_size, axis_name=self.cross_replica_axis)
      updated_ema_cluster_size = ema_cluster_size(cluster_size, update_stats=is_training)

      dw = jnp.matmul(flat_inputs.T, encodings)
      if self.cross_replica_axis:
        dw = jax.lax.psum(dw, axis_name=self.cross_replica_axis)
      updated_ema_dw = ema_dw(dw, update_stats=is_training)

      n = jnp.sum(updated_ema_cluster_size)
      updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                  (n + self.num_embeddings * self.epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_dw / jnp.reshape(updated_ema_cluster_size, [1, -1]))

      embeddings.value = normalised_updated_ema_w
      loss = self.commitment_cost * e_latent_loss

    else:
      loss = self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    avg_probs = jnp.mean(encodings, 0)
    if self.cross_replica_axis:
      avg_probs = jax.lax.pmean(avg_probs, axis_name=self.cross_replica_axis)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

    return {
        "quantize": quantized,
        "loss": loss,
        "perplexity": perplexity,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "distances": distances,
    }
