import jax
import jax.lax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.core import freeze, unfreeze

def _l2_normalize(x, axis=None, eps=1e-12):
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)

lecun_normal = nn.initializers.lecun_normal 
zeros = nn.initializers.zeros

class Conv2D(nn.Module):
    features: int 
    kernel_size: tuple
    padding: str = 'SAME'
    strides: tuple = None
    use_bias: bool = True
    transposed: bool = False
    
    @nn.compact
    def __call__(
      self,
      inputs,
      update_stats = False,
      rng = None,
      **kwargs,
    ) -> jnp.ndarray:
        
        
        kernel_size = self.kernel_size
        strides = self.strides or (1,) * (inputs.ndim - 2)
        in_features = inputs.shape[-1]
        kernel_shape = kernel_size + (
            in_features, self.features)
        kernel = self.param('kernel', lecun_normal(), kernel_shape)
        bias = self.param('bias', zeros, (self.features,))
        
        dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
        if self.transposed:
            y = jax.lax.conv_transpose(
                inputs,
                kernel,
                strides,
                self.padding)
        else:
            y = jax.lax.conv_general_dilated(
                inputs,
                kernel,
                strides,
                self.padding,
                dimension_numbers=dimension_numbers)


        if self.use_bias:
            y = y + bias
        return y


class SNConv2D(nn.Module):
    features: int 
    kernel_size: tuple
    padding: str = 'SAME'
    strides: tuple = None
    use_bias: bool = True
    transposed: bool = False
    eps: float = 1e-12

    @nn.compact
    def __call__(
      self,
      inputs,
      update_stats = False,
      rng = None,
    ) -> jnp.ndarray:
        
        
        kernel_size = self.kernel_size
        strides = self.strides or (1,) * (inputs.ndim - 2)
        in_features = inputs.shape[-1]
        kernel_shape = kernel_size + (
            in_features, self.features)
        kernel = self.param('kernel', lecun_normal(), kernel_shape)
        bias = self.param('bias', zeros, (self.features,))
        
        def conv(inputs, kernel, sigma=None):
            dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
            if self.transposed:
                y = jax.lax.conv_transpose(
                    inputs,
                    kernel,
                    strides,
                    self.padding)
            else:
                y = jax.lax.conv_general_dilated(
                    inputs,
                    kernel,
                    strides,
                    self.padding,
                    dimension_numbers=dimension_numbers)

            if sigma is not None:
                y = y / sigma
                
            if self.use_bias:
                y = y + bias
            return y

        value = kernel
        value_shape = value.shape

        value = jnp.reshape(value, [-1, value.shape[-1]])

        initialized = self.has_variable('stats', 'u0')
        u0_ = self.variable("stats", "u0", jax.random.normal, rng, [1, value.shape[-1]])
        sigma_ = self.variable("stats", "sigma", lambda: jnp.ones((), dtype=value.dtype))

        u0 = u0_.value
        sigma = sigma_.value

        if not initialized:
            return conv(inputs, kernel)

        if not update_stats:
            sigma = jax.lax.stop_gradient(sigma)
            return conv(inputs, value.reshape(value_shape), sigma)


        v0 = _l2_normalize(jnp.matmul(u0, value.transpose([1, 0])), eps=self.eps)
        u0 = _l2_normalize(jnp.matmul(v0, value), eps=self.eps)

        u0 = jax.lax.stop_gradient(u0)
        v0 = jax.lax.stop_gradient(v0)

        sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]

        u0_.value = u0
        sigma_.value = sigma

        return conv(inputs, value.reshape(value_shape), sigma)
