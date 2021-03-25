import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class RBFKernelFn(tf.keras.layers.Layer):
    """
    RBF kernel for Gaussian processes.
    """
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(10.0 * self._length_scale) # 5.
        )


def build_model(data_dims=2):
    def mc_sampling(x):
        """
        Monte Carlo Sampling of the GP output distribution.
        :param x:
        :return:
        """
        samples = x.sample(20)
        return samples

    def mil_aggregation(x):
        """
        Aggregation of all instance probabilities of one bag.
        We multiply all probabilities p(y_i = 0| x_i) to obtain p(Y=0|{x_1, .. ,x_N})
        :param x:
        :return:
        """
        zero_inst_probabilities_samples = tf.ones_like(x) - x
        zero_bag_probability_samples = tf.math.reduce_prod(zero_inst_probabilities_samples, axis=1)
        return zero_bag_probability_samples

    def mc_integration(x):
        """
        Monte Carlo integration is basically replacing an integral with the mean of samples.
        Here we take the mean of the previously generated samples.
        :param x:
        :return:
        """
        out = tf.math.reduce_mean(x, axis=0)
        return out

    def probability_inversion(x):
        """
        P(Y=1|X) = 1 - P(Y=0|X)
        :param x:
        :return:
        """
        x = tf.ones_like(x) - x
        return x

    input = tf.keras.layers.Input(shape=[data_dims])
    x = tfp.layers.VariationalGaussianProcess(
        num_inducing_points=10,
        kernel_provider=RBFKernelFn(),
        event_shape=[1],  # output dimensions
        inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
            minval=0.0, maxval=1.0, seed=None
        ),
        jitter=10e-3,
        convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(np.array(1.0).astype(np.float32))),
        )(input)
    x = tf.keras.layers.Lambda(mc_sampling)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Lambda(mil_aggregation)(x)
    x = tf.keras.layers.Lambda(mc_integration)(x)
    output = tf.keras.layers.Lambda(probability_inversion)(x)
    model = tf.keras.Model(inputs=input, outputs=output, name="sgp_mil")

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=1.0),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model