"""
Dice Activation (Data Adaptive Individualized Activation)
TF2 适配版：将 TF1 的 variable_scope / get_variable / tf.layers.batch_normalization
改为 tf.keras.layers.Layer 形式，变量和 BN 层由 Layer 自动管理。
"""

import tensorflow as tf


class DiceLayer(tf.keras.layers.Layer):
    """
    Dice 激活层，按 axis 对输入做归一化后学习 alpha 和 beta。
    等价于原 TF1 版 dice() 函数，但变量随 Layer 对象自动创建/复用。
    """

    def __init__(self, axis=-1, epsilon=1e-9, name='dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.epsilon = epsilon
        # 原 TF1 版里用了独立的 BN（center=False, scale=False）
        self._bn = None
        self._alphas = None
        self._beta = None

    def build(self, input_shape):
        param_shape = [input_shape[-1]]
        self._alphas = self.add_weight(
            name='alpha',
            shape=param_shape,
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True,
        )
        self._beta = self.add_weight(
            name='beta',
            shape=param_shape,
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True,
        )
        self._bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            center=False,
            scale=False,
            epsilon=self.epsilon,
            name='bn',
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # 对最后一维做 batch_norm（不带 center/scale）
        x_normed = self._bn(inputs, training=training)
        x_p = tf.sigmoid(self._beta * x_normed)
        return self._alphas * (1.0 - x_p) * inputs + x_p * inputs


class PReLULayer(tf.keras.layers.Layer):
    """
    对应原 TF1 版 parametric_relu()。
    """

    def __init__(self, name='prelu', **kwargs):
        super().__init__(name=name, **kwargs)
        self._alphas = None

    def build(self, input_shape):
        self._alphas = self.add_weight(
            name='alpha',
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = self._alphas * (inputs - tf.abs(inputs)) * 0.5
        return pos + neg


# 为了兼容原有的 dice(_x, name='xxx') 调用方式，保留函数式入口
_dice_layer_cache = {}


def dice(_x, axis=-1, epsilon=1e-9, name=''):
    """
    函数式 Dice 入口。name 相同时复用同一个 DiceLayer（对应原 TF1 的 reuse=AUTO_REUSE）。
    """
    layer_key = ('dice', name)
    if layer_key not in _dice_layer_cache:
        _dice_layer_cache[layer_key] = DiceLayer(axis=axis, epsilon=epsilon, name='dice_' + name if name else 'dice')
    return _dice_layer_cache[layer_key](_x)


_prelu_layer_cache = {}


def parametric_relu(_x, name=''):
    layer_key = ('prelu', name)
    if layer_key not in _prelu_layer_cache:
        _prelu_layer_cache[layer_key] = PReLULayer(name='prelu_' + name if name else 'prelu')
    return _prelu_layer_cache[layer_key](_x)
