import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from utils import linear

_checked_scope = core_rnn_cell_impl._checked_scope

class RANCellv2(RNNCell):
  """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

  def __init__(self, num_units, input_size=None, activation=tanh, normalize=False, reuse=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._normalize = normalize
    self._reuse = reuse

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._num_units, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "ran_cell", reuse=self._reuse):
      with vs.variable_scope("gates"):
        c, h = state
        gates = tf.nn.sigmoid(linear([inputs, h], 2 * self._num_units, True, normalize=self._normalize))
        i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

      with vs.variable_scope("candidate"):
        content = linear([inputs], self._num_units, True, normalize=self._normalize)

      new_c = i * content + f * c
      new_h = self._activation(c)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      output = new_h
    return output, new_state
